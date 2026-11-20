'''
Mock SEDM sky observations for relative flux calibration on the OH bands, 2026-11-16 (second round).

SEDM is being reinstalled at LMU's 2.1 m on the Wendelstein (1.84 km). Only Mauna Kea MODTRAN curves exist, so the atmosphere is mimicked by
matching slant columns: Wendelstein's surface pressure is ~1.33x Mauna Kea's, so Mauna Kea at ZD 40/50/60 has Wendelstein's O2 and Rayleigh
columns at ZD ~0/31/48, and Mauna Kea PWV p stands for Wendelstein PWV 1.33p (0-27 mm for the 0-20 mm grid). Ozone and aerosols are not matched.

Steps (each writes to sedm/out/):
  cache    line-by-line transmission of every line, and the LSF-smoothed transmission a continuum source sees, for the whole PWV x ZD grid
  figure   per-R decomposition of the R = 75/100/150 sky into the groups that vary together (v' levels, O2, OI, Na, H) and band labels
  pwv      how the band-flux ratios of the same-v' pairs change with PWV, against what a star's continuum sees
  mocks    mock observations (random airglow, atmosphere, LSF error, continuum, instrument colour, photon noise) and the recovery of the
           instrument's relative response with (A) fixed-position feature fits and (B) a forward-model fit of the whole 610-995 nm sky
           (Bk: PWV known, Bf: PWV fitted, Bh: PWV, hot-population scale and Tvib fitted as well)

Run from RAGNAR_new/: python sedm/mock_sedm.py cache figure pwv mocks [--nmock 300] [--ncpu 16]
'''
import os, sys, json, time, argparse, itertools
import numpy as np
import pandas as pd
import os.path as osp
from scipy.special import erf
from scipy.optimize import least_squares

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, osp.dirname(HERE))
import analyze_sedm as AS
from dev.species_intensities import get_pgopher_linelist, get_O2_at_T, R_HOT, T_HOT
from dev.make_spectra import load_transmission, make_intensity_ratios

OUT = osp.join(HERE, 'out')
CACHE = osp.join(OUT, 'mock_cache.npz')
FWHM2SIG = AS.FWHM2SIG
P_RATIO = 1.33 # Wendelstein / Mauna Kea surface pressure
MK_PWVS = np.arange(21) # Mauna Kea PWV grid [mm]; Wendelstein PWV = P_RATIO x
MK_ZDS = [40, 50, 60]
RS = [75, 100, 150]
LO, HI = 540.0, 1005.0 # pixel grid range [nm]
FIT = (610.0, 995.0) # forward-model fit range [nm]
PX_PER_FWHM = 3
LAM_EVAL = [650, 686, 730, 776, 838, 884, 940, 990] # where the recovered response is compared with the truth

def wst_zd(zd_mk):
    ''' Wendelstein ZD with the slant column of Mauna Kea at zd_mk '''
    c = P_RATIO*np.cos(np.radians(zd_mk))
    return 0.0 if c >= 1 else float(np.degrees(np.arccos(c)))

# The band features (cent = 0 line lists, flux_features_R*_cent0.csv), grouped by band. A band's flux is the sum of the
# amplitudes of its features, fitted together at fixed positions and widths with a linear background
BANDS = {75:  {'7-2': [687.39], '7-3': [885.73], '8-3': [730.00], '8-4': [939.87], '6-2': [838.19]},
         100: {'6-1': [654.96], '7-2': [686.06], '7-3': [884.07], '8-3': [728.86], '8-4': [936.88, 945.21], '6-2': [838.29], '9-4': [775.48], '9-5': [996.11]},
         150: {'6-1': [655.76], '7-2': [685.24, 693.21], '7-3': [878.08, 883.48, 891.03], '8-3': [727.26, 733.08], '8-4': [932.71, 938.02, 946.45],
               '6-2': [835.27, 842.11], '9-4': [773.94], '9-5': [995.23]}}
PAIRS = [('8-3', '8-4'), ('7-2', '7-3'), ('6-1', '6-2'), ('9-4', '9-5')]
# Nuisance features fitted together with a band but not counted in its flux. The O2 b(0-1) blend at 865 nm sits inside the 7-3 window and
# varies independently of OH: without it, x0.5/x1.5 O2 moves the 7-3 flux by +27/-24%
NUISANCE = {75: {'7-3': [865.32]}, 100: {'7-3': [864.91]}, 150: {'7-3': [864.83]}}

#########################
### Fast line model ###
#########################

class Lines:
    ''' All lines in 334-1015 nm (analyze_sedm.line_table order) with a vectorised copy of make_OH_intensities '''
    def __init__(self):
        h, c, kB = 6.62607015e-34, 2.9979e8, 1.380649e-23
        A = get_pgopher_linelist()
        self.A_index = A.index
        v = A["v'"].to_numpy().astype(int)
        E = A['Eupper'].to_numpy()*1e2*h*c/kB # [K]
        Ev = pd.Series(E).groupby(v).transform('min').to_numpy()
        self.v, self.Erot, self.Evib = v, E - Ev, Ev
        self.gA = A['A'].to_numpy()*2*(2*A["J'"].to_numpy() + 1)
        self.uE = {k: np.unique(self.Erot[v == k]) for k in np.unique(v)}
        self.uEv = np.unique(Ev)
        self.f = np.where(v == 8, 1.50, np.where(v == 9, 1.35, 1.0))
        L = AS.line_table()
        self.L = L
        self.pos = L.Position.to_numpy()
        sp = L.Species.to_numpy()
        self.oh, self.o2 = sp == 'OH', sp == 'O2'
        self.at = ~(self.oh | self.o2)
        self.oh_idx = pd.Index(A.index).get_indexer(L.orig_id[self.oh])
        assert np.all(self.oh_idx >= 0)
        # chemistry slot per line, as in get_spectra_and_peaks: OH, O2 b, (O2 a), green OI, red OI, Na; H goes with red OI
        slot = {'OH': 0, 'O2': 1, 'OI(1S)': 3, 'OI(1D)': 4, 'NaI': 5, 'H': 4}
        self.slot = np.array([slot[s] for s in sp])
        self.I_at = L.Intensity.to_numpy()*self.at
        lab = L.Label.astype(str).to_numpy()
        self.d2, self.d1 = lab == 'NaI(D2)', lab == 'NaI(D1)'
        # O2 b(0-1) intensities on a Trot grid, interpolated later
        self.TrO2 = np.arange(150, 241)
        ids = L.orig_id[self.o2].to_numpy()
        self.IO2 = np.array([get_O2_at_T(T).Intensity.reindex(ids).to_numpy() for T in self.TrO2])
        # groups that vary together, and bands
        vg = L.vgroup.to_numpy()
        names = {"OI(1S)": '[O I] 557.7', 'OI(1D)': '[O I] 630/636', 'NaI': 'Na D', 'H': 'H', 'O2 b(v=0)': 'O₂ b(0-1)'}
        self.group = np.array([names.get(g, g.replace("OH v'=", "v′=")) for g in vg])
        tr = L.trans.to_numpy()
        self.band = np.array([t[1:t.index(')')] if o else g for t, o, g in zip(tr, self.oh, self.group)])

    def oh_intensities(self, Tr, Tv, rh):
        r_hot = np.clip(R_HOT*rh, 0, 0.5); T_hot = T_HOT*(Tv/1e4)
        Qv = np.sum(np.exp(-self.uEv/Tv))
        I = np.empty(len(self.v))
        for k, uE in self.uE.items():
            m = self.v == k
            E = self.Erot[m]
            I[m] = self.gA[m]*((1 - r_hot[k])*np.exp(-E/Tr)/np.sum(np.exp(-uE/Tr)) + r_hot[k]*np.exp(-E/T_hot[k])/np.sum(np.exp(-uE/T_hot[k])))\
                   *np.exp(-self.Evib[m]/Tv)/Qv*self.f[m]
        return 1000*I/I.sum()

    def intensities(self, Tr = 195, Tv = 1e4, rh = 1.0, chem = None):
        ''' Intrinsic line intensities [R] at the given conditions; chem is the output of make_intensity_ratios (None = all 1) '''
        I = self.I_at.copy()
        I[self.oh] = self.oh_intensities(Tr, Tv, rh)[self.oh_idx]
        j = np.clip(np.searchsorted(self.TrO2, Tr) - 1, 0, len(self.TrO2) - 2); w = (Tr - self.TrO2[j])/(self.TrO2[j+1] - self.TrO2[j])
        I[self.o2] = (1 - w)*self.IO2[j] + w*self.IO2[j+1]
        I[self.d2] *= Tr/195; I[self.d1] *= 195/Tr # Na D line ratio as in get_spectra_and_peaks
        if chem is not None:
            I = I*np.asarray(chem)[self.slot]
        return I*1e3

#####################
### Pixels and LSF ###
#####################

def pixels(R):
    n = int(round(PX_PER_FWHM*R*np.log(HI/LO)))
    e = np.exp(np.linspace(np.log(LO), np.log(HI), n + 1))
    return e, np.sqrt(e[1:]*e[:-1])

def lsf_matrix(edges, lam, R, scale = 1.0):
    ''' [pixel, line] mean flux density in each pixel [per nm] of a unit-flux Gaussian line, sigma = scale*lam/(R*FWHM2SIG) '''
    s = scale*lam/(R*FWHM2SIG)*np.sqrt(2)
    c = erf((edges[:, None] - lam[None, :])/s[None, :])
    return 0.5*(c[1:] - c[:-1])/np.diff(edges)[:, None]

#################################
### Transmission cache (step 'cache') ###
#################################

def build_cache(lines):
    out = {'T_line': np.zeros((len(MK_PWVS), len(MK_ZDS), len(lines.pos)), np.float32)}
    for R in RS:
        out[f'T_cont_{R}'] = np.zeros((len(MK_PWVS), len(MK_ZDS), len(pixels(R)[1])), np.float32)
    for (i, p), (j, z) in itertools.product(enumerate(MK_PWVS), enumerate(MK_ZDS)):
        t0 = time.time()
        T, trans = AS.line_transmission(lines.L, float(p), z)
        AS._T.clear() # one curve at a time, they are big
        out['T_line'][i, j] = T
        nm, tc = trans['nm'].to_numpy()[::-1], trans['combin'].to_numpy()[::-1]
        for R in RS:
            _, lam = pixels(R)
            s = lam/(R*FWHM2SIG)
            lo, hi = np.searchsorted(nm, lam - 5*s), np.searchsorted(nm, lam + 5*s)
            out[f'T_cont_{R}'][i, j] = [np.average(tc[a:b], weights = np.exp(-0.5*((nm[a:b] - l)/q)**2)) for a, b, l, q in zip(lo, hi, lam, s)]
        print(f'PWV {p} ZD {z}: {time.time() - t0:.1f} s', flush = True)
    np.savez(CACHE, **out)

#####################################
### Simulated observation and fits ###
#####################################

class Sim:
    def __init__(self, lines, R, templates = True):
        self.lines, self.R = lines, R
        self.edges, self.lam = pixels(R)
        self.dlam = np.diff(self.edges)
        self.x = (self.lam - 700)/300
        c = np.load(CACHE)
        self.T_line, self.T_cont = c['T_line'], c[f'T_cont_{R}']
        self.M = lsf_matrix(self.edges, lines.pos, R)
        self.bands = BANDS[R]
        self.nuisance = True # fit the NUISANCE features with their bands
        # templates for the forward model: one per v' (3-9), plus O2 b, [O I] 630/636 and H, at nominal LSF, Tvib = 1e4 K, r_hot_scale = 1
        self.TG = np.arange(170, 231, 5.0)
        self.RHG = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.6]) # hot-population scale
        self.TVG = np.array([7000, 8500, 10000, 11500, 13000.0]) # Tvib, which sets the hot-population temperature
        self.vs = sorted({int(g[3:]) for g in lines.group if g.startswith('v′=')})
        self.comp = [f"v′={k}" for k in self.vs] + ['O₂ b(0-1)', '[O I] 630/636', 'H']
        self.cidx = np.array([self.comp.index(g) if g in self.comp else -1 for g in lines.group])
        self.fit = (self.lam > FIT[0]) & (self.lam < FIT[1])
        self.IT = np.array([lines.intensities(T, tv, rh) for T in self.TG for rh in self.RHG for tv in self.TVG]) # [(Trot, rh, Tvib), line]
        self.band_names = sorted({b for b in lines.band if len(b) == 3 and b[1] == '-'}) # OH bands only, e.g. '8-3'
        self.tsets = {'base': self.build_templates(self.T_line)} if templates else {}
        self.P = np.polynomial.legendre.legvander((self.lam[self.fit] - 800)/200, 3) # free continuum, observed through the response

    def build_templates(self, T_line):
        ''' templates on the fit pixels for every (Trot, rh, Tvib) x (PWV, ZD), with the given line transmissions [PWV, ZD, line] '''
        IT = self.IT
        B = np.zeros((len(IT), len(MK_PWVS), len(MK_ZDS), len(self.comp), self.fit.sum()), np.float32)
        bidx = {b: self.lines.band == b for b in self.band_names}
        BF = np.zeros((len(IT), len(MK_PWVS), len(MK_ZDS), len(self.band_names)))
        Mf = self.M[self.fit]
        for i, j in itertools.product(range(len(MK_PWVS)), range(len(MK_ZDS))):
            ITt = IT*T_line[i, j][None, :]
            for k in range(len(self.comp)):
                m = self.cidx == k
                B[:, i, j, k] = ITt[:, m] @ Mf[:, m].T
            BF[:, i, j] = np.array([ITt[:, bidx[b]].sum(1) for b in self.band_names]).T
        shape = (len(self.TG), len(self.RHG), len(self.TVG))
        return B.reshape(shape + B.shape[1:]), BF.reshape(shape + BF.shape[1:])

    def observe(self, Tr, Tv, rh, chem, lsf, ip, iz, C0, beta, s1, s2, g, rng):
        ''' A mock spectrum in counts/(g dlam) units, i.e. response x R/nm, and its variance. g = photons per rayleigh, inf = noiseless '''
        I = self.lines.intensities(Tr, Tv, rh, chem)*self.T_line[ip, iz]
        M = self.M if lsf == 1.0 else lsf_matrix(self.edges, self.lines.pos, self.R, lsf)
        sky = M @ I + C0*(self.lam/500)**beta*self.T_cont[ip, iz]
        S = np.exp(s1*self.x + s2*self.x**2)
        mu = S*sky
        var = mu/(min(g, 5e3)*self.dlam) # noiseless mocks are weighted as g = 5000
        if np.isfinite(g):
            mu = rng.poisson(mu*g*self.dlam)/(g*self.dlam)
        return mu, var, I

    def band_flux(self, d, var, band, lsf = 1.0, win = 2.0, tc = None):
        '''
        (A) sum of the fixed-position, fixed-width Gaussian amplitudes of a band's features, with a linear background over +-win FWHM (+-1 FWHM was 3-4x noisier).
        With tc (the transmission a continuum source sees, on the pixel grid) the background is (b0 + b1 x)*tc, so it follows telluric bands like O2 B.
        '''
        ws = np.array(self.bands[band])
        nu = np.array(NUISANCE[self.R].get(band, [])) if self.nuisance else np.array([])
        wa = np.r_[ws, nu]
        fw = win*ws.mean()/self.R
        m = (self.lam > wa.min() - fw) & (self.lam < wa.max() + fw)
        G = lsf_matrix(self.edges[np.r_[m, False] | np.r_[False, m]], wa, self.R, lsf)
        bg = np.ones(m.sum()) if tc is None else tc[m]
        X = np.column_stack([G, bg, bg*(self.lam[m] - ws.mean())])
        w = 1/np.sqrt(var[m])
        c = np.linalg.lstsq(X*w[:, None], d[m]*w, rcond = None)[0]
        return c[:len(ws)].sum()

    def band_lam(self, band):
        return float(np.mean(self.bands[band]))

    def templates(self, Tr, pwv, iz, rh = 1.0, Tv = 1e4, tset = 'base'):
        ''' multilinear interpolation of the templates in Trot, hot-population scale, Tvib and (Mauna Kea) PWV '''
        B_, BF_ = self.tsets[tset]
        idx, wts = [], []
        for g, v in [(self.TG, Tr), (self.RHG, rh), (self.TVG, Tv), (MK_PWVS, pwv)]:
            u = np.interp(v, g, np.arange(len(g))); i = min(int(u), len(g) - 2); f = u - i
            idx.append((i, i + 1)); wts.append((1 - f, f))
        Bt, BFt = 0.0, 0.0
        for c in itertools.product((0, 1), repeat = 4):
            w = np.prod([wts[k][c[k]] for k in range(4)])
            if w == 0:
                continue
            ii = tuple(idx[k][c[k]] for k in range(4))
            Bt = Bt + w*B_[ii][iz].astype(float); BFt = BFt + w*BF_[ii][iz]
        return Bt, BFt

    def forward_fit(self, d, var, iz, pwv = None, hot = False, tset = 'base', mask = None):
        '''
        (B) fit d = S(lambda) * sum_k a_k B_k(lambda; Trot, PWV) + Legendre(3) over 610-995 nm, S = exp(s1 x + s2 x^2) with x = (lambda - 700)/300.
        The linear amplitudes (one per v' level and species, continuum) are solved for at every step. PWV is fitted when pwv is None.
        tset picks the template set (perturbed transmissions), mask = (lo, hi) leaves that wavelength range out of the fit.
        '''
        dd, w = d[self.fit], 1/np.sqrt(var[self.fit])
        if mask is not None:
            w = w*~((self.lam[self.fit] > mask[0]) & (self.lam[self.fit] < mask[1])) # zero weight inside the mask
        xf = self.x[self.fit]
        Tc = self.T_cont[:, iz][:, self.fit]
        def solve(th):
            Tr, s1, s2 = th[:3]; p = pwv if pwv is not None else th[3]
            rh, Tv = (th[-2], th[-1]) if hot else (1.0, 1e4)
            Bt, BFt = self.templates(Tr, p, iz, rh, Tv, tset)
            S = np.exp(s1*xf + s2*xf**2)
            # the sky continuum is seen through the same atmosphere (O2 A and B bands, H2O) and the same response
            j = int(np.clip(p, 0, MK_PWVS[-1] - 1e-6)); b = np.clip(p, 0, MK_PWVS[-1]) - j
            T = (1 - b)*Tc[j] + b*Tc[min(j + 1, len(MK_PWVS) - 1)]
            X = np.column_stack([(S[:, None]*Bt.T), (S*T)[:, None]*self.P])
            c = np.linalg.lstsq(X*w[:, None], dd*w, rcond = None)[0]
            return c, X, BFt
        def res(th):
            c, X, _ = solve(th)
            return (X @ c - dd)*w
        th0 = [195.0, 0.0, 0.0] + ([] if pwv is not None else [6.0]) + ([1.0, 1e4] if hot else [])
        lb = [170.0, -3, -3] + ([] if pwv is not None else [0.0]) + ([0.6, 7000] if hot else [])
        ub = [230.0, 3, 3] + ([] if pwv is not None else [20.0]) + ([1.6, 13000] if hot else [])
        r = least_squares(res, th0, bounds = (lb, ub), x_scale = [10, 0.1, 0.1] + ([] if pwv is not None else [2]) + ([0.2, 1000] if hot else []), diff_step = 1e-3)
        c, X, BFt = solve(r.x)
        a = c[:len(self.comp)]
        # recovered band fluxes (before the response): a_v times the template's band flux
        BFrec = np.array([a[self.comp.index(f"v′={b.split('-')[0]}")]*BFt[k] for k, b in enumerate(self.band_names)])
        return r.x, BFrec, r.cost*2/len(dd)

##################
### Step: figure ###
##################

def figure_data(lines):
    ''' Per R: the share of the fiducial sky by group, on the RAGNAR mean-spectrum grid, and labels for the brightest band features '''
    c = np.load(CACHE)
    ip, iz = 6, 1 # Mauna Kea PWV 6 (Wendelstein 8 mm), ZD 50 (Wendelstein ~31 deg)
    I = lines.intensities()*c['T_line'][ip, iz]
    grid = np.arange(340, 1000.01, 0.5)
    groups = [f"v′={k}" for k in range(3, 10)] + ['O₂ b(0-1)', '[O I] 557.7', '[O I] 630/636', 'Na D', 'H']
    out = {}
    for R in RS:
        s = lines.pos/(R*FWHM2SIG)
        K = np.exp(-0.5*((grid[:, None] - lines.pos[None, :])/s[None, :])**2)/(np.sqrt(2*np.pi)*s[None, :]) # [grid, line] per nm
        tot = K @ I
        share = {g: (K[:, lines.group == g] @ I[lines.group == g])/np.maximum(tot, 1e-12) for g in groups}
        labels = []
        for b in sorted(set(lines.band)):
            m = lines.band == b
            sb = K[:, m] @ I[m]
            k = int(np.argmax(sb))
            if sb[k] >= 1.0 and sb[k]/tot[k] >= 0.25 and 345 < grid[k] < 998:
                labels.append(dict(band = b, wav = float(grid[k]), peak = round(float(sb[k]), 2), share = round(float(sb[k]/tot[k]), 3)))
        npz = np.load(f'/tigress/cj1223/spectra/sedm_R{R}_0_wst_20261116.npz')
        w, S = npz['wav_range'], npz['specs'].astype(float)*1e3
        idx = np.searchsorted(w, grid)
        out[R] = dict(mean = S.mean(0)[idx].round(3).tolist(), lo = S.min(0)[idx].round(3).tolist(), hi = S.max(0)[idx].round(3).tolist(),
                      share = {g: v.round(3).tolist() for g, v in share.items()}, labels = labels, n_spec = len(S))
    json.dump(dict(wav = grid.round(1).tolist(), groups = groups, R = out), open(osp.join(OUT, 'figure_groups.json'), 'w'), ensure_ascii = False)
    for R in RS:
        print(R, [(l['band'], l['wav'], l['share']) for l in out[R]['labels']])

###############
### Step: pwv ###
###############

def pwv_study(lines):
    ''' Fiducial airglow, no response, noiseless: band fluxes and pair ratios against PWV for each ZD, and the same for a star's continuum '''
    rows = []
    for R in RS:
        sim = Sim(lines, R)
        for (ip, p), (iz, z) in itertools.product(enumerate(MK_PWVS), enumerate(MK_ZDS)):
            d, var, _ = sim.observe(195, 1e4, 1.0, None, 1.0, ip, iz, 10.0, 0.0, 0.0, 0.0, np.inf, None)
            F = {b: sim.band_flux(d, var, b) for b in sim.bands}
            for b in sim.bands:
                k = np.argmin(np.abs(sim.lam - sim.band_lam(b)))
                rows.append(dict(R = R, pwv_mk = p, pwv_wst = round(P_RATIO*p, 2), zd_mk = z, zd_wst = round(wst_zd(z), 1), band = b, flux = F[b], t_cont = float(sim.T_cont[ip, iz, k])))
        print('pwv study R', R, flush = True)
    D = pd.DataFrame(rows)
    D.to_csv(osp.join(OUT, 'pwv_bands.csv'), index = False)
    P = []
    for (R, z), g in D.groupby(['R', 'zd_mk']):
        g = g.set_index(['pwv_mk', 'band'])
        for b1, b2 in PAIRS:
            if b1 not in BANDS[R] or b2 not in BANDS[R]:
                continue
            for p in MK_PWVS:
                r = g.loc[(p, b1), 'flux']/g.loc[(p, b2), 'flux']; r0 = g.loc[(0, b1), 'flux']/g.loc[(0, b2), 'flux']
                t = g.loc[(p, b1), 't_cont']/g.loc[(p, b2), 't_cont']; t0 = g.loc[(0, b1), 't_cont']/g.loc[(0, b2), 't_cont']
                P.append(dict(R = R, zd_mk = z, zd_wst = round(wst_zd(z), 1), pair = f'{b1}/{b2}', pwv_wst = round(P_RATIO*p, 2), sky = r/r0 - 1, star = t/t0 - 1))
    P = pd.DataFrame(P)
    P.to_csv(osp.join(OUT, 'pwv_pairs.csv'), index = False)
    print(P[(P.R == 100) & (P.zd_mk == 50) & P.pwv_wst.isin([0, 5.32, 10.64, 19.95, 26.6])].round(4).to_string())

#################
### Step: mocks ###
#################

NOISE = [np.inf, 5000.0, 500.0, 50.0] # detected photons per rayleigh; ~5 per spaxel in 30 min (see report), so ~1000, 100, 10 sky spaxels
_SIM = {}

def draw(rng):
    return dict(Tr = rng.uniform(175, 215), Tv = rng.uniform(7500, 12500), rh = rng.uniform(0.7, 1.4), alpha = rng.uniform(0.05, 0.95),
                lsf = rng.normal(1, 0.03), ip = int(rng.integers(0, len(MK_PWVS))), iz = int(rng.integers(0, len(MK_ZDS))),
                C0 = rng.uniform(5, 25), beta = rng.uniform(-0.5, 0.5), s1 = rng.normal(0, 0.3), s2 = rng.normal(0, 0.3), seed = int(rng.integers(2**31)))

def one_mock(args):
    R, k, cond = args
    sim = _SIM[R]
    np.random.seed(cond['seed']); chem = make_intensity_ratios(cond['alpha']) # it draws the red-line scatter from np.random
    rng = np.random.default_rng(cond['seed'])
    Strue = lambda l: np.exp(cond['s1']*(np.asarray(l) - 700)/300 + cond['s2']*((np.asarray(l) - 700)/300)**2)
    rows = []
    for g in NOISE:
        d, var, I = sim.observe(cond['Tr'], cond['Tv'], cond['rh'], chem, cond['lsf'], cond['ip'], cond['iz'], cond['C0'], cond['beta'], cond['s1'], cond['s2'], g, rng)
        row = dict(R = R, mock = k, g = g, **{c: cond[c] for c in ['Tr', 'Tv', 'rh', 'alpha', 'lsf', 'C0', 'beta', 's1', 's2']},
                   pwv_wst = P_RATIO*MK_PWVS[cond['ip']], zd_wst = wst_zd(MK_ZDS[cond['iz']]))
        # (A) band-flux ratios of the same-v' pairs, against the reference ratio of fiducial airglow seen through (i) the true atmosphere,
        # (ii) the true atmosphere but PWV off by N(0, 1.5 mm) at Wendelstein, (iii) a fixed atmosphere (Wendelstein 8 mm, ZD 31)
        F = {b: sim.band_flux(d, var, b) for b in sim.bands}
        dp = rng.normal(0, 1.5/P_RATIO)
        for b1, b2 in PAIRS:
            if b1 not in sim.bands:
                continue
            rho_true = Strue(sim.band_lam(b1))/Strue(sim.band_lam(b2))
            rr = F[b1]/F[b2]
            row[f'A_{b1}/{b2}'] = rr/(REF[R][(b1, b2)](MK_PWVS[cond['ip']], cond['iz'])*rho_true) - 1
            row[f'A_pwv1.5_{b1}/{b2}'] = rr/(REF[R][(b1, b2)](MK_PWVS[cond['ip']] + dp, cond['iz'])*rho_true) - 1
            row[f'A_fixed_{b1}/{b2}'] = rr/(REF[R][(b1, b2)](6, 1)*rho_true) - 1
        # (B) forward model, with PWV known and with PWV fitted
        Itrue = I
        for case, pw, hot in [('Bk', float(MK_PWVS[cond['ip']]), False), ('Bf', None, False), ('Bh', None, True)]:
            th, BFrec, chi2 = sim.forward_fit(d, var, cond['iz'], pw, hot)
            Srec = np.exp(th[1]*(np.array(LAM_EVAL) - 700)/300 + th[2]*((np.array(LAM_EVAL) - 700)/300)**2)
            for l, sr, st in zip(LAM_EVAL, Srec, Strue(LAM_EVAL)):
                row[f'{case}_S{l}'] = sr/st - 1
            for b1, b2 in PAIRS:
                if b1 in sim.bands:
                    Srec_f = lambda l: np.exp(th[1]*(l - 700)/300 + th[2]*((l - 700)/300)**2)
                    l1, l2 = sim.band_lam(b1), sim.band_lam(b2)
                    row[f'{case}_{b1}/{b2}'] = (Srec_f(l1)/Srec_f(l2))/(Strue(l1)/Strue(l2)) - 1
            row[f'{case}_Tr'] = th[0]; row[f'{case}_chi2'] = chi2
            if pw is None:
                row[f'{case}_pwv_wst'] = P_RATIO*th[3]
            if hot:
                row['Bh_rh'], row['Bh_Tv'] = th[-2], th[-1]
            for b in ['8-3', '8-4', '7-2', '7-3', '6-2', '9-4']:
                kb = sim.band_names.index(b)
                row[f'{case}_flux_{b}'] = BFrec[kb]/Itrue[sim.lines.band == b].sum() - 1
        rows.append(row)
    return rows

REF = {}
def build_refs(lines):
    ''' Reference pair ratios: fiducial airglow, unit response, noiseless, through each grid atmosphere; interpolated in PWV '''
    for R in RS:
        sim = _SIM[R]
        tab = {}
        for (ip, p), iz in itertools.product(enumerate(MK_PWVS), range(len(MK_ZDS))):
            d, var, _ = sim.observe(195, 1e4, 1.0, None, 1.0, ip, iz, 10.0, 0.0, 0.0, 0.0, np.inf, None)
            tab[(ip, iz)] = {b: sim.band_flux(d, var, b) for b in sim.bands}
        REF[R] = {}
        for b1, b2 in PAIRS:
            if b1 in sim.bands:
                arr = np.array([[tab[(ip, iz)][b1]/tab[(ip, iz)][b2] for iz in range(len(MK_ZDS))] for ip in range(len(MK_PWVS))])
                REF[R][(b1, b2)] = (lambda arr: lambda p, iz: np.interp(np.clip(p, 0, MK_PWVS[-1]), MK_PWVS, arr[:, iz]))(arr)

def run_mocks(lines, nmock, ncpu):
    from multiprocessing import Pool
    rng = np.random.default_rng(20261116)
    conds = [draw(rng) for _ in range(nmock)]
    for R in RS:
        _SIM[R] = Sim(lines, R)
    build_refs(lines)
    jobs = [(R, k, c) for R in RS for k, c in enumerate(conds)]
    t0 = time.time()
    with Pool(ncpu) as pool: # forked workers inherit _SIM and REF
        res = pool.map(one_mock, jobs, chunksize = 4)
    D = pd.DataFrame([r for rr in res for r in rr])
    D.to_csv(osp.join(OUT, 'mocks.csv'), index = False)
    print(f'{len(D)} mock fits in {time.time() - t0:.0f} s')
    cols = [c for c in D.columns if c.startswith(('A_', 'Bk_', 'Bf_', 'Bh_')) and not c.endswith(('_Tr', '_chi2', 'pwv_wst', '_rh', '_Tv'))]
    S = D.groupby(['R', 'g'])[cols].agg(lambda x: f'{100*np.nanmedian(x):+.2f} ± {100*(np.nanpercentile(x, 84) - np.nanpercentile(x, 16))/2:.2f}')
    pd.set_option('display.width', 400); pd.set_option('display.max_columns', 80)
    print(S.T.to_string())


###############################################
### O2 B band and 7-2: steps o2, transfig, o2mocks ###
###############################################

O2B = (679.0, 697.0) # the part of the spectrum the O2 B band touches at R ~ 100
SHIFT_PM = [-5, -2, -1, 1, 2, 5] # OH line-position errors to test [pm]

def band_T(lines, T, band):
    I = lines.intensities()
    m = lines.band == band
    return float(np.sum(I[m]*T[m])/np.sum(I[m]))

def o2_study(lines):
    ''' How deep the B band cuts the 7-2 band, how that scales with airmass, and how sensitive it is to the OH line positions '''
    c = np.load(CACHE)
    T = c['T_line']
    out = {'band_T': {}, 'zd_wst': [wst_zd(z) for z in MK_ZDS]}
    for b in ['6-1', '7-2', '8-3', '9-4', '6-2', '7-3', '8-4']:
        out['band_T'][b] = {str(round(P_RATIO*p, 2)): [round(band_T(lines, T[ip, iz], b), 4) for iz in range(len(MK_ZDS))] for ip, p in enumerate(MK_PWVS) if p in (0, 6, 20)}
    # the R = 100 7-2 feature: line-weighted transmission against what the continuum sees there
    sim = Sim.__new__(Sim); sim.R = 100; sim.edges, sim.lam = pixels(100)
    I = lines.intensities()
    w0, s = 686.0, 686.0/(100*FWHM2SIG)
    k = np.exp(-0.5*((lines.pos - w0)/s)**2)*I
    kp = np.argmin(np.abs(sim.lam - w0))
    out['feature_7-2'] = [dict(zd_wst = round(wst_zd(z), 1), t_lines = round(float(np.sum(k*T[6, iz])/np.sum(k)), 4), t_cont = round(float(c['T_cont_100'][6, iz, kp]), 4)) for iz, z in enumerate(MK_ZDS)]
    # airmass slope of the 7-2 band transmission (the O2 column scales with surface pressure x airmass)
    X = 1/np.cos(np.radians(MK_ZDS)) # proportional to the slant O2 column
    tb = np.array([band_T(lines, T[6, iz], '7-2') for iz in range(len(MK_ZDS))])
    out['dlnT_dlnX_7-2'] = round(float(np.polyfit(np.log(X), np.log(tb), 1)[0]), 4)
    # OH line-position errors: transmission of every line in 2 bands recomputed with shifted positions
    trans = load_transmission('background', 6.0, 50, basedir = AS.BASE, min_wav = 600, max_wav = 1010)
    from dev.make_spectra import get_transmission_at_lines
    out['shift'] = {}
    for b in ['7-2', '7-3', '8-4']:
        m = (lines.band == b) & (lines.pos > 610) & (lines.pos < 1005)
        L = lines.L[m][['Position', 'Species']].copy()
        T0 = get_transmission_at_lines(L, trans['nm'].to_numpy(), trans['combin'].to_numpy(), Trot = 195)
        Ib = I[m]
        res = {}
        for dpm in SHIFT_PM:
            Ls = L.copy(); Ls['Position'] = Ls.Position + dpm*1e-3
            Ts = get_transmission_at_lines(Ls, trans['nm'].to_numpy(), trans['combin'].to_numpy(), Trot = 195)
            res[str(dpm)] = dict(band = round(float(np.sum(Ib*Ts)/np.sum(Ib*T0) - 1), 5),
                                 lines_90 = round(float(np.percentile(np.abs(Ts - T0)[Ib > 0.01*Ib.max()], 90)), 4))
        out['shift'][b] = res
    json.dump(out, open(osp.join(OUT, 'o2_study.json'), 'w'), ensure_ascii = False)
    print(json.dumps(out, ensure_ascii = False, indent = 1))

def build_shift_cache(lines):
    ''' Line transmissions with the OH lines in the B-band region shifted by +-2 pm, for every grid atmosphere '''
    from dev.make_spectra import get_transmission_at_lines
    m = (lines.pos > O2B[0] - 5) & (lines.pos < O2B[1] + 5) & lines.oh
    L = lines.L[m][['Position', 'Species']].copy()
    out = {f's{d}': np.load(CACHE)['T_line'].copy() for d in (-2, 2)}
    for (i, p), (j, z) in itertools.product(enumerate(MK_PWVS), enumerate(MK_ZDS)):
        trans = load_transmission('background', float(p), z, basedir = AS.BASE, min_wav = O2B[0] - 10, max_wav = O2B[1] + 10)
        for d in (-2, 2):
            Ls = L.copy(); Ls['Position'] = Ls.Position + d*1e-3
            out[f's{d}'][i, j, m] = get_transmission_at_lines(Ls, trans['nm'].to_numpy(), trans['combin'].to_numpy(), Trot = 195)
        print('shift cache', p, z, flush = True)
    np.savez(osp.join(OUT, 'o2_shift_cache.npz'), **out)

def trans_figure(lines):
    ''' Per R, on the figure grid: the transmission a continuum source sees and the one the sky lines see, for 3 atmospheres '''
    c = np.load(CACHE)
    grid = np.arange(340, 1000.01, 0.5)
    atm = {'fid': (6, 1), 'best': (0, 0), 'worst': (20, 2)} # MK (PWV index, ZD index): Wendelstein 8 mm ZD 31, 0 mm zenith, 27 mm ZD 48
    curves = {}
    for k, (ip, iz) in atm.items():
        tr = load_transmission('background', float(MK_PWVS[ip]), MK_ZDS[iz], basedir = AS.BASE, min_wav = 330, max_wav = 1010)
        nm, tc = tr['nm'].to_numpy()[::-1], tr['combin'].to_numpy()[::-1]
        edges = np.arange(330, 1010.001, 0.05)
        num, _ = np.histogram(nm, edges, weights = tc); den, _ = np.histogram(nm, edges)
        curves[k] = (0.5*(edges[1:] + edges[:-1]), num/np.maximum(den, 1))
    I = lines.intensities()
    out = dict(wav = grid.round(1).tolist(), atm = {'fid': 'PWV 8 mm, ZD 31°', 'best': 'PWV 0, zenith', 'worst': 'PWV 27 mm, ZD 48°'}, R = {})
    for R in RS:
        s = grid/(R*FWHM2SIG)
        r = {}
        for k, (ip, iz) in atm.items():
            bc, bt = curves[k]
            W = np.exp(-0.5*((grid[:, None] - bc[None, :])/s[:, None])**2)
            r[f'cont_{k}'] = (W @ bt/W.sum(1)).round(4).tolist()
            sl = lines.pos/(R*FWHM2SIG)
            K = np.exp(-0.5*((grid[:, None] - lines.pos[None, :])/sl[None, :])**2)/sl[None, :]
            sky = K @ I/np.sqrt(2*np.pi)
            tl = (K @ (I*c['T_line'][ip, iz]))/np.maximum(K @ I, 1e-30)
            r[f'lines_{k}'] = [None if q < 0.5 else round(float(v), 4) for q, v in zip(sky, tl)]
        out['R'][str(R)] = r
    json.dump(out, open(osp.join(OUT, 'figure_trans.json'), 'w'), ensure_ascii = False, separators = (',', ':'))

O2CASES = [('Bh', {}), ('mask', dict(mask = O2B)), ('shift+2', dict(tset = 's2')), ('shift-2', dict(tset = 's-2'))]

def one_o2_mock(args):
    R, k, cond = args
    sim = _SIM[R]
    np.random.seed(cond['seed']); chem = make_intensity_ratios(cond['alpha'])
    rng = np.random.default_rng(cond['seed'])
    lam = np.array(LAM_EVAL)
    Strue = np.exp(cond['s1']*(lam - 700)/300 + cond['s2']*((lam - 700)/300)**2)
    rows = []
    for g in NOISE:
        d, var, I = sim.observe(cond['Tr'], cond['Tv'], cond['rh'], chem, cond['lsf'], cond['ip'], cond['iz'], cond['C0'], cond['beta'], cond['s1'], cond['s2'], g, rng)
        row = dict(R = R, mock = k, g = g, zd_wst = wst_zd(MK_ZDS[cond['iz']]), pwv_wst = P_RATIO*MK_PWVS[cond['ip']])
        for case, kw in O2CASES:
            th, _, chi2 = sim.forward_fit(d, var, cond['iz'], None, True, **kw)
            Srec = np.exp(th[1]*(lam - 700)/300 + th[2]*((lam - 700)/300)**2)
            for l, e in zip(LAM_EVAL, Srec/Strue - 1):
                row[f'{case}_S{l}'] = e
            row[f'{case}_7-2/7-3'] = (Srec[1]/Srec[5])/(Strue[1]/Strue[5]) - 1
            row[f'{case}_chi2'] = chi2
        rows.append(row)
    return rows

def run_o2_mocks(lines, nmock, ncpu):
    from multiprocessing import Pool
    rng = np.random.default_rng(20261116) # the same nights as run_mocks
    conds = [draw(rng) for _ in range(nmock)]
    sim = Sim(lines, 100)
    sc = np.load(osp.join(OUT, 'o2_shift_cache.npz'))
    for d in (-2, 2):
        sim.tsets[f's{d}'] = sim.build_templates(sc[f's{d}'])
    _SIM[100] = sim
    with Pool(ncpu) as pool:
        res = pool.map(one_o2_mock, [(100, k, c) for k, c in enumerate(conds)], chunksize = 4)
    D = pd.DataFrame([r for rr in res for r in rr])
    D.to_csv(osp.join(OUT, 'o2_mocks.csv'), index = False)
    h = lambda x: (np.nanpercentile(x, 84) - np.nanpercentile(x, 16))/2
    for case, _ in O2CASES:
        print(case)
        print(D.groupby('g')[[f'{case}_S{l}' for l in LAM_EVAL] + [f'{case}_7-2/7-3']].agg(lambda x: f'{100*np.median(x):+.2f}±{100*h(x):.2f}').to_string())

#########################################################
### Step: fmocks. Band-flux ratios only, no sky model fit ###
#########################################################

KNOW = {'full': 'PWV and ZD known', 'airmass': 'ZD known, PWV assumed 8 mm', 'none': 'atmosphere assumed (8 mm, ZD 31)'}
IP_FID, IZ_FID = 6, 1
CURVES = {'2': [('8-3', '8-4'), ('7-2', '7-3')], 'all': PAIRS}
FREF = {}

def build_feature_refs():
    ''' Band fluxes of fiducial airglow (unit response, noiseless) through every grid atmosphere, for both background shapes '''
    for R in RS:
        sim = _SIM[R]
        FREF[R] = {}
        for (ip, iz) in itertools.product(range(len(MK_PWVS)), range(len(MK_ZDS))):
            d, var, _ = sim.observe(195, 1e4, 1.0, None, 1.0, ip, iz, 10.0, 0.0, 0.0, 0.0, np.inf, None)
            for bg in ['lin', 'tel']:
                tc = None if bg == 'lin' else sim.T_cont[ip, iz]
                FREF[R][(bg, ip, iz)] = {b: sim.band_flux(d, var, b, tc = tc) for b in sim.bands}

def one_feature_mock(args):
    R, k, cond = args
    sim = _SIM[R]
    np.random.seed(cond['seed']); chem = make_intensity_ratios(cond['alpha'])
    rng = np.random.default_rng(cond['seed'])
    Sf = lambda l: np.exp(cond['s1']*(np.asarray(l) - 700)/300 + cond['s2']*((np.asarray(l) - 700)/300)**2)
    xf = lambda l: (np.asarray(l) - 700)/300
    ip, iz = cond['ip'], cond['iz']
    assumed = {'full': (ip, iz), 'airmass': (IP_FID, iz), 'none': (IP_FID, IZ_FID)}
    rows = []
    for g in NOISE:
        d, var, I = sim.observe(cond['Tr'], cond['Tv'], cond['rh'], chem, cond['lsf'], ip, iz, cond['C0'], cond['beta'], cond['s1'], cond['s2'], g, rng)
        row = dict(R = R, mock = k, g = g, **{c: cond[c] for c in ['Tr', 'Tv', 'rh', 'lsf', 'C0', 'beta']}, pwv_wst = P_RATIO*MK_PWVS[ip], zd_wst = wst_zd(MK_ZDS[iz]))
        for bg, (kn, (ipa, iza)) in itertools.product(['lin', 'tel'], assumed.items()):
            tc = None if bg == 'lin' else sim.T_cont[ipa, iza]
            F = {b: sim.band_flux(d, var, b, tc = tc) for b in sim.bands}
            ref = FREF[R][(bg, ipa, iza)]
            lnrho = {}
            for b1, b2 in PAIRS:
                if b1 in sim.bands:
                    l1, l2 = sim.band_lam(b1), sim.band_lam(b2)
                    rho = (F[b1]/F[b2])/(ref[b1]/ref[b2])
                    row[f'{bg}_{kn}_{b1}/{b2}'] = rho/(Sf(l1)/Sf(l2)) - 1
                    if rho > 0:
                        lnrho[(b1, b2)] = (np.log(rho), l1, l2)
            # the response exp(s1 x + s2 x^2) from the pair ratios alone (each ratio gives ln S(l1) - ln S(l2))
            for cname, cp in CURVES.items():
                use = [lnrho[p] for p in cp if p in lnrho]
                if len(use) < 2:
                    continue
                A = np.array([[xf(l1) - xf(l2), xf(l1)**2 - xf(l2)**2] for _, l1, l2 in use]); y = np.array([u[0] for u in use])
                s = np.linalg.lstsq(A, y, rcond = None)[0]
                for l in LAM_EVAL:
                    row[f'{bg}_{kn}_c{cname}_S{l}'] = np.exp(s[0]*xf(l) + s[1]*xf(l)**2)/Sf(l) - 1
        rows.append(row)
    return rows

def run_feature_mocks(lines, nmock, ncpu):
    from multiprocessing import Pool
    rng = np.random.default_rng(20261116) # the same nights as run_mocks
    conds = [draw(rng) for _ in range(nmock)]
    for R in RS:
        _SIM[R] = Sim(lines, R, templates = False)
    build_feature_refs()
    with Pool(ncpu) as pool:
        res = pool.map(one_feature_mock, [(R, k, c) for R in RS for k, c in enumerate(conds)], chunksize = 4)
    D = pd.DataFrame([r for rr in res for r in rr])
    D.to_csv(osp.join(OUT, 'feature_mocks.csv'), index = False)
    h = lambda x: (np.nanpercentile(x, 84) - np.nanpercentile(x, 16))/2
    pd.set_option('display.width', 400); pd.set_option('display.max_columns', 80); pd.set_option('display.max_rows', 400)
    cols = [c for c in D.columns if c.startswith(('lin_', 'tel_'))]
    S = D.groupby(['R', 'g'])[cols].agg(lambda x: f'{100*np.nanmedian(x):+.1f}±{100*h(x):.1f}')
    print(S.T.to_string())

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('steps', nargs = '+')
    ap.add_argument('--nmock', type = int, default = 300)
    ap.add_argument('--ncpu', type = int, default = 1)
    a = ap.parse_args()
    lines = Lines()
    if 'check' in a.steps: # the vectorised OH intensities against make_OH_intensities
        for c in [(180, 8000, 0.75), (210, 12000, 1.33)]:
            ref = AS.line_table(*c).Intensity.to_numpy()[lines.oh]*1e3
            new = lines.intensities(*c)[lines.oh]
            print('max |rel diff| OH intensities', c, np.max(np.abs(new/ref - 1)))
    if 'cache' in a.steps:
        build_cache(lines)
    if 'figure' in a.steps:
        figure_data(lines)
    if 'pwv' in a.steps:
        pwv_study(lines)
    if 'mocks' in a.steps:
        run_mocks(lines, a.nmock, a.ncpu)
    if 'o2' in a.steps:
        o2_study(lines)
    if 'shiftcache' in a.steps:
        build_shift_cache(lines)
    if 'transfig' in a.steps:
        trans_figure(lines)
    if 'fmocks' in a.steps:
        run_feature_mocks(lines, a.nmock, a.ncpu)
    if 'o2mocks' in a.steps:
        run_o2_mocks(lines, a.nmock, a.ncpu)
