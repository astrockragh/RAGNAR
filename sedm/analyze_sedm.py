'''
Analysis of the SEDM RAGNAR runs (run_sedm.py), 2026-11-16.

a) Wavelength calibration: which R = 100 peaks are robust, how much each moves with the emission, atmosphere and LSF parameters, and with a
   mis-specified resolution (centroider runs, cent = 1, and cent = 0 for comparison).
b) Relative flux calibration, without the centroider (cent = 0, raw peak heights):
   1. purity: the fraction of a peak's line-centre intensity that comes from a single upper state (v', J', F; the e/f Lambda-doublet components
      share an upper level and are counted together). Two peaks that are both >95-99% from the same upper state have a ratio fixed by the
      Einstein A's, whatever the atmospheric conditions. Scanned against R to see where such peaks start to exist.
   2. the purity of the R = 100 peaks themselves, by upper state and by upper vibrational level.
   3. how much the ratio of each pair of R = 100 peak heights changes over the simulated grid (empirical, so model dependent).
   4. the transmission seen by the narrow sky lines in each peak against the transmission a continuum source sees at R = 100.

Run from RAGNAR_new/: python sedm/analyze_sedm.py [--date 20261116]
'''
import re, sys, glob, argparse, itertools
import numpy as np
import pandas as pd
import os.path as osp

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(HERE))
from dev.species_intensities import get_pgopher_linelist, make_OH_intensities, get_O2_at_T, get_atomic_lines
from dev.make_spectra import load_transmission, get_transmission_at_lines, get_peaks

BASE = '/tigress/cj1223/'
SPEC_DIR = '/tigress/cj1223/spectra/'
OUT = osp.join(HERE, 'out')
WMIN, WMAX = 334, 1015 # as in run_sedm.py
BAND = (340, 1000) # the SEDM range
FWHM2SIG = 2*np.sqrt(2*np.log(2))
C_DARK = 10.0 # [R/nm] reference dark-sky continuum (V ~ 21.6 mag/arcsec^2 is ~12 R/nm at 550 nm), for the line-to-continuum contrast only

def sig_of(w, R):
    ''' Gaussian sigma [nm] at wavelength w for resolving power R = lambda/FWHM '''
    return np.asarray(w)/(R*FWHM2SIG)

#####################################
### Line table with upper states ###
#####################################

_A = None
def _pgopher():
    global _A
    if _A is None:
        _A = get_pgopher_linelist()
    return _A.copy()

ATOMIC_UPPER = {'OI(1S_0-1D_2)': 'O(1S)', 'OI(1D_2-3P_2)': 'O(1D)', 'OI(1D_2-3P_1)': 'O(1D)', 'OI(1D_2-3P_0)': 'O(1D)',
                'NaI(D2)': 'Na(3p 2P3/2)', 'NaI(D1)': 'Na(3p 2P1/2)', 'H_alpha': 'H(n=3)', 'H_beta': 'H(n=4)'}

def line_table(Tr = 195, Tv = 1e4, rh = 1.0):
    '''
    All lines in WMIN-WMAX with their intrinsic intensity [kR] at (Trot, Tvib, r_hot_scale), fiducial chemistry (all species scales 1),
    and their upper state (up), upper vibrational level or species (vgroup) and transition label (trans, e/f merged).
    '''
    A = _pgopher()
    A['Intensity'] = make_OH_intensities(A, Tr, Tv, r_hot_scale = rh) # normalized over the full list, so cut afterwards
    L = pd.concat([A, get_O2_at_T(Tr), get_atomic_lines()])
    L = L[(L.Position > WMIN) & (L.Position < WMAX)].copy()
    oh = (L.Species == 'OH').to_numpy()
    vu, vl = L["v'"].fillna(-1).astype(int).to_numpy(), L['v"'].fillna(-1).astype(int).to_numpy()
    Jp, Fp = L["J'"].to_numpy(), L["N'"].fillna(-1).astype(int).to_numpy()
    br = L['Branch'].fillna('').astype(str).to_numpy()
    lab = L['Label'].astype(str).to_numpy()
    up, vg, tr = [], [], []
    for i in range(len(L)):
        if oh[i]:
            up.append(f"OH v'={vu[i]} J'={Jp[i]} F{Fp[i]}")
            vg.append(f"OH v'={vu[i]}")
            tr.append(f"({vu[i]}-{vl[i]}){re.sub(r'[ef](?=\()', '', br[i])}") # e.g. (8-3)pP1(3.5)
        elif L.Species.iloc[i] == 'O2':
            up.append(f'O2 line {L.index[i]}') # HITRAN carries no upper-state quanta here, so every O2 line is its own upper state (conservative)
            vg.append('O2 b(v=0)')
            tr.append(f'O2 b(0-1) {L.Position.iloc[i]:.2f}')
        else:
            up.append(ATOMIC_UPPER[lab[i]])
            vg.append(L.Species.iloc[i])
            tr.append(lab[i])
    L['up'], L['vgroup'], L['trans'] = up, vg, tr
    return L.sort_values('Position', kind = 'mergesort').reset_index().rename(columns = {'index': 'orig_id'})

_T = {}
def line_transmission(L, pwv, zd, aerosol = 'background'):
    ''' Transmission at every line of L (Doppler-convolved, as RAGNAR does), cached per atmosphere '''
    key = (aerosol, pwv, zd, len(L))
    if key not in _T:
        trans = load_transmission(aerosol, pwv, zd, basedir = BASE, min_wav = WMIN - 2, max_wav = WMAX + 2)
        _T[key] = (get_transmission_at_lines(L, trans['nm'].to_numpy(), trans['combin'].to_numpy(), Trot = 195), trans)
    return _T[key]

##############
### Purity ###
##############

def decompose(w, sig, pos, I, codes, ncodes, nsig = 8):
    '''
    Line-centre decomposition (no centroider): contribution of each code (upper state or vgroup) to the observed intensity at wavelength(s) w,
    for a Gaussian LSF of width sig. pos must be ascending. Returns (M, ncodes) contributions and the (M,) totals.
    '''
    w, sig = np.atleast_1d(w), np.broadcast_to(np.atleast_1d(sig), np.shape(np.atleast_1d(w)))
    lo, hi = np.searchsorted(pos, w - nsig*sig), np.searchsorted(pos, w + nsig*sig)
    out = np.zeros((len(w), ncodes))
    for m in range(len(w)):
        sl = slice(lo[m], hi[m])
        c = I[sl]*np.exp(-0.5*((w[m] - pos[sl])/sig[m])**2) # the common 1/(sqrt(2 pi) sig) cancels in fractions
        out[m] = np.bincount(codes[sl], weights = c, minlength = ncodes)
    return out, out.sum(axis = 1)

def own_purity(w, sig, own_code, pos, I, codes, nsig = 8):
    ''' Fraction of the line-centre intensity at w that comes from the upper state own_code (one per w) '''
    lo, hi = np.searchsorted(pos, w - nsig*sig), np.searchsorted(pos, w + nsig*sig)
    p = np.empty(len(w))
    for m in range(len(w)):
        sl = slice(lo[m], hi[m])
        c = I[sl]*np.exp(-0.5*((w[m] - pos[sl])/sig[m])**2)
        p[m] = c[codes[sl] == own_code[m]].sum()/c.sum()
    return p

def transition_groups(L, I):
    ''' Collapse e/f components into transitions: intensity-weighted position, summed intensity, upper state '''
    g = pd.DataFrame({'trans': L.trans, 'up': L.up, 'vgroup': L.vgroup, 'pos': L.Position, 'I': I, 'wI': L.Position*I})
    g = g.groupby(['trans', 'up', 'vgroup'], as_index = False).agg(I = ('I', 'sum'), wI = ('wI', 'sum'))
    g['pos'] = g.wI/g.I
    return g.drop(columns = 'wI').sort_values('pos').reset_index(drop = True)

EMISSION_CORNERS = list(itertools.product([180, 210], [8000, 12000], [0.75, 1.33]))
ATMOS = [(1.0, 0), (10.0, 50)] # (PWV [mm], ZD [deg]) extremes of the run grid
FID_ATM = (5.0, 0) # fiducial atmosphere
ATM_GRID = ([1.0, 5.0, 10.0], [0, 50]) # the run's PWV and ZD values
def use_wendelstein():
    ''' Mauna Kea curves standing in for Wendelstein (see run_sedm.py): MK PWV 2-15 mm and ZD 40-60 '''
    global ATMOS, FID_ATM, ATM_GRID
    ATMOS, FID_ATM, ATM_GRID = [(2.0, 40), (15.0, 60)], (8.0, 50), ([2.0, 8.0, 15.0], [40, 60])
R_SCAN = [100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000, 20000]

def purity_scan(Imins = (5.0, 20.0), thrs = (0.95, 0.99), lever_min = 100.0):
    '''
    For every transition brighter than Imin [R] (fiducial, after transmission) in the SEDM range, the purity at its own position, minimised over
    the emission corners and atmosphere extremes. Then count pure transitions, and upper states with >= 2 pure transitions (calibration pairs).
    '''
    Lf = line_table()
    codes, uniq = pd.factorize(Lf.up)
    pos = Lf.Position.to_numpy()
    Ts = {a: line_transmission(Lf, *a)[0] for a in ATMOS}
    T_fid = line_transmission(Lf, *FID_ATM)[0]
    G = transition_groups(Lf, Lf.Intensity.to_numpy()*T_fid*1e3) # [R]
    G = G[(G.pos > BAND[0]) & (G.pos < BAND[1]) & (G.I >= min(Imins))].reset_index(drop = True)
    gcode = pd.Index(uniq).get_indexer(G.up)
    rows = []
    Lc = {c: line_table(*c) for c in EMISSION_CORNERS}
    for R in R_SCAN:
        sig = sig_of(G.pos.to_numpy(), R)
        pmin = np.ones(len(G))
        for c in EMISSION_CORNERS:
            for a in ATMOS:
                I = Lc[c].Intensity.to_numpy()*Ts[a]
                pmin = np.minimum(pmin, own_purity(G.pos.to_numpy(), sig, gcode, pos, I, codes))
        rows.append(pd.DataFrame({'R': R, 'trans': G.trans, 'up': G.up, 'pos': G.pos, 'I_R': G.I, 'purity_min': pmin}))
        print(f'purity scan R = {R}: done')
    P = pd.concat(rows, ignore_index = True)

    summ = []
    for R, d in P.groupby('R'):
        for Imin, thr in itertools.product(Imins, thrs):
            good = d[(d.I_R >= Imin) & (d.purity_min >= thr)]
            lev = good.groupby('up').pos.agg(lambda x: x.max() - x.min())
            n2 = good.groupby('up').size()
            summ.append(dict(R = R, Imin_R = Imin, purity = thr, n_pure_transitions = len(good), n_upper_states_with_pair = int((n2 >= 2).sum()),
                             n_pairs_lever_ge_100nm = int((lev >= lever_min).sum()),
                             pure_wav_min = good.pos.min() if len(good) else np.nan, pure_wav_max = good.pos.max() if len(good) else np.nan))
    return P, pd.DataFrame(summ)

def example_lines_vs_R(names = ('(8-3)pP1(3.5)', '(8-4)pP1(3.5)', '(8-3)pP2(3.5)', '(8-4)pP2(3.5)')):
    ''' Purity of the lines in the team's example against R, at fiducial conditions and minimised over the corners '''
    Lf = line_table()
    codes, uniq = pd.factorize(Lf.up)
    pos = Lf.Position.to_numpy()
    Ts = {a: line_transmission(Lf, *a)[0] for a in ATMOS + [FID_ATM]}
    G = transition_groups(Lf, Lf.Intensity.to_numpy()*Ts[FID_ATM]*1e3)
    G = G[G.trans.isin(names)].reset_index(drop = True)
    gcode = pd.Index(uniq).get_indexer(G.up)
    Rs = np.unique(np.round(np.logspace(2, np.log10(30000), 60)))
    Lc = {c: line_table(*c) for c in EMISSION_CORNERS}
    rows = []
    for R in Rs:
        sig = sig_of(G.pos.to_numpy(), R)
        pfid = own_purity(G.pos.to_numpy(), sig, gcode, pos, Lf.Intensity.to_numpy()*Ts[FID_ATM], codes)
        pmin = np.ones(len(G))
        for c in EMISSION_CORNERS:
            for a in ATMOS:
                pmin = np.minimum(pmin, own_purity(G.pos.to_numpy(), sig, gcode, pos, Lc[c].Intensity.to_numpy()*Ts[a], codes))
        for k in range(len(G)):
            rows.append(dict(R = R, trans = G.trans[k], pos = G.pos[k], I_R = G.I[k], purity_fid = pfid[k], purity_min = pmin[k]))
    return pd.DataFrame(rows)

###########################
### The R = 100 RAGNAR runs ###
###########################

def load_run(R, cent, date, tag = ''):
    lst = pd.read_pickle(osp.join(osp.dirname(HERE), 'lineLists', f'sedm_R{R:g}_{cent}{tag}_{date}.pkl'))
    f = glob.glob(osp.join(SPEC_DIR, f'sedm_R{R:g}_{cent}{tag}_{date}*.npz'))
    assert len(f) == 1, f
    z = np.load(f[0])
    P = pd.DataFrame(z['params'], columns = z['param_names'])
    return lst, z['wav_range'], z['specs'], P, float(z['res_in'])

def peak_purity_table(lst, R):
    ''' Upper-state and v'-level decomposition at each peak of a RAGNAR list, fiducial and minimised over the corners/atmospheres '''
    Lf = line_table()
    pos = Lf.Position.to_numpy()
    cu, uu = pd.factorize(Lf.up)
    cv, uv = pd.factorize(Lf.vgroup)
    w = lst.wav.to_numpy(); sig = sig_of(w, R)
    T5 = line_transmission(Lf, *FID_ATM)[0]
    du, tot = decompose(w, sig, pos, Lf.Intensity.to_numpy()*T5, cu, len(uu))
    dv, _ = decompose(w, sig, pos, Lf.Intensity.to_numpy()*T5, cv, len(uv))
    fu, fv = du/tot[:, None], dv/tot[:, None]
    # the dominant upper state / vgroup at fiducial, and its smallest fraction over the corners and atmospheres
    ku, kv = fu.argmax(1), fv.argmax(1)
    umin, vmin = fu.max(1), fv.max(1)
    for c in EMISSION_CORNERS:
        Lc = line_table(*c)
        for a in ATMOS:
            I = Lc.Intensity.to_numpy()*line_transmission(Lf, *a)[0]
            d1, t1 = decompose(w, sig, pos, I, cu, len(uu)); d2, _ = decompose(w, sig, pos, I, cv, len(uv))
            umin = np.minimum(umin, d1[np.arange(len(w)), ku]/t1); vmin = np.minimum(vmin, d2[np.arange(len(w)), kv]/t1)
    top3 = [' + '.join(f'{uu[j]} {fu[m, j]:.0%}' for j in np.argsort(fu[m])[::-1][:3]) for m in range(len(w))]
    n_up90 = [int(np.searchsorted(np.cumsum(np.sort(fu[m])[::-1]), 0.9) + 1) for m in range(len(w))]
    return pd.DataFrame({'wav': w, 'dominant_upper_state': uu[ku], 'f_upper_fid': fu.max(1), 'f_upper_min': umin, 'n_upper_states_for_90pct': n_up90,
                         'dominant_vgroup': uv[kv], 'f_vgroup_fid': fv.max(1), 'f_vgroup_min': vmin, 'top3_upper_states': top3})

def peak_heights(lst, wav_range, specs, R, half = 0.3):
    ''' Raw peak height (no centroider) of each listed peak in every spectrum: the maximum within +-half*sigma of the mean peak wavelength [kR/nm] '''
    H = np.empty((len(specs), len(lst)))
    for j, w in enumerate(lst.wav):
        s = sig_of(w, R)
        lo, hi = np.searchsorted(wav_range, [w - half*s, w + half*s])
        H[:, j] = specs[:, lo:hi].max(axis = 1)
    return H

def ratio_stability(lst, H, P):
    '''
    Half the full range of each pair's peak-height ratio, (max/min - 1)/2, over
      emission: Trot, Tvib, r_hot_scale and chemistry, at fixed atmosphere and LSF (worst of the 6 atmospheres)
      lsf:      the three LSF scatters, at fixed everything else (worst case)
      atmos:    PWV and ZD, at fixed emission and LSF (worst case)
      all:      the full grid
    '''
    n = len(lst)
    logH = np.log(H)
    def half_range(groups):
        worst = np.zeros((n, n))
        for _, idx in groups:
            lr = logH[idx.to_numpy()]
            d = lr[:, :, None] - lr[:, None, :] # log ratio i/j
            worst = np.maximum(worst, (np.exp(d.max(0) - d.min(0)) - 1)/2)
        return worst
    out = {}
    out['emission'] = half_range(P[P.lsf_k == 1].groupby(['pwv', 'zd']).groups.items())
    out['lsf'] = half_range(P.groupby(['pwv', 'zd', 'Trot', 'Tvib', 'r_hot_scale', 'chem_k']).groups.items())
    out['atmos'] = half_range(P[P.lsf_k == 1].groupby(['Trot', 'Tvib', 'r_hot_scale', 'chem_k']).groups.items())
    out['all'] = half_range([(0, P.index)])
    rows = []
    for i, j in itertools.combinations(range(n), 2):
        rows.append(dict(wav_1 = lst.wav.iloc[i], wav_2 = lst.wav.iloc[j], lever_nm = lst.wav.iloc[j] - lst.wav.iloc[i],
                         **{f'half_range_{k}': v[i, j] for k, v in out.items()}))
    return pd.DataFrame(rows)

PARS = [('Trot', 180, 210), ('Tvib', 8000, 12000), ('r_hot_scale', 0.75, 1.33), ('chem_k', 0, 4), ('lsf_k', 0, 2), ('pwv', 1.0, 10.0), ('zd', 0, 50)]

def flux_features(lst, wav_range, specs, min_peak = 2.0, min_det = 0.75):
    '''
    The peaks usable for flux work (no centroider): every peak of the cent = 0 list, robust or not, that is found in >= min_det of the
    spectra and has a mean raw peak height >= min_peak R/nm. Robustness matters for identifying a peak, not for measuring a flux at a fixed
    wavelength, and the robust list misses the 6-2 and 8-4 blends, which split into two raw peaks in some spectra.
    '''
    mean = specs.mean(0)*1e3
    h = mean[np.searchsorted(wav_range, lst.wav.to_numpy())]
    keep = (h >= min_peak) & (lst.detection_prob.to_numpy() >= min_det) & (lst.wav > BAND[0]).to_numpy() & (lst.wav < BAND[1]).to_numpy()
    return lst[keep].reset_index(drop = True)

def ratio_breakdown(feat, H, P, pairs):
    ''' For the given (i, j) index pairs: change of the ratio i/j between the extremes of each parameter, averaged over all others [fraction] '''
    rows = []
    for i, j in pairs:
        r = H[:, i]/H[:, j]
        row = dict(wav_1 = feat.wav[i], wav_2 = feat.wav[j], mean_ratio = r.mean())
        for par, _, _ in PARS:
            lo, hi = P[par].min(), P[par].max()
            row[f'd_{par}'] = r[P[par] == hi].mean()/r[P[par] == lo].mean() - 1
        rows.append(row)
    return pd.DataFrame(rows)

def wavelength_sensitivity(lst, wav_range, specs, P, R, res_in, cent):
    '''
    Re-find the peaks in every spectrum as RAGNAR does, match them to the listed peaks, and split the scatter of each peak position into
    the mean shift between the extremes of each parameter
    '''
    line_s = (wav_range.min() + wav_range.max())/2/(res_in*2.82982)
    pw = get_peaks(list(specs.astype(float)), wav_range, line_spread = line_s, intensity_cut = 0, use_centroider = cent, verbose = 0)
    W = np.full((len(specs), len(lst)), np.nan)
    for k, p in enumerate(pw):
        for j, w in enumerate(lst.wav):
            i = np.argmin(np.abs(p - w))
            if abs(p[i] - w) < 0.5*sig_of(w, R):
                W[k, j] = p[i]
    rows = []
    for j, w in enumerate(lst.wav):
        x = W[:, j]
        r = dict(wav = w, found_frac = np.mean(np.isfinite(x)), std_nm = np.nanstd(x), full_range_nm = np.nanmax(x) - np.nanmin(x), fwhm_nm = w/R)
        for par in ['Trot', 'Tvib', 'r_hot_scale', 'pwv', 'zd', 'lsf_k']:
            lo, hi = P[par].min(), P[par].max()
            r[f'shift_{par}_nm'] = np.nanmean(x[P[par] == hi]) - np.nanmean(x[P[par] == lo])
        cm = [np.nanmean(x[P.chem_k == k]) for k in range(int(P.chem_k.max()) + 1)]
        r['shift_chem_nm'] = cm[-1] - cm[0]
        rows.append(r)
    return pd.DataFrame(rows)

def transmission_table(lst, R):
    '''
    At each peak: the transmission the sky lines in it see (intensity weighted over the lines at line centre), and the transmission a
    continuum source sees after R = 100 smoothing, for the PWV/ZD grid. Their ratio is what a sky-line response would get wrong for a star.
    '''
    Lf = line_table()
    pos = Lf.Position.to_numpy()
    rows = []
    for pwv, zd in itertools.product(*ATM_GRID):
        t_line, trans = line_transmission(Lf, pwv, zd)
        nm, tc = trans['nm'].to_numpy()[::-1], trans['combin'].to_numpy()[::-1]
        for w in lst.wav:
            s = sig_of(w, R)
            lo, hi = np.searchsorted(pos, [w - 8*s, w + 8*s])
            k = Lf.Intensity.to_numpy()[lo:hi]*np.exp(-0.5*((w - pos[lo:hi])/s)**2)
            tl = np.sum(k*t_line[lo:hi])/np.sum(k)
            a, b = np.searchsorted(nm, [w - 5*s, w + 5*s])
            g = np.exp(-0.5*((nm[a:b] - w)/s)**2)
            rows.append(dict(wav = w, pwv = pwv, zd = zd, t_lines = tl, t_continuum = np.sum(g*tc[a:b])/np.sum(g)))
    T = pd.DataFrame(rows)
    T['lines_over_continuum'] = T.t_lines/T.t_continuum
    return T

def contrast_table(lst, wav_range, specs, R):
    from scipy.signal import peak_prominences
    mean = specs.mean(0)*1e3 # [R/nm]
    idx = np.searchsorted(wav_range, lst.wav.to_numpy())
    prom = peak_prominences(mean, idx)[0]
    return pd.DataFrame({'wav': lst.wav, 'peak_R_per_nm': mean[idx], 'prominence_R_per_nm': prom, 'peak_over_dark_continuum': mean[idx]/C_DARK,
                         'fwhm_nm': lst.wav/R, 'line_flux_R_approx': mean[idx]*sig_of(lst.wav, R)*np.sqrt(2*np.pi)})

if __name__ == '__main__':
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default = '20261116')
    ap.add_argument('--skip_scan', action = 'store_true') # skip the purity-vs-R scan
    ap.add_argument('--cents', type = int, nargs = '+', default = [0, 1])
    ap.add_argument('--tag', default = '') # '_wst' for the Wendelstein-like runs
    a = ap.parse_args()
    if a.tag == '_wst':
        use_wendelstein()
    os.makedirs(OUT, exist_ok = True)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 80)

    # b1) purity against R
    if not a.skip_scan:
        P, S = purity_scan()
        P.to_csv(osp.join(OUT, 'purity_scan_transitions.csv'), index = False)
        S.to_csv(osp.join(OUT, 'purity_scan_summary.csv'), index = False)
        print(S.to_string())
        E = example_lines_vs_R()
        E.to_csv(osp.join(OUT, 'example_lines_purity_vs_R.csv'), index = False)

    # R = 100 and the brackets
    for R in [100, 75, 150]:
        for cent in a.cents:
            try:
                lst, wav_range, specs, Pp, res_in = load_run(R, cent, a.date, a.tag)
            except (AssertionError, FileNotFoundError) as e:
                print(f'R = {R}, cent = {cent} not found ({e})'); continue
            tag = f'R{R}_cent{cent}{a.tag}'
            rob = lst[lst.robust & (lst.wav > BAND[0]) & (lst.wav < BAND[1])].reset_index(drop = True)
            ct = contrast_table(rob, wav_range, specs, R)
            pt = peak_purity_table(rob, R)
            ws = wavelength_sensitivity(rob, wav_range, specs, Pp, R, res_in, cent)
            base = rob[['wav', 'sigma_wav', 'intensity', 'sigma_intensity', 'detection_prob', 'transmission', 'transmission_origin', 'label']]
            tab = base.merge(ct, on = 'wav').merge(pt, on = 'wav').merge(ws, on = 'wav', suffixes = ('', '_ws'))
            tab.to_csv(osp.join(OUT, f'peaks_{tag}.csv'), index = False)
            print(f'\n=== {tag}: {len(lst)} peaks in the list, {len(rob)} robust in {BAND} ===')
            print(tab[['wav', 'sigma_wav', 'peak_R_per_nm', 'peak_over_dark_continuum', 'dominant_vgroup', 'f_vgroup_min', 'f_upper_min',
                       'n_upper_states_for_90pct', 'std_nm', 'full_range_nm', 'shift_Trot_nm', 'shift_pwv_nm', 'shift_lsf_k_nm']].round(3).to_string())
            if cent == 0:
                # flux calibration: raw peak heights of the usable features, not just the robust peaks
                feat = flux_features(lst, wav_range, specs)
                H = peak_heights(feat, wav_range, specs, R)
                ft = feat[['wav', 'robust', 'detection_prob', 'label']].merge(contrast_table(feat, wav_range, specs, R), on = 'wav').merge(peak_purity_table(feat, R), on = 'wav')
                ft.to_csv(osp.join(OUT, f'flux_features_{tag}.csv'), index = False)
                print(ft[['wav', 'robust', 'peak_R_per_nm', 'dominant_vgroup', 'f_vgroup_min', 'dominant_upper_state', 'f_upper_min', 'n_upper_states_for_90pct']].round(3).to_string())
                rs = ratio_stability(feat, H, Pp)
                rs.to_csv(osp.join(OUT, f'ratio_stability_{tag}.csv'), index = False)
                rb = ratio_breakdown(feat, H, Pp, list(itertools.combinations(range(len(feat)), 2)))
                rb.to_csv(osp.join(OUT, f'ratio_breakdown_{tag}.csv'), index = False)
                tt = transmission_table(feat, R)
                tt.to_csv(osp.join(OUT, f'transmission_{tag}.csv'), index = False)
