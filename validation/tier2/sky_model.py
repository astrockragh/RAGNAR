'''
Forward model of the PFS sky built from the RAGNAR line lists, fitted to the stacked PFS sky spectra (stack_sky.py).

Per arm, the model is
    S(lambda) = sum_bands A_band * OH_band(lambda) + sum_species A_s * line_spectrum_s(lambda) + P(lambda)
where
    OH_band:  the RAGNAR OH lines of one (v', v'') band at rotational temperature Trot (incl. the hot population), times the
              MODTRAN transmission at each line, through the line profile (utils.line_profile, Gaussian core + exponential wing)
    species:  O2 b band, O2 a band, OI 557.7, OI 630/636/639, NaI D, H
    P:        a Legendre polynomial, for the continuum and additive detector offsets (persistence)
The amplitudes are linear and solved for exactly. Trot, the LSF width (sigma0 + linear slope across the arm), a wavelength shift
and optionally the wing fraction are fitted on top of that.

The band and species amplitudes are relative to RAGNAR's own intensities (in data units per kR), so A_band / (a single scalar)
is the observed / model intensity ratio for each band.
'''
import sys, os.path as osp
import numpy as np
import pandas as pd
from numpy.polynomial import legendre
from scipy.optimize import minimize
from scipy.special import erf

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(osp.dirname(HERE)))
from dev.species_intensities import get_pgopher_linelist, make_OH_intensities, get_O2_at_T, get_atomic_lines
from dev.make_spectra import load_transmission, get_transmission_at_lines

BASE = '/tigress/cj1223/'
R_SCAT, WF = 0.0125, 3.4 # RAGNAR line wing defaults
HALF_WIDTH = 15 # profile half width in units of sigma, as in RAGNAR
# fit ranges per arm, kept away from the dichroic overlaps since the LSF differs between arms
# The b arm starts at 520 nm: below that the PFS sky has emission RAGNAR doesn't model (O2 Herzberg/Chamberlain bands, continuum structure)
ARMS = {'b': (520.0, 640.0, 0.091, 6), 'r': (660.0, 950.0, 0.108, 14), 'n': (990.0, 1250.0, 0.122, 13)}

_A_CACHE = {}

def line_table(Trot, Tvib = 1e4):
    '''All RAGNAR lines with intensities at Trot/Tvib, with a component label: OH(v'-v''), O2b, O2a, OI_green, OI_red, NaI, H'''
    if 'A' not in _A_CACHE:
        _A_CACHE['A'] = get_pgopher_linelist()
        _A_CACHE['atomic'] = get_atomic_lines()
    A = _A_CACHE['A'].copy()
    A['Intensity'] = make_OH_intensities(A, Trot, Tvib)
    A['component'] = 'OH(' + A["v'"].astype(int).astype(str) + '-' + A['v"'].astype(int).astype(str) + ')'
    O2 = get_O2_at_T(Trot)
    O2['component'] = np.where(O2['Position'] < 1000, 'O2b', 'O2a')
    at = _A_CACHE['atomic'].copy()
    at['component'] = at['Species'].map({'OI(1S)': 'OI_green', 'OI(1D)': 'OI_red', 'NaI': 'NaI', 'H': 'H'})
    cols = ['Position', 'Intensity', 'Species', 'component', 'Label', 'Branch', "v'", 'v"', "J'", "N'", 'Eupper', 'A']
    return pd.concat([d.reindex(columns = cols) for d in [A, O2, at]])

def tapered_profile(d, sig, r = R_SCAT, wf = WF, hw = HALF_WIDTH):
    '''utils.line_profile written for 2D arrays of offsets d (lines x pixels) with per-line sigma: unit flux, tapered to 0 at +-hw*sigma'''
    sig = sig[:, None]; gam = wf*sig; H = hw*sig
    P = lambda x: (np.exp(-0.5*(x/sig)**2)/(np.sqrt(2*np.pi)*sig) + r/gam*np.exp(-np.abs(x)/gam))/(1 + 2*r)
    edge = P(H)
    inside = (erf(H/(np.sqrt(2)*sig)) + 2*r*(1 - np.exp(-H/gam)))/(1 + 2*r)
    out = (P(d) - edge)/(inside - 2*H*edge)
    return np.where(np.abs(d) < H, out, 0.0)

class ArmModel:
    def __init__(self, arm, wav, sky, err, airmass, pwv, min_rel_intensity = 1e-6, zd_file = 40, mode = 'vup', minor_frac = 0.01, mask_t = None, mask_pwv = 3.0, level_min_frac = 2e-4):
        '''mode: 'vup' = one OH amplitude per upper vibrational level (the population), 'band' = one per (v', v'') band,
        'level' = one per upper rotational level (v', J', F), for levels carrying more than level_min_frac of the line flux; the weaker levels of
        each v' share one amplitude ("OH(v'=X) rest").
        Components carrying less than minor_frac of the line flux in the arm share a single amplitude ('minor'), since on their own
        they are degenerate with the polynomial.
        mask_t: if set, pixels where lines with MODTRAN transmission < mask_t (at mask_pwv) carry > 5% of the line flux are left out of the fit.
        Inside the H2O bands the line-by-line transmission doesn't match the sky well (small position errors against steep H2O line wings),
        so this keeps the amplitudes, Trot and LSF from being pulled by those lines. The transmission is then checked separately.'''
        self.arm = arm
        lo, hi, self.sig_guess, self.deg = ARMS[arm]
        m = (wav > lo) & (wav < hi) & np.isfinite(sky) & np.isfinite(err) & (err > 0)
        self.wav, self.sky, self.err = wav[m], sky[m], err[m]
        self.dx = np.median(np.diff(wav))
        self.x = 2*(self.wav - lo)/(hi - lo) - 1
        self.poly = legendre.legvander(self.x, self.deg) # additive smooth part
        self.lo, self.hi = lo, hi
        # lines in (and just around) the arm, at a reference temperature for selecting which are bright enough to matter
        L = line_table(195.0)
        L = L[(L['Position'] > lo - 2) & (L['Position'] < hi + 2)]
        L = L[L['Intensity'] > min_rel_intensity*L['Intensity'].max()]
        self.ids = L.index.to_numpy(); self.keys = L.index
        # transmission at each line: MODTRAN at ZD = zd_file, scaled to the observed airmass with t^(X/X0)
        trans = load_transmission('background', float(pwv), zd_file, basedir = BASE, min_wav = lo - 5, max_wav = hi + 5)
        t0 = get_transmission_at_lines(L, trans['nm'].to_numpy(), trans['combin'].to_numpy(), Trot = 195)
        X0 = 1/np.cos(np.deg2rad(zd_file))
        self.trans = np.clip(t0, 1e-6, 1)**(airmass/X0)
        comp = L['component'].to_numpy()
        if mode == 'vup':
            comp = np.array([f"OH(v'={c[3:].split('-')[0]})" if c.startswith('OH(') else c for c in comp])
        if mode == 'level':
            isOH = L['Species'].to_numpy() == 'OH'
            lev = np.array([f"L{int(v)}_{J}_{int(N)}" if o else c for v, J, N, o, c in zip(L["v'"].fillna(-1), L["J'"].fillna(-1), L["N'"].fillna(-1), isOH, comp)])
            lflux = pd.Series(L['Intensity'].to_numpy()*self.trans).groupby(lev).sum()
            weak = isOH & np.isin(lev, lflux.index[lflux < level_min_frac*lflux.sum()])
            comp = np.where(weak, np.array([f"OH(v'={int(v)}) rest" for v in L["v'"].fillna(-1)]), lev)
        flux = pd.Series(L['Intensity'].to_numpy()*self.trans).groupby(comp).sum()
        minor = flux.index[(flux < minor_frac*flux.sum()) & ~flux.index.str.startswith('L')]
        self.minor_members = list(minor)
        # components this weak are degenerate with the polynomial on their own (a free amplitude for them ran off to -500), so they are
        # folded into the strongest OH component: they keep their model intensity relative to it, and carry < minor_frac of the flux
        candidates = flux.drop(minor).filter(regex = '^OH') # fold into a v'-wide OH component; never into a single rotational level
        if len(candidates) == 0:
            minor = minor[:0] # (level mode with no v'-wide component left) keep the weak components as they are
        main = candidates.idxmax() if len(candidates) else None
        if len(minor):
            comp = np.where(np.isin(comp, minor), main, comp)
        self.comp = comp
        self.components = sorted(np.unique(comp), key = lambda c: (c[:2] != 'OH' and c[0] != 'L', c))
        self.L = L
        self.wmax = int(np.ceil(HALF_WIDTH*self.sig_guess*1.6/self.dx)) + 2 # pixels each side covering the profile window
        self.use = np.ones(len(self.wav), dtype = bool)
        if mask_t is not None:
            tr = load_transmission('background', float(mask_pwv), zd_file, basedir = BASE, min_wav = lo - 5, max_wav = hi + 5)
            t_line = np.clip(get_transmission_at_lines(L, tr['nm'].to_numpy(), tr['combin'].to_numpy(), Trot = 195), 1e-6, 1)
            # line-specific absorption: the line's transmission relative to the smooth envelope of the curve (Rayleigh, aerosols and ozone
            # take every line below 0.97, and those are smooth, so they're not the problem)
            from scipy.ndimage import maximum_filter1d
            nm_t, c_t = tr['nm'].to_numpy()[::-1], tr['combin'].to_numpy()[::-1]
            env = maximum_filter1d(c_t, size = max(3, int(1.0/np.median(np.diff(nm_t))))) # upper envelope over 1 nm
            t_env = np.interp(L['Position'].to_numpy(), nm_t, env)
            t_ref = (t_line/np.clip(t_env, 1e-6, None))**(airmass/X0)
            self.t_ref = t_ref
            I0 = L['Intensity'].to_numpy()
            keep = self.trans, self.comp, self.components
            self.trans = np.ones(len(L))
            self.comp = np.where(t_ref < mask_t, 'absorbed', 'clear')
            self.components = ['absorbed', 'clear']
            Bm = self.basis(195.0, 1.05*self.sig_guess, 0.0, -0.025)
            self.trans, self.comp, self.components = keep
            self.use = ~(Bm[0] > 0.05*(Bm[0] + Bm[1]) + 1e-12*Bm.max())
            self.mask_t, self.mask_pwv = mask_t, mask_pwv

    def intensities(self, Trot):
        L = line_table(Trot)
        I = L['Intensity']
        # line_table keeps RAGNAR's indices (OH, O2 +100000, atomic +200000), so look the selected lines up by index
        return I.loc[self.keys].to_numpy()*self.trans

    def basis(self, Trot, sig0, sig1, shift, r = R_SCAT):
        '''Model spectrum of each component on the data grid'''
        I = self.intensities(Trot)
        pos = self.L['Position'].to_numpy() + shift
        sig = sig0*(1 + sig1*(pos - 0.5*(self.lo + self.hi))/(self.hi - self.lo))
        centre = np.searchsorted(self.wav, pos)
        offs = np.arange(-self.wmax, self.wmax + 1)
        idx = centre[:, None] + offs[None, :]
        valid = (idx >= 0) & (idx < len(self.wav))
        idxc = np.clip(idx, 0, len(self.wav) - 1)
        prof = tapered_profile(self.wav[idxc] - pos[:, None], sig, r = r)*valid
        B = np.zeros((len(self.components), len(self.wav)))
        for k, c in enumerate(self.components):
            sel = self.comp == c
            np.add.at(B[k], idxc[sel].ravel(), (I[sel, None]*prof[sel]).ravel())
        return B

    def solve(self, B, sys_frac = 0.01):
        '''Weighted linear least squares for the component amplitudes and the polynomial. sys_frac adds a fractional model-error floor'''
        X = np.vstack([B, self.poly.T]).T
        w = self.use/np.sqrt(self.err**2 + (sys_frac*np.abs(self.sky))**2)
        coef, *_ = np.linalg.lstsq(X*w[:, None], self.sky*w, rcond = None)
        model = X @ coef
        chi2 = np.sum(((self.sky - model)*w)**2)
        return coef, model, chi2

    def fit(self, free_wing = False, Trot0 = 195.0, shift0 = -0.03):
        def unpack(p):
            Trot, sig0, sig1, shift = p[:4]
            r = p[4] if free_wing else R_SCAT
            return Trot, sig0, sig1, shift, r
        def cost(p):
            Trot, sig0, sig1, shift, r = unpack(p)
            if not (120 < Trot < 320 and 0.3*self.sig_guess < sig0 < 3*self.sig_guess and abs(sig1) < 0.5 and abs(shift) < 0.3 and 0 <= r < 0.2):
                return 1e30
            return self.solve(self.basis(Trot, sig0, sig1, shift, r))[2]
        p0 = [Trot0, 1.05*self.sig_guess, 0.0, shift0] + ([R_SCAT] if free_wing else [])
        steps = [10, 0.1*self.sig_guess, 0.05, 0.01] + ([0.01] if free_wing else [])
        simplex = np.array([p0] + [np.array(p0) + np.eye(len(p0))[i]*steps[i] for i in range(len(p0))])
        res = minimize(cost, p0, method = 'Nelder-Mead', options = dict(initial_simplex = simplex, xatol = 1e-5, fatol = 1e-3, maxiter = 3000))
        Trot, sig0, sig1, shift, r = unpack(res.x)
        B = self.basis(Trot, sig0, sig1, shift, r)
        coef, model, chi2 = self.solve(B)
        self.result = dict(Trot = Trot, sig0 = sig0, sig1 = sig1, shift = shift, r_scat = r, chi2 = chi2, npix = int(self.use.sum()), nfev = res.nfev,
                           amplitudes = dict(zip(self.components, coef[:len(self.components)])))
        self.B, self.coef, self.model = B, coef, model
        self.continuum = self.poly @ coef[len(self.components):]
        return self.result
