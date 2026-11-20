'''
Empirical line spread function of the PFS sky lines on native pfsArm pixels, with the Gaussian width and the exponential wing fitted together
(they trade off against each other, so neither can be refitted alone).

Per arm and spectrograph:
  1. strong, isolated, unabsorbed OH Lambda-doublets from the RAGNAR line list (other lines carry < CONTAM of the flux within +-HW sigma,
     no line-specific H2O/O2 absorption)  [CONTAM, HW below]
  2. in every good sky fibre: fit centre, amplitude and a local linear background for each doublet with a provisional profile
     (pfsArm FLUX is the sky without subtraction; the background also absorbs the n-arm persistence offsets)
  3. one joint fit of sigma, r_scat and wf (wing scale / sigma) to all normalized pixels of all doublets and fibres, with the model integrated
     over each pixel; then refit the per-fibre centres/amplitudes with that profile and repeat
Usage: python lsf_pfsarm.py ARM [--fix-wing R_SCAT WF] [--contam C]
  -> lsf_{arm}.json and lsf_{arm}_profile.npz (the stacked, oversampled profile, for plotting)
With --fix-wing only sigma is fitted, at the given wing (the per-line local backgrounds here absorb a faint wing, so the wing itself is
better constrained by the whole-arm fits in sky_model.py); output then goes to lsf_{arm}_fixwing.json
'''
import sys, glob, json, os.path as osp
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.special import erf
from scipy.optimize import least_squares, minimize_scalar
from scipy.ndimage import maximum_filter1d

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(osp.dirname(HERE)))
from sky_model import line_table, BASE
from dev.make_spectra import load_transmission, get_transmission_at_lines

DATA = '/scratch/gpfs/cj1223/pfs_sky'
SKY, GOOD = 2, 1 # pfs.datamodel TargetType.SKY, FiberStatus.GOOD
VISITS = [129688, 129689, 129806, 129807]
DROP = {(129689, 3)}
ARM_RANGE = {'b': (520, 645, 0.088), 'r': (640, 965, 0.108), 'n': (975, 1255, 0.10)}
HW = 10 # half width of the fit window, in sigma
CONTAM = float(sys.argv[sys.argv.index('--contam') + 1]) if '--contam' in sys.argv else 0.02 # largest fraction of foreign line flux allowed within the window
FIX_WING = (float(sys.argv[sys.argv.index('--fix-wing') + 1]), float(sys.argv[sys.argv.index('--fix-wing') + 2])) if '--fix-wing' in sys.argv else None
N_FIBRES = 120 # random good sky fibres per visit and spectrograph, plenty for the stacked profile

def pixel_profile(x, delta, sig, r, wf):
    '''Unit-flux Gaussian core + exponential wing profile, averaged over pixels of width delta centred on x'''
    gam = wf*sig
    a, b = x - delta/2, x + delta/2
    core = 0.5*(erf(b/(np.sqrt(2)*sig)) - erf(a/(np.sqrt(2)*sig)))
    W = lambda t: np.where(t < 0, r*np.exp(t/gam), 2*r - r*np.exp(-t/gam))
    return (core + W(b) - W(a))/delta/(1 + 2*r)

def select_lines(arm):
    lo, hi, sig = ARM_RANGE[arm]
    L = line_table(195.0)
    L = L[(L.Position > lo - 3) & (L.Position < hi + 3)]
    L = L[L.Intensity > 1e-7*L.Intensity.max()]
    tr = load_transmission('background', 2.0, 40, basedir = BASE, min_wav = lo - 5, max_wav = hi + 5)
    t_line = get_transmission_at_lines(L, tr['nm'].to_numpy(), tr['combin'].to_numpy(), Trot = 195)
    nm_t, c_t = tr['nm'].to_numpy()[::-1], tr['combin'].to_numpy()[::-1]
    env = maximum_filter1d(c_t, size = max(3, int(1.0/np.median(np.diff(nm_t)))))
    t_rel = t_line/np.interp(L.Position.to_numpy(), nm_t, env)
    pos, I = L.Position.to_numpy(), L.Intensity.to_numpy()
    OH = L[L.Species == 'OH'].copy()
    OH['key'] = OH["v'"].astype(int).astype(str) + '-' + OH['v"'].astype(int).astype(str) + OH.Branch.str.replace(r'[ef]\(', '(', regex = True)
    out = []
    for key, g in OH.groupby('key'):
        if len(g) > 2 or g.Position.max() - g.Position.min() > 0.3*sig: continue
        p0 = np.average(g.Position, weights = g.Intensity)
        if not (lo + HW*sig < p0 < hi - HW*sig): continue
        own = np.isin(L.index.to_numpy(), g.index.to_numpy())
        near = (np.abs(pos - p0) < HW*sig + 1.0) & ~own
        # flux of other lines within the window, with a generous wing (0.0125) so wings of neighbours count
        d = np.clip(np.abs(pos[near] - p0) - HW*sig, 0, None)
        contam = np.sum(I[near]*np.where(np.abs(pos[near] - p0) < HW*sig, 1.0, 0.0125*np.exp(-d/(3.4*sig))))
        if contam > CONTAM*g.Intensity.sum(): continue
        if np.min(t_rel[own]) < 0.97: continue
        out.append(dict(key = key, pos = p0, I = g.Intensity.sum(), comp_pos = g.Position.to_numpy() - p0, comp_w = (g.Intensity/g.Intensity.sum()).to_numpy()))
    if not out:
        raise RuntimeError(f'no isolated lines in the {arm} arm with CONTAM = {CONTAM}')
    S = pd.DataFrame(out).sort_values('I', ascending = False)
    return S[S.I > 1e-3*S.I.max()].head(40)

def model_line(x, delta, comp_pos, comp_w, mu, A, c0, c1, sig, r, wf):
    prof = sum(w*pixel_profile(x - mu - c, delta, sig, r, wf) for c, w in zip(comp_pos, comp_w))
    return A*prof + c0 + c1*(x - mu)

def fit_one(x, yy, delta, comp_pos, comp_w, mu0, sig, r, wf):
    """Centre by a bounded 1D search; for each centre the amplitude and linear background are an exact linear solve"""
    def inner(mu):
        prof = sum(w*pixel_profile(x - mu - c, delta, sig, r, wf) for c, w in zip(comp_pos, comp_w))
        X = np.column_stack([prof, np.ones_like(x), x - mu])
        coef, *_ = np.linalg.lstsq(X, yy, rcond = None)
        return coef, yy - X @ coef
    res = minimize_scalar(lambda mu: np.sum(inner(mu)[1]**2), bounds = (mu0 - 0.5*sig, mu0 + 0.5*sig), method = 'bounded', options = dict(xatol = 1e-4*sig))
    coef, resid = inner(res.x)
    return res.x, coef, resid

def collect(arm, lines, sig, r, wf, centres = None, seed = 0):
    """Per fibre and line: fit the centre, amplitude and background with a fixed profile. Returns the pixels in a normalized frame"""
    rng = np.random.default_rng(seed)
    pts = []
    for visit in VISITS:
        cfg = fits.open(glob.glob(f'{DATA}/**/pfsConfig_PFS_{visit}_*.fits', recursive = True)[0])['CONFIG'].data
        good = set(np.array(cfg['fiberId'])[(np.array(cfg['targetType']) == SKY) & (np.array(cfg['fiberStatus']) == GOOD)])
        for sm in [1, 2, 3, 4]:
            if (visit, sm) in DROP: continue
            f = glob.glob(f'{DATA}/**/pfsArm_PFS_{visit}_{arm}{sm}_*.fits', recursive = True)
            if not f: continue
            h = fits.open(f[0])
            W, F, N, M = [np.asarray(h[k].data, dtype = float) for k in ['WAVELENGTH', 'FLUX', 'NORM', 'MASK']]
            fib = h['FIBERID'].data
            use = np.nonzero(np.isin(fib, list(good)))[0]
            for k in rng.choice(use, size = min(N_FIBRES, len(use)), replace = False):
                w, y = W[k], F[k]/N[k]
                ok = np.isfinite(y) & (M[k] == 0) & np.isfinite(w)
                dw = np.gradient(w)
                for li, L in lines.iterrows():
                    s_ = ok & (np.abs(w - L.pos) < HW*sig)
                    if s_.sum() < 10: continue
                    x, yy, delta = w[s_], y[s_], dw[s_]
                    mu0 = L.pos + (centres.get((visit, sm, li), -0.02) if centres else -0.02)
                    mu, (A, c0, c1), resid = fit_one(x, yy, delta, L.comp_pos, L.comp_w, mu0, sig, r, wf)
                    if A <= 0 or np.std(resid) == 0: continue
                    peak = A*pixel_profile(np.zeros(1), np.array([np.median(delta)]), sig, r, wf)[0]
                    pts.append(dict(visit = visit, sm = sm, fiber = fib[k], line = li, mu = mu, A = A, snr = peak/np.std(resid),
                                    x = x - mu, y = (yy - c0 - c1*(x - mu))/A, delta = delta))
    return pts

def joint_fit(pts, lines, p0):
    if FIX_WING is not None:
        return joint_fit_sigma(pts, lines, p0)
    X = np.concatenate([p['x'] for p in pts]); Y = np.concatenate([p['y'] for p in pts]); D = np.concatenate([p['delta'] for p in pts])
    L = np.concatenate([np.full(len(p['x']), p['line']) for p in pts])
    def f(par):
        sig, r, wf = par
        m = np.zeros_like(X)
        for li, Lr in lines.iterrows():
            s = L == li
            m[s] = sum(w*pixel_profile(X[s] - c, D[s], sig, r, wf) for c, w in zip(Lr.comp_pos, Lr.comp_w))
        # weight each pixel by the peak-normalized noise: the normalized data is noisier in the wings in relative terms only through
        # the background, so use uniform weights in the normalized frame (bright lines dominate through their many fibres)
        return m - Y
    sol = least_squares(f, p0, bounds = ([0.03, 0.0, 0.5], [0.3, 0.1, 15.0]), x_scale = [0.01, 0.003, 0.5])
    J = sol.jac; cov = np.linalg.pinv(J.T @ J)*np.sum(sol.fun**2)/(len(sol.fun) - 3)
    return sol.x, np.sqrt(np.diag(cov)), X, Y, D

def joint_fit_sigma(pts, lines, p0):
    '''joint_fit with the wing fixed at FIX_WING, fitting sigma only'''
    X = np.concatenate([p['x'] for p in pts]); Y = np.concatenate([p['y'] for p in pts]); D = np.concatenate([p['delta'] for p in pts])
    L = np.concatenate([np.full(len(p['x']), p['line']) for p in pts])
    r, wf = FIX_WING
    def f(par):
        m = np.zeros_like(X)
        for li, Lr in lines.iterrows():
            s = L == li
            m[s] = sum(w*pixel_profile(X[s] - c, D[s], par[0], r, wf) for c, w in zip(Lr.comp_pos, Lr.comp_w))
        return m - Y
    sol = least_squares(f, [p0[0]], bounds = ([0.03], [0.3]), x_scale = [0.01])
    J = sol.jac; cov = np.linalg.pinv(J.T @ J)*np.sum(sol.fun**2)/(len(sol.fun) - 1)
    return np.array([sol.x[0], r, wf]), np.array([np.sqrt(cov[0, 0]), 0, 0]), X, Y, D

if __name__ == '__main__':
    arm = sys.argv[1]
    lines = select_lines(arm)
    sig0 = ARM_RANGE[arm][2]
    print(f'{arm}: {len(lines)} isolated doublets: ' + ', '.join(f'{p:.2f}' for p in lines.pos), flush = True)
    par = np.array([sig0, *(FIX_WING if FIX_WING else (0.0125, 3.4))]); centres = None
    for it in range(3):
        pts = collect(arm, lines, *par, centres = centres)
        pts = [p for p in pts if p['snr'] > 30]
        centres = {(p['visit'], p['sm'], p['line']): p['mu'] - lines.loc[p['line'], 'pos'] for p in pts}
        results = {}
        for sm in [1, 2, 3, 4, 'all']:
            sel = [p for p in pts if sm == 'all' or p['sm'] == sm]
            if len(sel) < 20: continue
            (s, r, wf), (es, er, ewf), X, Y, D = joint_fit(sel, lines, par)
            results[str(sm)] = dict(sigma = s, sigma_err = es, r_scat = r, r_scat_err = er, wf = wf, wf_err = ewf, n_profiles = len(sel),
                                    pixel = float(np.median(D)), sigma_eff = float(np.sqrt(s**2 + np.median(D)**2/12)))
            print(f'  iteration {it}, SM {sm}: sigma {s:.4f}+-{es:.4f} nm (incl. pixel {np.sqrt(s**2 + np.median(D)**2/12):.4f}), r_scat {r:.4f}+-{er:.4f}, wf {wf:.2f}+-{ewf:.2f}, {len(sel)} line profiles', flush = True)
        par = np.array([results['all']['sigma'], results['all']['r_scat'], results['all']['wf']])
    # the same joint fit with wf fixed at 3.4, for comparison with RAGNAR's parametrization
    sel = pts
    X = np.concatenate([p['x'] for p in sel]); Y = np.concatenate([p['y'] for p in sel]); D = np.concatenate([p['delta'] for p in sel])
    Lidx = np.concatenate([np.full(len(p['x']), p['line']) for p in sel])
    def f2(q):
        m = np.zeros_like(X)
        for li, Lr in lines.iterrows():
            s_ = Lidx == li
            m[s_] = sum(w*pixel_profile(X[s_] - c, D[s_], q[0], q[1], 3.4) for c, w in zip(Lr.comp_pos, Lr.comp_w))
        return m - Y
    sol = least_squares(f2, par[:2], bounds = ([0.03, 0.0], [0.3, 0.1])) if FIX_WING is None else None
    if FIX_WING is None: results['all_wf3.4'] = dict(sigma = sol.x[0], r_scat = sol.x[1], wf = 3.4, sigma_eff = float(np.sqrt(sol.x[0]**2 + np.median(D)**2/12)))
    if FIX_WING is None: print(f'  wf fixed at 3.4: sigma {sol.x[0]:.4f}, r_scat {sol.x[1]:.4f}', flush = True)
    # stacked profile for plotting: bin the normalized pixels in offset/sigma
    s, r, wf = par
    bins = np.linspace(-HW, HW, 241); c = 0.5*(bins[1:] + bins[:-1])
    idx = np.digitize(X/s, bins) - 1; ok = (idx >= 0) & (idx < len(c))
    med = np.array([np.median(Y[ok][idx[ok] == i]) if np.any(idx[ok] == i) else np.nan for i in range(len(c))])
    tag = '_fixwing' if FIX_WING else ''
    np.savez(osp.join(HERE, f'lsf_{arm}{tag}_profile.npz'), offset_sigma = c, profile = med, sigma = s, r_scat = r, wf = wf, pixel = float(np.median(D)))
    json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, open(osp.join(HERE, f'lsf_{arm}{tag}.json'), 'w'), indent = 1)
