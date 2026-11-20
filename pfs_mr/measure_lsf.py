'''
The PFS medium-resolution (m arm) line spread function from sky lines, 2026-11-16.

Data: /scratch/gpfs/cj1223/pfs_sky_medium, 6 consecutive 60 s windowed visits (150805-150810, airmass 1.16), sky fibres of m1-m4.
Only ~722-770 nm is read out. The 6 visits share the wavelength grid, so the counts are summed per fibre.

Each bright, isolated OH line (Lambda-doublet, both components) is fitted per fibre on native pixels, with the profile integrated over each
pixel exactly: a Gaussian core plus RAGNAR's exponential wing, (G(sigma) + r/gamma exp(-|x|/gamma))/(1 + 2r), gamma = wf*sigma.
Neighbouring lines inside the window are free-amplitude components at their model positions; the target's e/f weights come from the model
(fiducial airglow, Mauna Kea PWV 1 mm, ZD 30). Free per fit: shift, sigma, amplitudes, linear background. The wing (r, wf) is fitted globally.

sigma is reported as fitted (intrinsic, pixel integrated separately) and including the pixel, sqrt(sigma^2 + dlam^2/12), which is the
convention of the LR lists in run_RAGNAR.py.

Run from RAGNAR_new/: python pfs_mr/measure_lsf.py [--ncpu 16]
'''
import os, sys, glob, json, argparse, itertools
import numpy as np
import pandas as pd
import os.path as osp
from astropy.io import fits
from scipy.special import erf
from scipy.optimize import least_squares

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(osp.dirname(HERE), 'sedm')); sys.path.insert(0, osp.dirname(HERE))
import analyze_sedm as AS # line table with Lambda-doublet labels and MODTRAN transmission

DATA = '/scratch/gpfs/cj1223/pfs_sky_medium/'
OUT = osp.join(HERE, 'out')
SPECS = [1, 2, 3, 4]
TARGETS = ['(8-3)pP1(2.5)', '(8-3)pP1(3.5)', '(8-3)pP1(4.5)', '(8-3)pP1(5.5)'] # bright and isolated at sigma ~ 0.063 nm (model check, 2026-11-16)
CORE, FULL = 0.30, 0.90 # half windows [nm]: the core fit for sigma, and the full fit including the wing
SIG0 = 0.063
WING_R = [0.0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.0125, 0.015, 0.02, 0.03]
WING_WF = [2.0, 3.4, 5.0, 8.0]

def load(sp):
    ''' visit-summed counts and variances of the sky fibres of one spectrograph, NaN where any visit is masked '''
    fs = sorted(glob.glob(f'{DATA}pfsArm-*-m{sp}-sky.fits'))
    with fits.open(fs[0]) as h:
        W, fid = h['WAVELENGTH'].data.astype(float), h['FIBERID'].data
    F, V = 0.0, 0.0
    for f in fs:
        with fits.open(f) as h:
            assert np.array_equal(h['WAVELENGTH'].data, W)
            bad = h['MASK'].data != 0
            F = F + np.where(bad, np.nan, h['FLUX'].data.astype(float))
            V = V + np.where(bad, np.nan, h['COVAR'].data[:, 0, :].astype(float))
    return fid, W, F, V, len(fs)

def edges(w):
    m = 0.5*(w[1:] + w[:-1])
    return np.r_[2*w[0] - m[0], m, 2*w[-1] - m[-1]]

def pix_profile(e, mu, sig, r, wf):
    ''' [pixel, component] fraction of a unit line at mu falling in each pixel (edges e), Gaussian core + exponential wing, exactly integrated '''
    a, b = e[:-1, None] - mu[None, :], e[1:, None] - mu[None, :]
    g = 0.5*(erf(b/(np.sqrt(2)*sig)) - erf(a/(np.sqrt(2)*sig)))
    if r == 0:
        return g
    gam = wf*sig
    lap = lambda x: np.where(x < 0, 0.5*np.exp(np.minimum(x, 0)/gam), 1 - 0.5*np.exp(-np.maximum(x, 0)/gam))
    return (g + 2*r*(lap(b) - lap(a)))/(1 + 2*r)

class Model:
    ''' the line components around each target: positions, groups (Lambda doublets) and model weights '''
    def __init__(self):
        L = AS.line_table()
        T, _ = AS.line_transmission(L, 1.0, 30)
        self.pos, self.I, self.trans = L.Position.to_numpy(), L.Intensity.to_numpy()*T*1e3, L.trans.to_numpy()
        self.comp = {}
        for t in TARGETS:
            mt = self.trans == t
            mu0 = np.sum(self.pos[mt]*self.I[mt])/np.sum(self.I[mt])
            near = np.abs(self.pos - mu0) < FULL + 0.3
            groups, gidx = np.unique(self.trans[near], return_inverse = True)
            Ig = np.bincount(gidx, weights = self.I[near])
            It = Ig[list(groups).index(t)]
            # free amplitude for the target and every neighbour group above 0.3% of it; the rest scale with the target
            gpos = np.bincount(gidx, weights = self.I[near]*self.pos[near])/Ig
            # free amplitude for every neighbour group above 0.3% of the target that is more than ~2 sigma away; closer ones would trade
            # against sigma (they made the per-line widths disagree by 5%), so they scale with the target at their model weights
            free = (Ig > 0.003*It) & (np.abs(gpos - mu0) > 2*SIG0)
            order = np.r_[[list(groups).index(t)], [k for k in np.nonzero(free)[0] if groups[k] != t]]
            colmap = {k: i for i, k in enumerate(order)}
            ci = np.array([colmap.get(k, 0) for k in gidx])
            w = self.I[near]/It # weights relative to the target's total
            self.comp[t] = dict(mu0 = mu0, pos = self.pos[near], col = ci, w = w, ncol = len(order), names = [groups[k] for k in order])

def fit_one(x, e, d, v, cm, r, wf, half, sig0 = SIG0):
    ''' fit one line in one fibre: nonlinear shift and sigma, linear amplitudes and background '''
    m = np.abs(x - cm['mu0']) < half
    if m.sum() < 8 or not np.all(np.isfinite(d[m])) or not np.all(np.isfinite(v[m])) or np.any(v[m] <= 0):
        return None
    xm, dm, wt = x[m], d[m], 1/np.sqrt(v[m])
    em = e[np.r_[m, False] | np.r_[False, m]]
    def design(th):
        P = pix_profile(em, cm['pos'] + th[0], th[1], r, wf)*cm['w'][None, :]
        Gm = np.zeros((m.sum(), cm['ncol']))
        np.add.at(Gm.T, cm['col'], P.T)
        return np.column_stack([Gm, np.ones(m.sum()), xm - cm['mu0']])
    def res(th):
        X = design(th)
        c = np.linalg.lstsq(X*wt[:, None], dm*wt, rcond = None)[0]
        return (X @ c - dm)*wt
    try:
        rr = least_squares(res, [0.0, sig0], bounds = ([-0.08, 0.025], [0.08, 0.15]), x_scale = [0.005, 0.005])
    except Exception:
        return None
    X = design(rr.x); c = np.linalg.lstsq(X*wt[:, None], dm*wt, rcond = None)[0]
    J = rr.jac
    try:
        cov = np.linalg.inv(J.T @ J)*max(1.0, 2*rr.cost/max(1, m.sum() - len(c) - 2))
        esig = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        esig = np.nan
    return dict(shift = rr.x[0], sigma = rr.x[1], e_sigma = esig, amp = c[0], chi2 = 2*rr.cost, npix = int(m.sum()), dlam = float(np.median(np.diff(xm))))

DATAS, MODEL = {}, None

def work(args):
    ''' all fits of one spectrograph for one wing setting '''
    sp, r, wf, half = args
    fid, W, F, V, nv = DATAS[sp]
    out = []
    for k, t in itertools.product(range(len(fid)), TARGETS):
        x = W[k]; e = edges(x)
        f = fit_one(x, e, F[k], V[k], MODEL.comp[t], r, wf, half)
        if f is not None:
            out.append(dict(spec = sp, fiberId = int(fid[k]), line = t, mu0 = MODEL.comp[t]['mu0'], r = r, wf = wf, half = half, **f))
    return out

if __name__ == '__main__':
    from multiprocessing import Pool
    ap = argparse.ArgumentParser(); ap.add_argument('--ncpu', type = int, default = 1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok = True)
    MODEL = Model()
    for t in TARGETS:
        print(t, 'mu0', round(MODEL.comp[t]['mu0'], 4), 'free columns', MODEL.comp[t]['names'])
    for sp in SPECS:
        DATAS[sp] = load(sp)
    with Pool(a.ncpu) as pool:
        # 1) the wing: total chi2 of the full-window fits on a grid of (r, wf)
        grid = list(itertools.product(WING_R, WING_WF))
        jobs = [(sp, r, wf, FULL) for (r, wf) in grid for sp in SPECS if not (r == 0 and wf != WING_WF[0])]
        res = pool.map(work, jobs)
        G = pd.DataFrame([x for rr in res for x in rr])
        # compare on the same set of (fibre, line) fits
        key = G.groupby(['spec', 'fiberId', 'line']).size()
        full = key[key == key.max()].index
        G = G.set_index(['spec', 'fiberId', 'line']).loc[full].reset_index()
        chi = G.groupby(['r', 'wf']).agg(chi2 = ('chi2', 'sum'), npix = ('npix', 'sum'), sigma_med = ('sigma', 'median')).reset_index()
        chi['dchi2'] = chi.chi2 - chi.chi2.min()
        chi.to_csv(osp.join(OUT, 'wing_grid.csv'), index = False)
        print(chi.sort_values('dchi2').head(12).to_string())
        best = chi.loc[chi.chi2.idxmin()]
        # per spectrograph too
        chs = G.groupby(['spec', 'r', 'wf']).chi2.sum().reset_index()
        bests = {int(s): chs[chs.spec == s].sort_values('chi2').iloc[0][['r', 'wf']].tolist() for s in SPECS}
        print('best wing per spectrograph', bests)
        # 2) sigma with the best wing, core window and full window, and with no wing
        jobs = [(sp, best.r, best.wf, h) for sp in SPECS for h in (CORE, FULL)] + [(sp, 0.0, 2.0, CORE) for sp in SPECS]
        res = pool.map(work, jobs)
    S = pd.DataFrame([x for rr in res for x in rr])
    S['sigma_pix'] = np.sqrt(S.sigma**2 + S.dlam**2/12) # including the pixel
    S.to_csv(osp.join(OUT, 'lsf_fits.csv'), index = False)
    sel = S[(S.r == best.r) & (S.wf == best.wf) & (S.half == CORE) & (S.e_sigma < 0.01)]
    summ = dict(wing = dict(r = float(best.r), wf = float(best.wf)), wing_per_spec = bests,
                per_spec = {int(s): dict(sigma = float(g.sigma.median()), sigma_pix = float(g.sigma_pix.median()),
                                         err = float(1.2533*g.sigma_pix.std()/np.sqrt(len(g))), n = int(len(g))) for s, g in sel.groupby('spec')},
                per_line = {l: dict(sigma_pix = float(g.sigma_pix.median()), err = float(1.2533*g.sigma_pix.std()/np.sqrt(len(g))), mu0 = float(g.mu0.iloc[0])) for l, g in sel.groupby('line')})
    summ['sigma_pix_mean'] = float(np.mean([v['sigma_pix'] for v in summ['per_spec'].values()]))
    for h in (CORE, FULL):
        q = S[(S.r == best.r) & (S.wf == best.wf) & (S.half == h)]
        summ[f'sigma_pix_all_half{h}'] = float(q.sigma_pix.median())
    q = S[(S.r == 0) & (S.half == CORE)]; summ['sigma_pix_nowing_core'] = float(q.sigma_pix.median())
    json.dump(summ, open(osp.join(OUT, 'lsf_summary.json'), 'w'), indent = 1)
    print(json.dumps(summ, indent = 1))
