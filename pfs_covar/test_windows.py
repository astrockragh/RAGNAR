'''
Is every feature in a window the same in all exposures up to one scalar per exposure? RAGNAR's prediction against the PFS sky.

Model (dev/covariation.linear_windows, intercept = False): in a window, each simulated sky is A_s*shape(lambda); E_worst is the worst
feature's rms residual (over its pixels and all skies) relative to the rms of its own mean spectrum, E_window the same over the window.

Data, per spectrograph, over that spectrograph's exposures v of the same window:
    d_v(lambda) = A_v*g(lambda) + B_v*g'(lambda) + alpha_v + beta_v*(lambda - lambda_0)
g is one line shape common to all exposures (free per pixel), A_v the scalar, alpha_v + beta_v*lambda a linear continuum per exposure (the
continuum is not part of the test) and B_v*g' a small wavelength shift per exposure (the n-arm stacks move by up to 13 pm between visits).
Fitted by alternating least squares with the stack errors. Reported: chi2/dof, and per feature (the model's features, mapped onto the data
pixels) the rms residual over its pixels and exposures beyond the noise, relative to the rms of the fitted line shape: the same quantity as
the model's E_worst (features with at least 2% of the window's line flux), but for the two nights (LR) or the 9 minutes (m) actually
observed rather than the whole grid of skies. The noise taken out is scaled by the median within-night chi2/dof (LR; m: the median over
windows), so the numbers do not count underestimated errors as shape changes.
  LR (b, r, n): pfsMerged per-spectrograph sky stacks of the four visits (validation/tier2), 0.08 nm pixels, two nights.
  m: per-visit, per-spectrograph stacks of the sky fibres on a 0.016 nm grid, from the pfsArm pixels with each fibre's wavelength
     correction (pfs_mr/test_linelist.py), six 60 s visits within 9 minutes, 724-767 nm.

For LR the same test is also run within each night (two visits 16 minutes apart, nearly the same sky; n1_*, n2_*), which measures the
instrumental floor: about 0.1% for clean windows, against 1-2% between the nights.

Run from RAGNAR_new/: python pfs_covar/test_windows.py [arm ...]  ->  pfs_covar/out/windows_{arm}_5-10nm.csv (every window, model and data)
'''
import sys, glob, pickle, time
import numpy as np
import pandas as pd
import os.path as osp
from astropy.io import fits

HERE = osp.dirname(osp.abspath(__file__)); ROOT = osp.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, osp.join(ROOT, 'pfs_mr'))
from dev.covariation import linear_windows, bin_spectra, PARAM_NAMES
from measure import RUNS, DATE, SPEC_DIR, OUT, VISITS_LR, STACKS, MR_LIST
from sections import ARM_RANGE

W0, W1 = 5.0, 10.0
MR_RANGE, MR_PIX = (724.0, 767.0), 0.016

def fit_window(x, D, E, n_iter = 40):
    '''
    D, E: (n_exp, n_pix) data and errors (NaN = missing). Returns (residuals, model line part A_v*g, chi2, dof) or None.
    '''
    ok = np.all(np.isfinite(D) & np.isfinite(E) & (E > 0), axis = 0) # pixels present in every exposure
    if ok.sum() < 12:
        return None
    x, D, W = x[ok], D[:, ok], 1/E[:, ok]**2
    nv, npix = D.shape
    t = (x - x.mean())/max(np.ptp(x), 1e-9)
    base = np.vstack([np.ones(npix), t])
    # start: the shape from the exposure-mean minus its lower envelope, continuum from the fit
    g = D.mean(axis = 0); g = g - np.percentile(g, 10)
    A, B, C = np.ones(nv), np.zeros(nv), np.zeros((nv, 2))
    for it in range(n_iter):
        gp = np.gradient(g, x)
        for v in range(nv): # per exposure: scalar, shift and continuum
            M = np.vstack([g, gp, base]).T*np.sqrt(W[v])[:, None]
            c = np.linalg.lstsq(M, D[v]*np.sqrt(W[v]), rcond = None)[0]
            A[v], B[v], C[v] = c[0], c[1], c[2:]
        R = D - B[:, None]*gp[None, :] - C @ base # what the common shape has to explain
        g = np.sum(W*A[:, None]*R, axis = 0)/np.sum(W*A[:, None]**2, axis = 0)
        s = A.mean(); A, B, g = A/s, B/s, g*s
    gp = np.gradient(g, x)
    line = A[:, None]*g[None, :]
    res = D - line - B[:, None]*gp[None, :] - C @ base
    chi2 = float(np.sum(res**2*W))
    dof = nv*npix - npix - 4*nv + 3 # g per pixel; scalar, shift and two continuum terms per exposure; scale and linear-in-g degeneracies
    full = np.full((nv, len(ok)), np.nan); full[:, ok] = res
    lf = np.full((nv, len(ok)), np.nan); lf[:, ok] = line
    return full, lf, chi2, dof

def test_windows(Tw, x, sets, noise_scale = 1.0, min_frac = 0.02):
    '''
    Tw: the model's windows (with feature edges and flux shares); x: data pixel centres; sets: list of (D, E) per spectrograph.
    For each window: chi2/dof summed over spectrographs, and the rms residual beyond the noise relative to the line flux, per feature (the
    worst among the features with at least min_frac of the window's model line flux, as for the model) and over the whole window.
    noise_scale multiplies the noise variance taken out (the within-night chi2/dof, so the between-night numbers are conservative).
    '''
    out = []
    for _, w in Tw.iterrows():
        e = np.asarray(w.edges)
        sel = (x >= e[0]) & (x < e[-1])
        feat = np.searchsorted(e, x[sel], side = 'right') - 1
        c2 = dof = 0.0; r2 = np.zeros(len(e) - 1); n2 = np.zeros(len(e) - 1); l2 = np.zeros(len(e) - 1)
        for D, E in sets:
            f = fit_window(x[sel], D[:, sel], E[:, sel])
            if f is None:
                continue
            res, line, chi2, d = f
            c2 += chi2; dof += d
            frac = d/np.sum(np.isfinite(res)) # share of the noise variance left in the residuals
            for k in range(len(e) - 1):
                m = (feat == k)[None, :] & np.isfinite(res)
                if m.sum():
                    r2[k] += np.sum(res[m]**2); n2[k] += noise_scale*np.sum(E[:, sel][m]**2)*frac; l2[k] += np.sum(line[m]**2)
        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            exc = np.sqrt(np.maximum(r2 - n2, 0)/l2)
            noise = np.sqrt(n2/l2)
        good = (l2 > 0) & (np.asarray(w.feat_frac) >= min_frac)
        out.append(dict(chi2_dof = c2/dof if dof > 0 else np.nan, obs_worst = float(np.nanmax(np.where(good, exc, np.nan))) if good.any() else np.nan,
                        obs_window = float(np.sqrt(max(r2.sum() - n2.sum(), 0)/l2.sum())) if l2.sum() > 0 else np.nan,
                        noise_worst = float(np.nanmax(np.where(good, noise, np.nan))) if good.any() else np.nan,
                        obs_feat = np.round(np.where(good, exc, np.nan), 4).tolist()))
    return pd.DataFrame(out, index = Tw.index)

def model_windows(arm, lo = None, hi = None):
    name, sig = RUNS[arm]
    P = np.load(osp.join(ROOT, 'lineLists', f'{name}_{DATE}_covar.npz'))['params']
    nom = np.isclose(P[:, PARAM_NAMES.index('lsf_factor')], 1.0)
    wr, specs = pickle.load(open(osp.join(SPEC_DIR, f'sweep_0_{sig}_{arm}_fix4_{DATE}.pkl'), 'rb'))
    lo = max(lo or ARM_RANGE[arm][0], wr[0] + sig); hi = min(hi or ARM_RANGE[arm][1], wr[-1] - sig)
    xm, Y = bin_spectra(wr, [specs[i] for i in np.flatnonzero(nom)], lo, hi, sig/2)
    del specs
    T = linear_windows(xm, Y, W0, W1)
    L = pd.read_pickle(osp.join(ROOT, 'lineLists', f'{name}_{DATE}.pkl'))
    rob = L[L.robust.astype(bool)].sort_values('wav')
    lw, li = rob.wav.to_numpy(), rob.intensity.to_numpy()
    ab = rob.transmission_origin.astype(str).map(lambda o: o[3] == '1' or '1' in o[5:]).to_numpy()
    ci, ca = np.r_[0, np.cumsum(li)], np.r_[0, np.cumsum(li*ab)]
    a, z = np.searchsorted(lw, T.lo.to_numpy()), np.searchsorted(lw, T.hi.to_numpy(), side = 'right')
    T['absorbed'] = np.where(ci[z] > ci[a], (ca[z] - ca[a])/np.maximum(ci[z] - ci[a], 1e-30), 0.0)
    T['mostly'] = [rob.iloc[i:j].groupby('vgroup').intensity.sum().idxmax() if j > i else '' for i, j in zip(a, z)]
    T['n_lines'] = z - a
    return T

def lr_sets(arm, visits = VISITS_LR):
    ''' per spectrograph, the stacks of the given visits that exist (129689 SM3 was dropped); spectrographs with fewer than two are skipped '''
    sets = []
    for sm in [1, 2, 3, 4]:
        fs = [osp.join(STACKS, f'sky_stack_{v}_sm{sm}.npz') for v in visits]
        d = [np.load(f) for f in fs if osp.exists(f)]
        if len(d) < 2:
            continue
        x = d[0]['wav']
        sets.append((np.vstack([q['sky'] for q in d]), np.vstack([q['err'] for q in d])))
    return x, sets

def mr_sets():
    import measure_lsf as ML, test_linelist as TL
    _, _, corrs = TL.evaluate(MR_LIST, 0.0637, TL.load_all())
    grid = np.arange(MR_RANGE[0], MR_RANGE[1], MR_PIX)
    sets = []
    for sp in ML.SPECS:
        Ds, Es = [], []
        for f in sorted(glob.glob(f'{ML.DATA}pfsArm-*-m{sp}-sky.fits')):
            with fits.open(f) as h:
                fid, Wl = h['FIBERID'].data, h['WAVELENGTH'].data.astype(float)
                F, N = h['FLUX'].data.astype(float), h['NORM'].data.astype(float)
                V, bad = h['COVAR'].data[:, 0, :].astype(float), h['MASK'].data != 0
            num, den = np.zeros(len(grid)), np.zeros(len(grid))
            for k in range(len(fid)):
                c = corrs.get((sp, int(fid[k])))
                if c is None:
                    continue
                tt = (Wl[k] - 745)/25
                xc = Wl[k] - (c[0] + c[1]*tt + c[2]*tt**2)
                ok = ~bad[k] & (N[k] > 0) & (V[k] > 0) & (xc > grid[0]) & (xc < grid[-1] + MR_PIX)
                bi = np.clip(np.round((xc[ok] - grid[0])/MR_PIX).astype(int), 0, len(grid) - 1)
                y, vy = F[k][ok]/N[k][ok], V[k][ok]/N[k][ok]**2
                np.add.at(num, bi, y/vy); np.add.at(den, bi, 1/vy)
            Ds.append(np.where(den > 0, num/np.maximum(den, 1e-30), np.nan)); Es.append(np.where(den > 0, 1/np.sqrt(np.maximum(den, 1e-30)), np.nan))
        sets.append((np.vstack(Ds), np.vstack(Es)))
    return grid, sets

if __name__ == '__main__':
    for arm in (sys.argv[1:] or ['b', 'r', 'n', 'm']):
        t = time.time()
        if arm == 'm':
            Tw = model_windows('m', MR_RANGE[0], MR_RANGE[1])
            x, sets = mr_sets()
        else:
            Tw = model_windows(arm)
            x, sets = lr_sets(arm)
        if arm != 'm': # the same test within each night first (16 minutes apart, nearly the same sky): the floor and the noise scale
            Rn = {nm: test_windows(Tw, *lr_sets(arm, vis)) for nm, vis in [('n1', VISITS_LR[:2]), ('n2', VISITS_LR[2:])]}
            scale = float(np.nanmedian(np.r_[Rn['n1'].chi2_dof, Rn['n2'].chi2_dof]))
        else: # one night: the median chi2/dof over the windows, i.e. assuming most windows do not change in 9 minutes
            scale = float(np.nanmedian(test_windows(Tw, x, sets).chi2_dof))
        R = test_windows(Tw, x, sets, noise_scale = max(scale, 1.0))
        T = Tw.join(R)
        T['noise_scale'] = max(scale, 1.0)
        if arm != 'm':
            for nm in ['n1', 'n2']:
                T[f'{nm}_window'], T[f'{nm}_worst'], T[f'{nm}_chi2_dof'] = Rn[nm].obs_window, Rn[nm].obs_worst, Rn[nm].chi2_dof
        T.insert(0, 'arm', arm)
        T.to_csv(osp.join(OUT, f'windows_{arm}_{W0:g}-{W1:g}nm.csv'), index = False)
        print(f'{arm}: {len(T)} windows, data tested {np.isfinite(T.chi2_dof).sum()}; median chi2/dof {np.nanmedian(T.chi2_dof):.2f}, noise scale {max(scale, 1.0):.2f} ({time.time() - t:.0f} s)', flush = True)
