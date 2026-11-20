'''
Line amplitudes in the PFS sky, measured the same way in the data and in RAGNAR's simulated spectra, to test which lines vary together.

The amplitude of a line is the least-squares amplitude of a Gaussian of the arm's LSF sigma at a fixed centre (the cent=0 list wavelength,
plus a wavelength correction in the data), fitted within +-CORE sigma, above a local baseline. Apart from the pixel phase, which it removes,
this is the raw peak height above the baseline, i.e. what the cent=0 list's intensity column is.

The baseline is there because RAGNAR's continuum is not reliable. On each side of the line, between GAP sigma and W from it, it is the mean of
the faintest fraction Q of the pixels, and the two sides are joined by a straight line. The same operator is applied to RAGNAR's line-only
spectra, so a smooth continuum in the data drops out, and what the baseline picks up from line wings and faint neighbours is the same in
data and model. VARIANTS change (W, Q) and drop the baseline, to show how much the continuum treatment matters.

LR (b, r, n): the per-spectrograph median sky stacks of the four all-sky visits (validation/tier2/sky_stack_{visit}_sm{sm}.npz, from pfsMerged
on a 0.08 nm grid; the files the LR LSFs were calibrated on), with one wavelength shift per stack and arm from the brighter lines.
MR (m): per fibre on the pfsArm pixels of each of the 6 visits, at the list wavelength plus that fibre's correction (pfs_mr/test_linelist.py),
then the median over the sky fibres of each spectrograph, with the error from their scatter.
Model: each simulated spectrum of the cent=0 run, averaged into the data's pixels (0.08 nm for LR, 0.0468 nm for m).

Run from RAGNAR_new/: python pfs_covar/measure.py   (needs ~8 GB for the simulated spectra of one arm at a time)
'''
import sys, glob, pickle, time
import numpy as np
import pandas as pd
import os.path as osp
from astropy.io import fits

HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.dirname(HERE)
OUT = osp.join(HERE, 'out')
sys.path.insert(0, osp.join(ROOT, 'pfs_mr'))

DATE = '20261120'
RUNS = {'b': ('sweep_0.088_0_b_fix4', 0.088), 'r': ('sweep_0.112_0_r_fix4', 0.112), 'n': ('sweep_0.105_0_n_fix4', 0.105), 'm': ('sweep_0.0637_0_m_fix4', 0.0637)}
SPEC_DIR = '/tigress/cj1223/spectra/'
LR_RANGE = {'b': (400.0, 635.0), 'r': (640.0, 965.0), 'n': (965.0, 1262.0)} # the merged stacks: each arm where it has the spectrum to itself
MR_RANGE = (724.0, 767.0) # the windowed m-arm readout
VISITS_LR = [129688, 129689, 129806, 129807]
STACKS = osp.join(ROOT, 'validation', 'tier2')
MR_LIST = osp.join(ROOT, 'lineLists', 'sweep_0.0637_1_m_fix4_691-908nm_20261116.csv') # the list the per-fibre corrections were made with
LR_PIX, MR_PIX = 0.08, 0.0468
GAP, CORE = 2.5, 1.0
# baseline variants: half window W in units of sigma, faintest fraction Q; None = no baseline (the continuum stays in the amplitude)
VARIANTS = {'base': (14.0, 0.3), 'wide': (25.0, 0.2), 'narrow': (8.0, 0.4), 'none': None}

def faint_mean(xs, Ys, Q):
    ''' mean value and mean wavelength of the faintest fraction Q of the pixels, per row of Ys (NaNs sort last and are skipped) '''
    nfin = np.sum(np.isfinite(Ys), axis = 1)
    n = np.maximum(2, np.round(Q*nfin).astype(int))
    idx = np.argsort(Ys, axis = 1)
    sel = np.arange(Ys.shape[1])[None, :] < n[:, None]
    v = np.take_along_axis(Ys, idx, 1)
    return np.nansum(np.where(sel, v, 0), 1)/n, np.sum(np.where(sel, xs[idx], 0), 1)/n

def amplitudes(x, Y, lams, sig, variant = 'base', var = None):
    '''
    x: (npix,) ascending, Y: (nspec, npix), lams: (nline,) centres, sig: LSF sigma [nm]
    Returns the amplitudes (nspec, nline), their errors from the core pixels if var (npix,) is given (NaN where data are missing),
    and the baseline at the line centre (0 without a baseline)
    '''
    Y = np.atleast_2d(Y)
    A = np.full((len(Y), len(lams)), np.nan); E = np.full_like(A, np.nan); B0 = np.zeros_like(A)
    par = VARIANTS[variant]
    for k, lam in enumerate(lams):
        c0, c1 = np.searchsorted(x, [lam - CORE*sig, lam + CORE*sig])
        if c1 - c0 < 2:
            continue
        xc = x[c0:c1]; g = np.exp(-0.5*((xc - lam)/sig)**2); gg = g @ g
        if par is None:
            base = 0.0
        else:
            W, Q = par[0]*sig, par[1]
            l0, l1 = np.searchsorted(x, [lam - W, lam - GAP*sig]); r0, r1 = np.searchsorted(x, [lam + GAP*sig, lam + W])
            if l1 - l0 < 4 or r1 - r0 < 4:
                continue
            bl, xl = faint_mean(x[l0:l1], Y[:, l0:l1], Q); br, xr = faint_mean(x[r0:r1], Y[:, r0:r1], Q)
            base = bl[:, None] + ((br - bl)/(xr - xl))[:, None]*(xc[None, :] - xl[:, None])
            B0[:, k] = bl + (br - bl)/(xr - xl)*(lam - xl)
        A[:, k] = ((Y[:, c0:c1] - base) @ g)/gg # NaN if a core pixel is missing
        if var is not None:
            E[:, k] = np.sqrt(var[c0:c1] @ g**2)/gg
    return A, E, B0

def best_shift(x, y, lams, sig, var):
    ''' one wavelength shift for a stack: the one that maximizes the summed amplitude of the brighter lines, refined with a parabola '''
    A0, E0, _ = amplitudes(x, y, lams, sig, 'base', var)
    snr = A0[0]/E0[0]
    use = np.isfinite(snr) & (snr > np.nanpercentile(snr, 70))
    sh = np.arange(-0.03, 0.0301, 0.002)
    tot = np.array([np.nansum(amplitudes(x, y, lams[use] + s, sig, 'base')[0]) for s in sh])
    k = int(np.clip(np.argmax(tot), 1, len(sh) - 2))
    a, b, c = tot[k-1:k+2]
    return sh[k] + 0.5*(a - c)/(a - 2*b + c)*(sh[1] - sh[0])

def load_list(arm):
    name, sig = RUNS[arm]
    return pd.read_pickle(osp.join(ROOT, 'lineLists', f'{name}_{DATE}.pkl')), sig

def measure_lr():
    rows = []
    for arm in 'brn':
        L, sig = load_list(arm)
        lo, hi = LR_RANGE[arm]
        sel = np.flatnonzero((L.wav > lo) & (L.wav < hi))
        lams = L.wav.to_numpy()[sel]
        for visit in VISITS_LR:
            for f in sorted(glob.glob(osp.join(STACKS, f'sky_stack_{visit}_sm[1-4].npz'))):
                sm = int(f.split('_sm')[1][0])
                d = np.load(f)
                x, y, var = d['wav'], d['sky'], d['err']**2
                s = best_shift(x, y, lams, sig, var)
                for v in VARIANTS:
                    A, E, B0 = amplitudes(x, y, lams + s, sig, v, var)
                    rows.append(pd.DataFrame(dict(arm = arm, visit = visit, sm = sm, variant = v, idx = sel, wav = lams, A = A[0], E = E[0], B = B0[0], shift_pm = 1e3*s)))
                print(f'{arm} {visit} SM{sm}: shift {1e3*s:+.1f} pm', flush = True)
    R = pd.concat(rows, ignore_index = True)
    R.to_csv(osp.join(OUT, 'obs_lr.csv'), index = False)
    return R

def measure_mr():
    import measure_lsf as ML
    import test_linelist as TL
    t = time.time()
    _, _, corrs = TL.evaluate(MR_LIST, 0.0637, TL.load_all())
    print(f'per-fibre corrections for {len(corrs)} fibres ({time.time() - t:.0f} s)', flush = True)
    L, sig = load_list('m')
    sel = np.flatnonzero((L.wav > MR_RANGE[0] + 1) & (L.wav < MR_RANGE[1] - 1))
    lams = L.wav.to_numpy()[sel]
    rows = []
    for sp in ML.SPECS:
        for f in sorted(glob.glob(f'{ML.DATA}pfsArm-*-m{sp}-sky.fits')):
            visit = int(osp.basename(f).split('-')[1])
            with fits.open(f) as h:
                fid, W = h['FIBERID'].data, h['WAVELENGTH'].data.astype(float)
                F, N = h['FLUX'].data.astype(float), h['NORM'].data.astype(float)
                V, bad = h['COVAR'].data[:, 0, :].astype(float), h['MASK'].data != 0
            per = {v: [] for v in VARIANTS}; perB = []
            for k in range(len(fid)):
                c = corrs.get((sp, int(fid[k])))
                if c is None:
                    continue
                tt = (W[k] - 745)/25
                xc = W[k] - (c[0] + c[1]*tt + c[2]*tt**2) # measured = true + correction
                ok = ~bad[k] & (N[k] > 0)
                y = np.where(ok, F[k]/np.where(N[k] > 0, N[k], 1), np.nan)
                for v in VARIANTS:
                    A, _, B0 = amplitudes(xc, y[None, :], lams, sig, v)
                    per[v].append(A[0])
                    if v == 'base': perB.append(B0[0])
            for v in VARIANTS:
                P = np.array(per[v])
                n = np.sum(np.isfinite(P), axis = 0)
                med = np.nanmedian(P, axis = 0)
                mad = 1.4826*np.nanmedian(np.abs(P - med), axis = 0)
                rows.append(pd.DataFrame(dict(visit = visit, sm = sp, variant = v, idx = sel, wav = lams, A = med, E = 1.2533*mad/np.sqrt(np.maximum(n, 1)), n_fib = n,
                                              B = np.nanmedian(np.array(perB), axis = 0) if v == 'base' else 0.0)))
            print(f'm{sp} {visit}: {len(per["base"])} fibres', flush = True)
    R = pd.concat(rows, ignore_index = True)
    R.to_csv(osp.join(OUT, 'obs_mr.csv'), index = False)
    return R

def bin_spectra(wr, specs, lo, hi, pix):
    ''' average the fine-grid spectra into pixels of width pix on lo-hi '''
    centres = np.arange(lo, hi, pix)
    e = np.searchsorted(wr, np.r_[centres - pix/2, centres[-1] + pix/2])
    S = np.vstack(specs)
    Y = np.add.reduceat(S[:, e[0]:e[-1]], e[:-1] - e[0], axis = 1)/np.diff(e)[None, :]
    return centres, Y

def measure_model():
    for arm in 'brnm':
        L, sig = load_list(arm)
        name = RUNS[arm][0]
        t = time.time()
        wr, specs = pickle.load(open(osp.join(SPEC_DIR, f'sweep_0_{sig}_{arm}_fix4_{DATE}.pkl'), 'rb'))
        if arm == 'm':
            lo, hi, pix = MR_RANGE[0] - 2, MR_RANGE[1] + 2, MR_PIX
            sel = np.flatnonzero((L.wav > MR_RANGE[0] + 1) & (L.wav < MR_RANGE[1] - 1))
        else:
            lo, hi, pix = LR_RANGE[arm][0] - 3, LR_RANGE[arm][1] + 3, LR_PIX
            sel = np.flatnonzero((L.wav > LR_RANGE[arm][0]) & (L.wav < LR_RANGE[arm][1]))
        x, Y = bin_spectra(wr, specs, max(lo, wr[0] + pix), min(hi, wr[-1] - pix), pix)
        del specs
        lams = L.wav.to_numpy()[sel]
        out = {f'A_{v}': amplitudes(x, Y, lams, sig, v)[0].astype(np.float32) for v in VARIANTS}
        np.savez_compressed(osp.join(OUT, f'model_op_{arm}.npz'), idx = sel, wav = lams, **out)
        print(f'model {arm}: {len(sel)} lines x {len(Y)} spectra ({time.time() - t:.0f} s)', flush = True)

if __name__ == '__main__':
    import os
    os.makedirs(OUT, exist_ok = True)
    steps = sys.argv[1:] or ['lr', 'mr', 'model']
    if 'lr' in steps: measure_lr()
    if 'mr' in steps: measure_mr()
    if 'model' in steps: measure_model()
