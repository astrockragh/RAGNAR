'''
Test m-arm line lists against the PFS medium-resolution sky spectra (6 visits summed per fibre, sky fibres of m1-m4, 724-767 nm).

The data centroid of a line is measured the way RAGNAR's centroider defines it: the peak of the spectrum convolved with a Gaussian of the
list's sigma, i.e. the maximum over mu of sum_p (d_p - bg) exp(-(x_p - mu)^2/(2 sigma^2)), refined with a parabola, searched near the list
wavelength. The background is the median of the faintest 30% of pixels within +-1.5 nm.

The wavelengths in the data may be off, so per fibre a smooth correction (quadratic in wavelength) is fitted to the robust, bright lines with
3-sigma clipping, in two passes, and the line offsets are measured after it. A list is then judged by
  - the per-line median offset over all fibres (systematic error of the list wavelength, relative to the ensemble),
  - the scatter of those offsets beyond their errors, for robust lines detected at matched-filter S/N >= 10 in most fibres,
  - completeness: the fraction of robust lines found at S/N >= 5.

Run from RAGNAR_new/: python pfs_mr/test_linelist.py LIST.csv[:sigma] [LIST2.csv[:sigma] ...]
'''
import sys, json, itertools
import numpy as np
import pandas as pd
import os.path as osp

HERE = osp.dirname(osp.abspath(__file__)); sys.path.insert(0, HERE)
import measure_lsf as ML

LO, HI = 724.0, 767.0
SNR_FIT, SNR_DET = 10.0, 5.0

def load_all():
    D = {}
    for sp in ML.SPECS:
        fid, W, F, V, nv = ML.load(sp)
        for k in range(len(fid)):
            D[(sp, int(fid[k]))] = (W[k], F[k], V[k])
    return D

def centroid(x, d, v, lam0, sig, search):
    ''' matched-filter peak near lam0: (mu, snr), or None if there is no interior maximum or the data are masked '''
    m = np.abs(x - lam0) < search + 4*sig
    w = np.abs(x - lam0) < 1.5
    if m.sum() < 6 or not np.all(np.isfinite(d[m])) or not np.all(np.isfinite(v[m])):
        return None
    dw = d[w][np.isfinite(d[w])]
    bg = np.median(np.sort(dw)[:max(3, int(0.3*len(dw)))])
    xm, dm, vm = x[m], d[m] - bg, v[m]
    mu = lam0 + np.linspace(-search, search, 241)
    G = np.exp(-0.5*((xm[None, :] - mu[:, None])/sig)**2)
    R = G @ dm
    k = int(np.argmax(R))
    if k == 0 or k == len(mu) - 1:
        return None
    a, b, c = R[k-1], R[k], R[k+1]
    den = a - 2*b + c
    off = 0.5*(a - c)/den if den != 0 else 0.0
    mpk = mu[k] + off*(mu[1] - mu[0])
    g = np.exp(-0.5*((xm - mpk)/sig)**2)
    return mpk, float(g @ dm/np.sqrt(np.sum(vm*g**2)))

def evaluate(path, sig, D, name = None):
    L = pd.read_csv(path)
    L = L[(L.wav > LO) & (L.wav < HI)].reset_index(drop = True)
    lam = L.wav.to_numpy()
    rob = L.robust.to_numpy().astype(bool) if 'robust' in L else np.ones(len(L), bool)
    bright = rob & (L.intensity.to_numpy() >= np.percentile(L.intensity[rob], 40)) # the brighter robust lines anchor the per-fibre correction
    rows, corrs = [], {}
    for key, (x, d, v) in D.items():
        corr = np.zeros(3) # offset(lambda) = c0 + c1 t + c2 t^2, t = (lambda - 745)/25
        t = (lam - 745)/25
        for search, it in [(0.15, 0), (0.7*sig, 1), (0.7*sig, 2)]:
            exp_ = lam + corr[0] + corr[1]*t + corr[2]*t**2
            meas = [centroid(x, d, v, e, sig, search) for e in exp_]
            mu = np.array([np.nan if q is None else q[0] for q in meas]); snr = np.array([0 if q is None else q[1] for q in meas])
            use = bright & (snr >= SNR_FIT) & np.isfinite(mu)
            if use.sum() < 6:
                break
            off = mu - lam
            keep = use.copy()
            for _ in range(3): # 3-sigma clipping
                A = np.vstack([np.ones(keep.sum()), t[keep], t[keep]**2]).T
                c = np.linalg.lstsq(A, off[keep], rcond = None)[0]
                rr = off - (c[0] + c[1]*t + c[2]*t**2)
                s = 1.4826*np.median(np.abs(rr[keep]))
                keep = use & (np.abs(rr) < 3*max(s, 1e-4))
            corr = c
        else:
            corrs[key] = corr
            for i in range(len(L)):
                rows.append(dict(spec = key[0], fiberId = key[1], i = i, snr = snr[i], resid = (mu[i] - lam[i]) - (corr[0] + corr[1]*t[i] + corr[2]*t[i]**2)))
    R = pd.DataFrame(rows)
    det = R[(R.snr >= SNR_DET) & np.isfinite(R.resid)]
    per = det.groupby('i').agg(n = ('resid', 'size'), off = ('resid', 'median'), scat = ('resid', lambda x: 1.4826*np.median(np.abs(x - np.median(x)))), snr = ('snr', 'median'))
    per['err'] = 1.2533*per.scat/np.sqrt(per.n)
    nf = R.groupby(['spec', 'fiberId']).ngroups
    out = L[['wav', 'sigma_wav', 'intensity', 'robust', 'label'] if 'label' in L else ['wav']].join(per, how = 'left')
    out['det_frac'] = out.n.fillna(0)/nf
    good = out[out.robust.astype(bool) & (out.det_frac > 0.5) & (out.snr >= SNR_FIT)]
    w = 1/good.err**2
    extra = np.sqrt(max(0.0, np.average(good.off**2, weights = w) - np.average(good.err**2, weights = w)))
    summ = dict(list = name or osp.basename(path), sigma = sig, n_fibres = nf, n_lines = len(L), n_robust = int(out.robust.sum()),
                robust_detected = float(np.mean(out[out.robust.astype(bool)].det_frac > 0.5)),
                n_good = len(good), rms_off_pm = float(1e3*np.sqrt(np.average(good.off**2, weights = w))),
                median_err_pm = float(1e3*good.err.median()), extra_scatter_pm = float(1e3*extra),
                chi2_dof = float(np.sum((good.off/good.err)**2)/max(1, len(good) - 1)),
                n_outliers_3sig = int(np.sum(np.abs(good.off) > 3*np.hypot(good.err, good.sigma_wav))))
    return summ, out, corrs

if __name__ == '__main__':
    D = load_all()
    res = []
    for arg in sys.argv[1:]:
        path, _, s = arg.partition(':')
        sig = float(s) if s else float(osp.basename(path).split('_')[1])
        summ, out, _ = evaluate(path, sig, D)
        out.to_csv(osp.join(ML.OUT, 'test_' + osp.basename(path).replace('.csv', f'_s{sig}.csv')), index = False)
        res.append(summ)
        print(json.dumps(summ))
    pd.set_option('display.width', 250)
    print(pd.DataFrame(res).round(3).to_string())
