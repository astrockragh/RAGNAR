'''
Diagnostic for measure_lsf.py: the stacked, oversampled m-arm line profile.

For every fibre and target line, the full-window fit (best wing) gives the background and the neighbouring lines; subtracting them leaves the
target alone. Normalised by its fitted flux and placed at its fitted centre, the pixels of all fibres sample the profile at random phases.
The stack is compared with pixel-integrated Gaussians, and single-Gaussian widths are fitted to it over different half windows.

Run from RAGNAR_new/ after measure_lsf.py: python pfs_mr/stack_profile.py
'''
import sys, json, itertools
import numpy as np
import pandas as pd
import os.path as osp
from scipy.special import erf
from scipy.optimize import least_squares

HERE = osp.dirname(osp.abspath(__file__)); sys.path.insert(0, HERE)
import measure_lsf as ML

ML.MODEL = ML.Model()
summ = json.load(open(osp.join(ML.OUT, 'lsf_summary.json')))
r, wf = summ['wing']['r'], summ['wing']['wf']
rows = []
for sp in ML.SPECS:
    fid, W, F, V, nv = ML.load(sp)
    for k, t in itertools.product(range(len(fid)), ML.TARGETS):
        cm = ML.MODEL.comp[t]
        x, d, v = W[k], F[k], V[k]
        m = np.abs(x - cm['mu0']) < ML.FULL
        if m.sum() < 8 or not np.all(np.isfinite(d[m])) or not np.all(np.isfinite(v[m])) or np.any(v[m] <= 0):
            continue
        f = ML.fit_one(x, ML.edges(x), d, v, cm, r, wf, ML.FULL)
        if f is None or f['e_sigma'] > 0.01:
            continue
        # rebuild the fitted design to separate the target from the rest
        e = ML.edges(x); em = e[np.r_[m, False] | np.r_[False, m]]; xm = x[m]
        P = ML.pix_profile(em, cm['pos'] + f['shift'], f['sigma'], r, wf)*cm['w'][None, :]
        Gm = np.zeros((m.sum(), cm['ncol'])); np.add.at(Gm.T, cm['col'], P.T)
        X = np.column_stack([Gm, np.ones(m.sum()), xm - cm['mu0']]); wt = 1/np.sqrt(v[m])
        c = np.linalg.lstsq(X*wt[:, None], d[m]*wt, rcond = None)[0]
        other = X[:, 1:] @ c[1:]
        tgt = (d[m] - other)/c[0]/np.diff(em) # flux density per unit target flux [1/nm]
        mu = cm['mu0'] + f['shift']
        for xi, yi, vi, wi in zip(xm, tgt, v[m]/(c[0]*np.diff(em))**2, np.diff(em)):
            rows.append((sp, t, xi - mu, yi, vi, wi))
S = pd.DataFrame(rows, columns = ['spec', 'line', 'dx', 'y', 'var', 'dlam'])
S.to_csv(osp.join(ML.OUT, 'stack_points.csv'), index = False)
dl = S.dlam.median()
bins = np.arange(-0.9, 0.9001, 0.01)
S['b'] = np.digitize(S.dx, bins)
st = S.groupby('b').apply(lambda g: pd.Series(dict(dx = g.dx.mean(), y = np.average(g.y, weights = 1/g['var']), e = 1/np.sqrt(np.sum(1/g['var'])), n = len(g)))).reset_index(drop = True)
st.to_csv(osp.join(ML.OUT, 'stack_profile.csv'), index = False)
def gpix(dx, s): # a pixel-integrated Gaussian of intrinsic width s, at random pixel phase it is sampled as itself at the pixel centre offset
    return (erf((dx + dl/2)/(np.sqrt(2)*s)) - erf((dx - dl/2)/(np.sqrt(2)*s)))/(2*dl)
res = {}
for half in [0.1, 0.15, 0.2, 0.3, 0.5, 0.9]:
    q = S[np.abs(S.dx) < half]
    fr = least_squares(lambda p: (p[0]*gpix(q.dx.to_numpy(), p[1]) + p[2] - q.y.to_numpy())/np.sqrt(q['var'].to_numpy()), [1, 0.06, 0], bounds = ([0.5, 0.03, -1], [1.5, 0.1, 1]))
    res[half] = dict(sigma = fr.x[1], sigma_pix = float(np.sqrt(fr.x[1]**2 + dl**2/12)), amp = fr.x[0], bg = fr.x[2])
# FWHM and moments of the stacked profile
yy = st.y.to_numpy(); xx = st.dx.to_numpy(); pk = yy.max()
above = xx[yy > pk/2]
print('stacked profile: FWHM ~', round(above.max() - above.min(), 4), 'nm (Gaussian incl. pixel would be 2.3548*sigma_pix)')
print('pixel width', round(dl, 4))
for h, v in res.items():
    print(f'single Gaussian fit to the stack within +-{h} nm: sigma_int {v["sigma"]:.4f}, incl. pixel {v["sigma_pix"]:.4f}, amp {v["amp"]:.3f}, bg {v["bg"]:.4f}')
left = st[st.dx < 0]; right = st[st.dx > 0]
for lo, hi in [(0.05, 0.1), (0.1, 0.15), (0.15, 0.25), (0.25, 0.5)]:
    l = left[(-left.dx > lo) & (-left.dx < hi)].y.mean(); rr = right[(right.dx > lo) & (right.dx < hi)].y.mean()
    print(f'|dx| {lo}-{hi} nm: left {l:.4f} right {rr:.4f} (asymmetry {(rr - l)/(rr + l):+.3f})')
json.dump({str(k): v for k, v in res.items()}, open(osp.join(ML.OUT, 'stack_fits.json'), 'w'), indent = 1)
