'''
The m-arm sky in 724-767 nm, stacked over all sky fibres after each fibre's own wavelength correction (from test_linelist.py with the given
list), on a fine grid. pfsArm pixels sample the profile at random phases across fibres, so the stack is oversampled. FLUX/NORM removes the
fibre throughputs. Then:
  - peaks of the stacked spectrum (matched filter of the list's sigma, S/N >= 5) that have no list line within 0.5 sigma ("missing" lines),
  - the RAGNAR mean spectrum of the same list's run, scaled to the data, for a direct comparison of the blends.

Run from RAGNAR_new/: python pfs_mr/stack_spectrum.py LIST.csv SIGMA [RAGNAR_SPECTRA.pkl]
'''
import sys, json, pickle
import numpy as np
import pandas as pd
import os.path as osp
from astropy.io import fits
import glob

HERE = osp.dirname(osp.abspath(__file__)); sys.path.insert(0, HERE)
import measure_lsf as ML
import test_linelist as TL

path, sig = sys.argv[1], float(sys.argv[2])
D = TL.load_all()
summ, out, corrs = TL.evaluate(path, sig, D)
grid = np.arange(TL.LO, TL.HI, 0.005)
num, den, var = np.zeros(len(grid)), np.zeros(len(grid)), np.zeros(len(grid))
for sp in ML.SPECS:
    fs = sorted(glob.glob(f'{ML.DATA}pfsArm-*-m{sp}-sky.fits'))
    with fits.open(fs[0]) as h:
        fid, N = h['FIBERID'].data, h['NORM'].data.astype(float)
    for k, f in enumerate(fid):
        key = (sp, int(f))
        if key not in corrs:
            continue
        x, d, v = D[key]
        c = corrs[key]; tt = (x - 745)/25
        xc = x - (c[0] + c[1]*tt + c[2]*tt**2) # measured = true + correction
        n = len(fs)*N[k] # the visits are summed
        ok = np.isfinite(d) & np.isfinite(v) & (n > 0) & (xc > grid[0]) & (xc < grid[-1])
        b = np.clip(np.round((xc[ok] - grid[0])/0.005).astype(int), 0, len(grid) - 1)
        y = d[ok]/n[ok]; vy = v[ok]/n[ok]**2
        np.add.at(num, b, y/vy); np.add.at(den, b, 1/vy)
stack = np.where(den > 0, num/np.maximum(den, 1e-30), np.nan); err = np.where(den > 0, 1/np.sqrt(np.maximum(den, 1e-30)), np.nan)
# fill empty bins and smooth to the pixel scale before peak finding (the stack is oversampled by ~9x)
ok = np.isfinite(stack)
stack_i = np.interp(grid, grid[ok], stack[ok])
from scipy.ndimage import gaussian_filter1d
mf = gaussian_filter1d(stack_i, sig/0.005)
from scipy.signal import find_peaks
pk, _ = find_peaks(mf)
noise = gaussian_filter1d(np.nan_to_num(err, nan = np.nanmedian(err)), sig/0.005)/np.sqrt(sig/0.005*np.sqrt(np.pi)*2)
bgl = pd.Series(stack_i).rolling(int(3/0.005), center = True, min_periods = 1).quantile(0.2).to_numpy()
snr = (mf[pk] - gaussian_filter1d(bgl, sig/0.005)[pk])/noise[pk]
L = pd.read_csv(path)
lw = np.sort(L.wav.to_numpy())
near = np.array([np.min(np.abs(lw - grid[p])) for p in pk])
P = pd.DataFrame(dict(wav = grid[pk], snr = snr, nearest_list_pm = 1e3*near))
P = P[P.snr >= 5]
miss = P[P.nearest_list_pm > 0.5*sig*1e3]
print(f'{len(P)} stacked-data peaks at S/N >= 5 in {TL.LO}-{TL.HI} nm, {len(miss)} without a list line within 0.5 sigma:')
print(miss.round(3).to_string())
rob = L[(L.wav > TL.LO) & (L.wav < TL.HI) & L.robust.astype(bool)]
dm = np.array([np.min(np.abs(P.wav.to_numpy() - w)) if len(P) else np.inf for w in rob.wav])
print(f'robust list lines with a stacked-data peak within 0.5 sigma: {np.mean(dm < 0.5*sig):.2f} of {len(rob)}')
res = dict(grid = grid.round(4).tolist(), stack = np.round(stack_i, 6).tolist(), err = np.round(np.nan_to_num(err), 6).tolist(), peaks = P.round(4).to_dict('records'),
           missing = miss.round(4).to_dict('records'), summary = summ)
if len(sys.argv) > 3:
    wr, specs = pickle.load(open(sys.argv[3], 'rb'))
    mean = np.mean(np.vstack(specs), axis = 0)
    model = np.interp(grid, wr, mean)
    s = np.sum(model*stack_i)/np.sum(model**2) # one scale for the whole window
    res['model'] = np.round(s*model, 6).tolist()
    print('model scale', s)
json.dump(res, open(osp.join(ML.OUT, 'stack_' + osp.basename(path).replace('.csv', '.json')), 'w'))
