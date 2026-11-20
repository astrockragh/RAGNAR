'''
The best 5-10 nm windows of each PFS arm in which every feature is the same in all skies up to one scalar per sky, by the model (RAGNAR's
120 skies) and by the data (night 1 -> night 2 for b/r/n, six visits in 9 minutes for m), from pfs_covar/test_windows.py.

The measure, for model and data alike: fit the window as A(sky)*shape(lambda) (the data also get a linear continuum and a small wavelength
shift per exposure); for each feature (one peak with its surroundings, edges at minima of the mean spectrum), the rms of the residual over
its pixels and all skies or exposures, relative to the rms of its own line spectrum; "worst" is the largest in the window (the data value has
the noise taken out). Windows must hold at least MIN_FLUX of the arm's line flux and at most MAX_ABS of their robust-line intensity in lines
with H2O or O2 absorption (the grid has one airmass and two PWVs), and at least MIN_FEAT features that count (>= 2% of the line flux each).

Run from RAGNAR_new/: python pfs_covar/best_windows.py  ->  pfs_covar/out/best_windows_5-10nm.csv
'''
import numpy as np
import pandas as pd
import os.path as osp
from scipy import stats

HERE = osp.dirname(osp.abspath(__file__)); OUT = osp.join(HERE, 'out')
MIN_FLUX, MAX_ABS, NKEEP = 0.01, 0.10, 5
MIN_FEAT = 3 # features with >= 2% of the window's line flux: a window that one line dominates is trivially the same up to a scalar

def pick(T, by):
    kk = []
    for _, w in T.sort_values(by).iterrows(): # best first, no overlaps
        if np.isfinite(w[by]) and all(w.hi < k.lo or w.lo > k.hi for k in kk):
            kk.append(w)
        if len(kk) == NKEEP:
            break
    return kk

if __name__ == '__main__':
    rows, corr = [], {}
    for arm in 'brnm':
        T = pd.read_csv(osp.join(OUT, f'windows_{arm}_5-10nm.csv'))
        use = T[(T.flux_frac >= MIN_FLUX) & (T.absorbed <= MAX_ABS) & (T.n_feat_counted >= MIN_FEAT) & np.isfinite(T.obs_worst)].copy()
        use['both'] = np.maximum(use.E_worst, use.obs_worst) # good in the model and in the data
        rho = stats.spearmanr(use.E_worst, use.obs_worst).statistic if len(use) > 3 else np.nan
        corr[arm] = dict(n_windows = len(T), n_usable = len(use), spearman_model_data = rho)
        for by, name in [('E_worst', 'model'), ('obs_worst', 'data'), ('both', 'both')]:
            for rank, w in enumerate(pick(use, by), 1):
                rows.append(dict(arm = arm, ranked_by = name, rank = rank, **{k: w[k] for k in ['lo', 'hi', 'width', 'n_feat', 'n_lines', 'mostly', 'flux_frac', 'absorbed',
                                 'E_worst', 'E_window', 'obs_worst', 'obs_window', 'noise_worst', 'chi2_dof'] + [c for c in ['n1_worst', 'n2_worst'] if c in w]}))
    R = pd.DataFrame(rows)
    R.to_csv(osp.join(OUT, 'best_windows_5-10nm.csv'), index = False)
    # for the report: every usable window (model and data), the best ones and the model-data correlation
    import json
    fin = lambda v: round(float(v), 5) if np.isfinite(float(v)) else None # JSON has no NaN
    corr = {a: {k: fin(v) for k, v in d.items()} for a, d in corr.items()}
    rep = dict(corr = corr, best = json.loads(R.round(5).to_json(orient = 'records')), windows = {})
    for arm in 'brnm':
        T = pd.read_csv(osp.join(OUT, f'windows_{arm}_5-10nm.csv'))
        use = T[(T.flux_frac >= MIN_FLUX) & (T.absorbed <= MAX_ABS) & (T.n_feat_counted >= MIN_FEAT) & np.isfinite(T.obs_worst)]
        cols = ['lo', 'hi', 'E_worst', 'E_window', 'obs_worst', 'obs_window'] + [c for c in ['n1_worst', 'n2_worst'] if c in use]
        rep['windows'][arm] = dict(columns = cols + ['mostly'], data = [[fin(v) for v in r[cols]] + [r.mostly] for _, r in use.iterrows()])
    json.dump(rep, open(osp.join(OUT, 'windows_data.json'), 'w'), separators = (',', ':'))
    pd.set_option('display.width', 250)
    print(pd.DataFrame(corr).T.round(3))
    print(R.round(4).to_string())
