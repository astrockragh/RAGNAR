'''
Combine the per-arm RAGNAR line lists into one file.

Each arm's list is simulated with that arm's line spread and wing (run_RAGNAR.py), so the blend structure and centroid of a line depend on
the arm that measures it. The arms overlap (b and r in 600-650 nm, r and n in 950-1000 nm), and in those ranges both arms' entries are kept,
marked with in_overlap. Select by `arm` to get one arm's list back exactly.

Added columns:
    arm             b, r or n
    line_spread     the Gaussian sigma [nm] the arm's list was simulated with
    r_scat          the line wing fraction it was simulated with
    in_overlap      True if the line lies in the wavelength range of another arm's list
    other_arm_dwav  for overlap lines: wavelength of the nearest line in the other arm's list minus this line's [nm], if within 0.5 sigma
    line_id         unique id, arm + wavelength
Usage: python make_unified_list.py [--tag _fix4] [--date 20261007] [--cent 1]
'''
import argparse, glob, os.path as osp
import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
ARMS = {'b': dict(sig = 0.088, r_scat = 0.0125, range = (400, 650)),     # as in run_RAGNAR.py
        'r': dict(sig = 0.112, r_scat = 0.0125/1.8, range = (600, 1000)),
        'n': dict(sig = 0.105, r_scat = 0.0125/1.8, range = (950, 1290))}

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default = '_fix4'); ap.add_argument('--date', default = '20261007'); ap.add_argument('--cent', type = int, default = 1)
a = ap.parse_args()

parts = {}
for arm, p in ARMS.items():
    f = osp.join(HERE, 'lineLists', f"sweep_{p['sig']}_{a.cent}_{arm}{a.tag}_{a.date}.pkl")
    d = pd.read_pickle(f).copy()
    d.insert(0, 'arm', arm)
    d.insert(1, 'line_spread', p['sig']); d.insert(2, 'r_scat', p['r_scat'])
    parts[arm] = d
    print(f'{arm}: {len(d)} lines from {osp.basename(f)}')

for arm, d in parts.items():
    others = [o for o in ARMS if o != arm]
    in_ov = np.zeros(len(d), bool); dwav = np.full(len(d), np.nan)
    for o in others:
        lo, hi = ARMS[o]['range']
        sel = (d.wav > lo) & (d.wav < hi)
        in_ov |= sel.to_numpy()
        ow = np.sort(parts[o].wav.to_numpy())
        for i in np.nonzero(sel.to_numpy())[0]:
            j = np.searchsorted(ow, d.wav.iloc[i]); cand = ow[max(j - 1, 0):j + 1]
            if len(cand):
                k = np.argmin(np.abs(cand - d.wav.iloc[i]))
                if abs(cand[k] - d.wav.iloc[i]) < 0.5*ARMS[arm]['sig']: dwav[i] = cand[k] - d.wav.iloc[i]
    d['in_overlap'] = in_ov; d['other_arm_dwav'] = dwav
    d['line_id'] = [f'{arm}{w:.4f}' for w in d.wav]

U = pd.concat(parts.values()).sort_values(['wav', 'arm']).reset_index(drop = True)
cols = ['line_id', 'arm', 'wav'] + [c for c in U.columns if c not in ('line_id', 'arm', 'wav')]
U = U[cols]
out = osp.join(HERE, 'lineLists', f'sweep_unified_{a.cent}{a.tag}_{a.date}')
U.to_csv(out + '.csv', index = False)
U.to_pickle(out + '.pkl')
ov = U[U.in_overlap]
print(f'unified: {len(U)} lines ({U.robust.sum()} robust), {len(ov)} in arm overlaps, of which {np.isfinite(ov.other_arm_dwav).sum()} have a counterpart in the other arm within 0.5 sigma '
      f'(median |dwav| {1e3*np.nanmedian(np.abs(ov.other_arm_dwav)):.1f} pm)')
print('written', out + '.csv', 'and .pkl')
