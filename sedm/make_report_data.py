'''
Collect the numbers for the SEDM sky-line report into sedm/out/report_data.json (run after analyze_sedm.py).
'''
import json, numpy as np, pandas as pd, os.path as osp

HERE = osp.dirname(osp.abspath(__file__))
OUT = osp.join(HERE, 'out')
C = 299792.458 # km/s

def r(x, n = 3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)

d = {}

# the R = 100 sky (spectra are the same in the cent = 0 and 1 runs, only the peak finding differs)
z = np.load('/tigress/cj1223/spectra/sedm_R100_0_20261116.npz')
w, S = z['wav_range'], z['specs'].astype(float)*1e3 # [R/nm]
grid = np.arange(340, 1000.01, 0.5)
idx = np.searchsorted(w, grid)
d['spectrum'] = dict(wav = grid.round(1).tolist(), mean = S.mean(0)[idx].round(3).tolist(), lo = S.min(0)[idx].round(3).tolist(), hi = S.max(0)[idx].round(3).tolist(), n_spec = len(S))
blue = (w > 340) & (w < 550)
d['blue_max_R_per_nm'] = r(S[:, blue].max(), 3)

# a) wavelength calibration: centroider list, peaks with >= 2 R/nm
names = {557.9: 'OI 557.7', 589.5: 'Na D + OH 8-2', 630.7: 'OI 630.0 + OH 9-3', 653.9: 'OH 6-1', 687.7: 'OH 7-2 (in the O₂ B band)', 730.1: 'OH 8-3',
         838.2: 'OH 6-2', 885.8: 'OH 7-3', 939.2: 'OH 8-4 (in the H₂O band)'}
p = pd.read_csv(osp.join(OUT, 'peaks_R100_cent1.csv'))
p = p[p.peak_R_per_nm >= 2].reset_index(drop = True)
mis = {}
for R in [75, 150]:
    o = pd.read_pickle(osp.join(osp.dirname(HERE), 'lineLists', f'sedm_R{R}_1_20261116.pkl')).wav.to_numpy()
    mis[R] = [o[np.argmin(np.abs(o - x))] - x for x in p.wav]
drivers = {'Trot': 'T_rot', 'Tvib': 'T_vib', 'r_hot_scale': 'hot population', 'pwv': 'PWV', 'zd': 'airmass', 'lsf_k': 'LSF width ±3%', 'chem': 'chemistry'}
rows = []
for i, q in p.iterrows():
    sh = {k: abs(q[f'shift_{k}_nm']) for k in drivers}
    top = max(sh, key = sh.get)
    rows.append(dict(wav = r(q.wav, 2), name = names.get(round(q.wav, 1), ''), peak = r(q.peak_R_per_nm, 1), sigma_nm = r(q.sigma_wav, 3),
                     sigma_kms = r(q.sigma_wav/q.wav*C, 0), range_nm = r(q.full_range_nm, 2), sigma_over_fwhm = r(q.sigma_wav/q.fwhm_nm, 3),
                     driver = drivers[top] if sh[top] > 0.02 else '–', driver_shift_nm = r(q[f'shift_{top}_nm'], 2),
                     dR75 = r(mis[75][i], 2), dR150 = r(mis[150][i], 2)))
d['wavecal'] = rows

# b) purity against R
s = pd.read_csv(osp.join(OUT, 'purity_scan_summary.csv'))
d['purity_scan'] = {f'{int(t*100)}_{int(I)}': dict(R = g.R.tolist(), n_pairs = g.n_pairs_lever_ge_100nm.tolist(), n_states = g.n_upper_states_with_pair.tolist(),
                                                     n_pure = g.n_pure_transitions.tolist())
                    for (t, I), g in s.groupby(['purity', 'Imin_R'])}
P = pd.read_csv(osp.join(OUT, 'purity_scan_transitions.csv'))
pairs = {}
for R in [700, 1000, 2000]:
    for thr in [0.95, 0.99]:
        g = P[(P.R == R) & (P.purity_min >= thr) & (P.I_R >= 5)]
        out = []
        for up, h in g.groupby('up'):
            if len(h) >= 2 and h.pos.max() - h.pos.min() >= 100:
                h = h.sort_values('pos')
                out.append(dict(up = up, lines = [f"{t} {x:.1f} nm ({I:.0f} R, {pu:.1%})" for t, x, I, pu in zip(h.trans, h.pos, h.I_R, h.purity_min)]))
        pairs[f'{R}_{int(thr*100)}'] = out
d['pure_pairs'] = pairs
E = pd.read_csv(osp.join(OUT, 'example_lines_purity_vs_R.csv'))
d['example'] = {t: dict(pos = r(g.pos.iloc[0], 2), I = r(g.I_R.iloc[0], 1), R = g.R.tolist(), fid = g.purity_fid.round(4).tolist(), min = g.purity_min.round(4).tolist())
                for t, g in E.groupby('trans')}

# b) the R = 100 flux features (no centroider)
f = pd.read_csv(osp.join(OUT, 'flux_features_R100_cent0.csv'))
d['flux_features'] = [dict(wav = r(q.wav, 1), peak = r(q.peak_R_per_nm, 1), vgroup = q.dominant_vgroup, f_v = r(q.f_vgroup_min, 3), up = q.dominant_upper_state,
                           f_up = r(q.f_upper_min, 3), n90 = int(q.n_upper_states_for_90pct)) for _, q in f.iterrows()]
rs = pd.read_csv(osp.join(OUT, 'ratio_stability_R100_cent0.csv')).merge(pd.read_csv(osp.join(OUT, 'ratio_breakdown_R100_cent0.csv')), on = ['wav_1', 'wav_2'])
T = pd.read_csv(osp.join(OUT, 'transmission_R100_cent0.csv'))
def pair(w1, w2, label):
    q = rs[(rs.wav_1.round(1) == w1) & (rs.wav_2.round(1) == w2)].iloc[0]
    tb = [T[T.wav.round(1) == x].lines_over_continuum for x in (w1, w2)]
    return dict(label = label, wav_1 = w1, wav_2 = w2, emission = r(q.half_range_emission, 4), lsf = r(q.half_range_lsf, 4), atmos = r(q.half_range_atmos, 4),
                all = r(q.half_range_all, 4), **{k: r(q[f'd_{k}'], 4) for k in ['Trot', 'Tvib', 'r_hot_scale', 'chem_k', 'lsf_k', 'pwv', 'zd']},
                tbias = [r(tb[0].min() - 1, 3), r(tb[0].max() - 1, 3), r(tb[1].min() - 1, 3), r(tb[1].max() - 1, 3)])
d['pairs'] = [pair(728.9, 945.2, 'OH 8-3 / 8-4'), pair(728.9, 936.9, 'OH 8-3 / 8-4 (blue half)'), pair(686.1, 884.1, 'OH 7-2 / 7-3'),
              pair(655.0, 838.3, 'OH 6-1 / 6-2'), pair(775.5, 996.1, 'OH 9-4 / 9-5')]
best = rs[rs.lever_nm >= 100].sort_values('half_range_emission')
d['n_pairs_lever100'] = int(len(rs[rs.lever_nm >= 100]))
d['n_pairs_lever100_lt3pct'] = int((best.half_range_emission < 0.03).sum())

json.dump(d, open(osp.join(OUT, 'report_data.json'), 'w'), ensure_ascii = False)
print(json.dumps({k: d[k] for k in ['wavecal', 'pure_pairs', 'flux_features', 'pairs', 'n_pairs_lever100', 'n_pairs_lever100_lt3pct', 'blue_max_R_per_nm']}, indent = 1, ensure_ascii = False)[:9000])
