'''
Numbers for the second version of the SEDM sky-line report (Wendelstein, relative flux calibration from mock observations) into
sedm/out/report_data2.json. Run after run_sedm.py (_wst runs), analyze_sedm.py --tag _wst and mock_sedm.py cache figure pwv mocks.
'''
import json, numpy as np, pandas as pd, os.path as osp

HERE = osp.dirname(osp.abspath(__file__))
OUT = osp.join(HERE, 'out')
C = 299792.458
r = lambda x, n = 3: None if x is None or not np.isfinite(x) else round(float(x), n)
h68 = lambda x: (np.nanpercentile(x, 84) - np.nanpercentile(x, 16))/2
d = {}

# the sky figure: RAGNAR means and ranges (Wendelstein-like runs), shares of the groups that vary together, band labels
d['fig'] = json.load(open(osp.join(OUT, 'figure_groups.json')))

# a) wavelength calibration on the Wendelstein-like runs (centroider lists), with the shift for a wrong R
names = {557.9: '[O I] 557.7', 589.5: 'Na D + OH 8-2', 630.7: '[O I] 630.0 + OH 9-3', 653.9: 'OH 6-1', 687.6: 'OH 7-2 (in the O₂ B band)', 730.2: 'OH 8-3',
         838.2: 'OH 6-2', 885.8: 'OH 7-3', 939.1: 'OH 8-4 (in the H₂O band)'}
p = pd.read_csv(osp.join(OUT, 'peaks_R100_cent1_wst.csv'))
p = p[p.peak_R_per_nm >= 2].reset_index(drop = True)
mis = {R: pd.read_pickle(osp.join(osp.dirname(HERE), 'lineLists', f'sedm_R{R}_1_wst_20261116.pkl')).wav.to_numpy() for R in [75, 150]}
drv = {'Trot': 'T_rot', 'Tvib': 'T_vib', 'r_hot_scale': 'hot population', 'pwv': 'PWV', 'zd': 'airmass', 'lsf_k': 'LSF width', 'chem': 'chemistry'}
rows = []
for _, q in p.iterrows():
    sh = {k: abs(q[f'shift_{k}_nm']) for k in drv}
    top = max(sh, key = sh.get)
    rows.append(dict(wav = r(q.wav, 2), name = names.get(round(q.wav, 1), ''), peak = r(q.peak_R_per_nm, 1), sigma_nm = r(q.sigma_wav, 2),
                     range_nm = r(q.full_range_nm, 2), driver = drv[top] if sh[top] > 0.02 else '–', driver_shift = r(q[f'shift_{top}_nm'], 2),
                     dR75 = r(mis[75][np.argmin(np.abs(mis[75] - q.wav))] - q.wav, 2), dR150 = r(mis[150][np.argmin(np.abs(mis[150] - q.wav))] - q.wav, 2)))
d['wavecal'] = rows
S = np.load('/tigress/cj1223/spectra/sedm_R100_0_wst_20261116.npz')
w = S['wav_range']; d['blue_max'] = r(S['specs'][:, (w > 340) & (w < 550)].max()*1e3, 2)

# b) v'-group purity of the R = 100 band features (no centroider)
f = pd.read_csv(osp.join(OUT, 'flux_features_R100_cent0_wst.csv'))
d['features'] = [dict(wav = r(q.wav, 1), peak = r(q.peak_R_per_nm, 1), group = q.dominant_vgroup.replace("OH v'=", 'v′='), f_v = r(q.f_vgroup_min, 3), f_up = r(q.f_upper_min, 3))
                 for _, q in f.iterrows()]

# PWV: ratio change of the same-v' pairs relative to PWV 0, sky lines and a star's continuum
P = pd.read_csv(osp.join(OUT, 'pwv_pairs.csv'))
d['pwv'] = {}
for (R, z), g in P.groupby(['R', 'zd_wst']):
    d['pwv'].setdefault(str(R), {})[f'{z:.0f}'] = {pr: dict(pwv = gg.pwv_wst.round(2).tolist(), sky = gg.sky.round(4).tolist(), star = gg.star.round(4).tolist())
                                                   for pr, gg in g.groupby('pair')}

# mocks
D = pd.read_csv(osp.join(OUT, 'mocks.csv'))
noise = [5000.0, 500.0, 50.0, np.inf]
def summ(col, R, g):
    x = D[(D.R == R) & (D.g == g)][col]
    return [r(100*np.nanmedian(x), 2), r(100*h68(x), 2)]
cases = [('A_', 'Feature fits, telluric known'), ('A_pwv1.5_', 'Feature fits, PWV known to ±1.5 mm'), ('A_fixed_', 'Feature fits, no telluric correction'),
         ('Bk_', 'Forward model, PWV known'), ('Bf_', 'Forward model, PWV fitted'), ('Bh_', 'Forward model, PWV and hot population fitted')]
d['mock_pairs'] = {str(R): {pr: [dict(case = lab, cells = [summ(f'{c}{pr}', R, g) for g in noise]) for c, lab in cases] for pr in ['8-3/8-4', '7-2/7-3', '6-1/6-2', '9-4/9-5']
                            if f'A_{pr}' in D and D[(D.R == R)][f'A_{pr}'].notna().any()} for R in [75, 100, 150]}
lam = [650, 686, 730, 776, 838, 884, 940, 990]
d['mock_S'] = {str(R): {c: {str(g): [r(100*h68(D[(D.R == R) & (D.g == g)][f'{c}_S{l}']), 2) for l in lam] for g in noise} for c in ['Bk', 'Bf', 'Bh']} for R in [75, 100, 150]}
d['mock_S_bias'] = {str(R): {str(g): [r(100*np.nanmedian(D[(D.R == R) & (D.g == g)][f'Bh_S{l}']), 2) for l in lam] for g in noise} for R in [75, 100, 150]}
d['lam_eval'] = lam
d['mock_flux'] = {str(R): {b: [summ(f'Bh_flux_{b}', R, g) for g in noise] for b in ['8-3', '8-4', '7-2', '7-3', '6-2', '9-4']} for R in [75, 100, 150]}
d['mock_pwv'] = {str(R): {c: [r(h68(D[(D.R == R) & (D.g == g)][f'{c}_pwv_wst'] - D[(D.R == R) & (D.g == g)].pwv_wst), 2) for g in noise] for c in ['Bf', 'Bh']} for R in [75, 100, 150]}
d['mock_trot'] = {str(R): [r(h68(D[(D.R == R) & (D.g == g)].Bh_Tr - D[(D.R == R) & (D.g == g)].Tr), 1) for g in noise] for R in [75, 100, 150]}
d['n_mock'] = int(D.mock.nunique())
d['noise'] = ['5000', '500', '50', 'none']

# single upper states (first round, site independent to the level that matters)
d0 = json.load(open(osp.join(OUT, 'report_data.json')))
d['purity_scan'] = d0['purity_scan']; d['example'] = {k: v for k, v in d0['example'].items() if 'pP1' in k}

json.dump(d, open(osp.join(OUT, 'report_data2.json'), 'w'), ensure_ascii = False, separators = (',', ':'))
print(json.dumps({k: d[k] for k in ['wavecal', 'blue_max', 'mock_S', 'mock_flux', 'mock_pwv', 'mock_trot']}, ensure_ascii = False)[:6000])
print(json.dumps(d['mock_pairs']['100'], ensure_ascii = False))
