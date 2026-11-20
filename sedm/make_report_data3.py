'''
Third version of the report data (band-flux calibration only, the O2 B band and 7-2/7-3, the transmission panel):
report_data2.json + o2_study.json + figure_trans.json + feature_mocks.csv -> sedm/out/report_data3.json.
Run after make_report_data2.py and mock_sedm.py o2 transfig fmocks.
'''
import json, numpy as np, pandas as pd, os.path as osp

HERE = osp.dirname(osp.abspath(__file__))
OUT = osp.join(HERE, 'out')
r = lambda x, n = 2: None if x is None or not np.isfinite(x) else round(float(x), n)
h68 = lambda x: (np.nanpercentile(x, 84) - np.nanpercentile(x, 16))/2

d = json.load(open(osp.join(OUT, 'report_data2.json')))
for k in ['mock_pairs', 'mock_S', 'mock_S_bias', 'mock_flux', 'mock_pwv', 'mock_trot']: # full-fit results, not used any more
    d.pop(k, None)
d['trans'] = json.load(open(osp.join(OUT, 'figure_trans.json')))
d['o2'] = json.load(open(osp.join(OUT, 'o2_study.json')))

F = pd.read_csv(osp.join(OUT, 'feature_mocks.csv'))
noise = [5000.0, 500.0, 50.0, np.inf]
cases = [('lin_full', 'Linear background, telluric known'), ('tel_full', 'Telluric-shaped background, telluric known'),
         ('tel_airmass', 'Telluric-shaped, ZD known, PWV assumed'), ('tel_none', 'Telluric-shaped, atmosphere assumed')]
cell = lambda col, R, g: [r(100*np.nanmedian(F[(F.R == R) & (F.g == g)][col]), 1), r(100*h68(F[(F.R == R) & (F.g == g)][col]), 1)]
d['fm_pairs'] = {str(R): {pr: [dict(case = lab, cells = [cell(f'{c}_{pr}', R, g) for g in noise]) for c, lab in cases]
                          for pr in ['8-3/8-4', '7-2/7-3', '6-1/6-2', '9-4/9-5'] if F[F.R == R][f'tel_full_{pr}'].notna().any()} for R in [75, 100, 150]}
d['fm_curve'] = {str(R): {kn: {str(g): dict(hw = [r(100*h68(F[(F.R == R) & (F.g == g)][f'tel_{kn}_c2_S{l}']), 2) for l in d['lam_eval']],
                                            med = [r(100*np.nanmedian(F[(F.R == R) & (F.g == g)][f'tel_{kn}_c2_S{l}']), 2) for l in d['lam_eval']]) for g in noise}
                          for kn in ['full', 'airmass', 'none']} for R in [75, 100, 150]}
json.dump(d, open(osp.join(OUT, 'report_data3.json'), 'w'), ensure_ascii = False, separators = (',', ':'))
print({R: {pr: v[1]['cells'][1] for pr, v in d['fm_pairs'][R].items()} for R in d['fm_pairs']})
print(d['fm_curve']['100']['full']['500.0'])
