'''Are the biases of clean, robust Lambda-doublets explained by an e/f intensity ratio different from the model's?
For a doublet with components at l1 < l2 and model red fraction x_model, a measured centroid bias b implies x_obs = x_model + b/(l2 - l1).
Input: clipped_robust_lines.csv from inspect_clipped.py'''
import numpy as np, pandas as pd, re, os.path as osp
HERE = osp.dirname(osp.abspath(__file__))
R = pd.read_csv(osp.join(HERE, 'clipped_robust_lines.csv'))
rows = []
for _, r in R.iterrows():
    if not isinstance(r.subline_wav, str) or not np.isfinite(r.bias) or not bool(r.clean): continue
    sw = np.array([float(x) for x in r.subline_wav.strip('[]').split()]); si = np.array([float(x) for x in r.subline_I.strip('[]').split()])
    labs = r.label.split('|')
    if len(sw) != 2 or len(labs) != 2: continue
    mJ = [re.search(r'\((\d+\.5)\)', l) for l in labs]; mb = re.search(r'\)([opqrs][OPQRS]\d+)', labs[0]); mp = [re.search(r'([ef])\(\d', l) for l in labs]
    if not all(mJ) or not mb or not all(mp): continue
    o = np.argsort(sw); sw, si = sw[o], si[o]; par = [mp[i].group(1) for i in o]
    sep = sw[1] - sw[0]; x_model = si[1]/si.sum()
    rows.append(dict(arm = r.arm, wav = r.wav, band = labs[0][:5], branch = mb.group(1), J = float(mJ[0].group(1)), sep_pm = 1e3*sep, red_parity = par[1],
                     bias_pm = 1e3*r.bias, x_model = x_model, x_obs = x_model + r.bias/sep if sep > 0 else np.nan, clipf = r.clip_frac, snr = r.snr, n_meas = r.n_meas))
D = pd.DataFrame(rows); D = D[D.n_meas > 200]
for arm, g in D.groupby('arm'):
    res = g[g.sep_pm > 50]
    print(f'== {arm}: {len(g)} clean OH Lambda-doublets measured in >200 fibres; with separation > 50 pm: {len(res)}')
    if len(res) > 2:
        print(f'   corr(separation, bias) = {np.corrcoef(res.sep_pm, res.bias_pm)[0, 1]:+.2f}; median bias {res.bias_pm.median():+.1f} pm; implied red-component fraction {res.x_obs.median():.2f} '
              f'(model {res.x_model.median():.2f}), 16-84% {res.x_obs.quantile(.16):.2f}-{res.x_obs.quantile(.84):.2f}')
        for rp, gg in res.groupby('red_parity'):
            print(f'   red component has parity {rp}: n={len(gg)}, median bias {gg.bias_pm.median():+.1f} pm, implied red fraction {gg.x_obs.median():.2f}')
        for br, gg in res.groupby(res.branch.str[1]):
            print(f'   {br} branch: n={len(gg)}, median bias {gg.bias_pm.median():+.1f} pm, implied red fraction {gg.x_obs.median():.2f}')
    small = g[g.sep_pm <= 50]
    print(f'   separation <= 50 pm: n={len(small)}, median |bias| {small.bias_pm.abs().median():.1f} pm, clipped in >50% of fibres: {(small.clipf > 0.5).sum()}')
    if len(res):
        print(res.sort_values('sep_pm', ascending = False).head(12)[['wav', 'band', 'branch', 'J', 'sep_pm', 'red_parity', 'bias_pm', 'x_model', 'x_obs', 'clipf']].round(2).to_string(index = False))
D.to_csv(osp.join(HERE, 'lambda_doublets.csv'), index = False)
