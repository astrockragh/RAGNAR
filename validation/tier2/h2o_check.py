'''Line-by-line H2O transmission: observed / intrinsic flux of doublets in an H2O band vs MODTRAN at several PWV and wavelength shifts.
The intrinsic strengths come from a fit to the unabsorbed lines only (mask_t), with no water in the model (PWV = 0).
Usage: python h2o_check.py STACK.npz ARM LO HI [vup|band]'''
import numpy as np, pandas as pd, sys
sys.path.insert(0, '../..')
from sky_model import ArmModel, BASE, tapered_profile
from dev.make_spectra import load_transmission, get_transmission_at_lines
stack, arm, lo, hi = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
d = np.load(stack); X = float(d['airmass'])
M = ArmModel(arm, d['wav'], d['sky'], d['err'], X, pwv = 0.0, mode = sys.argv[5] if len(sys.argv) > 5 else 'vup', mask_t = 0.97) # amplitudes from unabsorbed lines only
r = M.fit()
I = M.intensities(r['Trot']); pos = M.L['Position'].to_numpy()
coef = dict(zip(M.components, M.coef[:len(M.components)]))
amp = np.array([coef[c] for c in M.comp])
# doublet groups: lines within 0.3 sigma of each other
order = np.argsort(pos); grp = np.zeros(len(pos), int); g = 0
for a, b in zip(order[:-1], order[1:]):
    grp[b] = g if pos[b] - pos[a] < 0.3*r['sig0'] else (g := g + 1)
grp[order[0]] = 0
df = pd.DataFrame(dict(pos = pos, I = I*amp, grp = grp, idx = np.arange(len(pos))))
G = df.groupby('grp').agg(pos = ('pos', 'mean'), I = ('I', 'sum'), members = ('idx', list))
G = G[(G.pos > lo) & (G.pos < hi)].sort_values('I', ascending = False).head(150)
rows = []
for _, g in G.iterrows():
    p = g.pos + r['shift']; s = np.abs(M.wav - p) < r['sig0']
    if s.sum() < 2: continue
    own = sum(amp[k]*I[k]*tapered_profile((M.wav[s] - pos[k] - r['shift'])[None, :], np.array([r['sig0']]))[0] for k in g.members)
    tot = (M.model - M.continuum)[s]
    if own.sum() < 0.8*tot.sum(): continue
    rows.append(dict(pos = g.pos, members = g.members, t_obs = 1 + (M.sky[s] - M.model[s]).sum()/own.sum()))
R = pd.DataFrame(rows)
print(f'{stack} {arm} {lo}-{hi} nm: {len(R)} isolated strong doublets; observed transmission 10/50/90%: {np.round(R.t_obs.quantile([.1,.5,.9]).values, 3)}')
for pwv in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]:
    tr = load_transmission('background', pwv, 40, basedir = BASE, min_wav = lo - 5, max_wav = hi + 5)
    best = None
    for dshift_pm in [-30, -20, -10, -5, 0, 5, 10, 20, 30]:
        Lx = M.L.iloc[np.concatenate(R.members.to_numpy())].copy(); Lx['Position'] += dshift_pm*1e-3
        t = np.clip(get_transmission_at_lines(Lx, tr['nm'].to_numpy(), tr['combin'].to_numpy(), Trot = 195), 1e-6, 1)**(X/(1/np.cos(np.deg2rad(40))))
        tl = pd.Series(t, index = np.concatenate(R.members.to_numpy()))
        w = pd.Series(I, index = range(len(I)))
        t_mod = np.array([np.sum(w[m]*tl[m])/np.sum(w[m]) for m in R.members])
        rms = np.sqrt(np.mean((R.t_obs - t_mod)**2)); bias = np.mean(R.t_obs - t_mod)
        if best is None or rms < best[1]: best = (dshift_pm, rms, bias, np.corrcoef(R.t_obs, t_mod)[0, 1])
        if dshift_pm == 0: zero = (rms, bias, np.corrcoef(R.t_obs, t_mod)[0, 1])
    print(f'  PWV {pwv:3.0f} mm: no shift rms {zero[0]:.3f}, bias {zero[1]:+.3f}, corr {zero[2]:.3f} | best shift {best[0]:+d} pm: rms {best[1]:.3f}, bias {best[2]:+.3f}, corr {best[3]:.3f}', flush = True)
