'''
Summarize the RAGNAR forward-model fits to the PFS sky stacks (fit_all.py output in fits/).

1. Fitted parameters per stack (Trot, LSF width and slope, wavelength shift, fit quality) and their consistency across spectrographs
2. Vibrational populations: the per-v' amplitudes are observed/model ratios. With RAGNAR's model at Tvib = 1e4 K and f8 = 1.6, f9 = 1.4,
   ln(ratio_v) = const_arm - E_v/k (1/Tvib - 1/1e4) + ln(f_v,true / f_v,model). Fitting v' = 3-7 gives Tvib, and v' = 8, 9 sit off that
   line by ln(f_true/f_model). Each arm gets its own constant, since the response may differ between arms.
3. Species intensities relative to OH, observed / model, after a single scalar
Writes TIER2_fit_summary.txt
'''
import glob, re, sys, os.path as osp
import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(osp.dirname(HERE)))
from dev.species_intensities import get_pgopher_linelist
HC_K = 1.4387769
out = []
def say(s = ''):
    print(s); out.append(s)

rows, amps = [], []
for f in sorted(glob.glob(osp.join(HERE, 'fits', 'fit_*.npz'))):
    m = re.match(r'fit_(\d+)_sm(\d)_([brn])(_band)?(_wing)?\.npz', osp.basename(f))
    visit, sm, arm, band, wing = int(m[1]), int(m[2]), m[3], bool(m[4]), bool(m[5])
    d = np.load(f)
    resid = d['sky'] - d['model']
    lines = d['model'] - d['continuum']
    rows.append(dict(visit = visit, sm = sm, arm = arm, mode = 'band' if band else ('wing' if wing else 'vup'), Trot = float(d['Trot']), sig0 = float(d['sig0']),
                     sig1 = float(d['sig1']), shift_pm = 1e3*float(d['shift']), r_scat = float(d['r_scat']), chi2_pix = float(d['chi2'])/int(d['npix']),
                     rms_resid_over_lines = np.std(resid)/np.std(lines), pwv = float(d['pwv'])))
    for c, a, B in zip(d['components'], d['coef'], d['B']):
        amps.append(dict(visit = visit, sm = sm, arm = arm, mode = rows[-1]['mode'], component = str(c), amp = float(a), flux = float(np.sum(B)*a)))
P = pd.DataFrame(rows); Am = pd.DataFrame(amps)
night = {129688: 1, 129689: 1, 129806: 2, 129807: 2}
P['night'] = P.visit.map(night); Am['night'] = Am.visit.map(night)

say('# Tier 2: RAGNAR forward model fitted to PFS all-sky stacks')
say(f'{P[P["mode"]=="vup"].shape[0]} fits (visit x spectrograph x arm) with per-v\' amplitudes. Pixels dominated by lines with line-specific H2O/O2 absorption are masked,'
    ' so the fits are insensitive to the assumed PWV (nominal 2 mm); the transmission itself is tested separately with h2o_check.py.')

say('\n## 1. Fitted parameters (median over spectrographs, with the spread between spectrographs)')
V = P[P['mode'] == 'vup']
for (arm, visit), g in V.groupby(['arm', 'visit']):
    say(f'  {arm} {visit}: Trot {g.Trot.median():.1f} K (+-{g.Trot.std():.1f}), sigma {g.sig0.median():.4f} nm (+-{g.sig0.std():.4f}), sigma slope {g.sig1.median():+.3f}, '
        f'shift {g.shift_pm.median():+.1f} pm (+-{g.shift_pm.std():.1f}), rms residual / rms line signal {g.rms_resid_over_lines.median():.3f}, SMs {sorted(g.sm)}')
W = P[P['mode'] == 'wing']
if len(W):
    say('  Wing fraction r_scat with it free (RAGNAR default 0.0125): ' + ', '.join(f'{arm}: {g.r_scat.median():.4f} (16-84% {g.r_scat.quantile(.16):.4f}-{g.r_scat.quantile(.84):.4f})' for arm, g in W.groupby('arm')))
    j = V.merge(W, on = ['visit', 'sm', 'arm'], suffixes = ('', '_w'))
    say('  chi2 improvement from freeing the wing: ' + ', '.join(f'{arm}: median {100*(1 - (g.chi2_pix_w/g.chi2_pix)).median():.1f}%' for arm, g in j.groupby('arm')))

say('\n## 2. Vibrational populations (observed / model per upper level v\')')
A = get_pgopher_linelist()
Ev = A.groupby("v'")['Eupper'].min()
OH = Am[(Am['mode'] == 'vup') & Am.component.str.startswith("OH(v'=")].copy()
OH['v'] = OH.component.str.extract(r"v'=(\d)").astype(int)
res = []
for (visit, sm), g in OH.groupby(['visit', 'sm']):
    g = g[g.amp > 0]
    fit_rows = g[(g.v >= 3) & (g.v <= 7)]
    if fit_rows.arm.nunique() == 0 or len(fit_rows) < 3: continue
    # ln(amp) = c_arm + s*E_v, over v' = 3-7, with s = -(1/Tvib - 1/1e4)/k
    arms = sorted(fit_rows.arm.unique())
    X = np.column_stack([Ev[fit_rows.v].to_numpy()] + [(fit_rows.arm == a).astype(float) for a in arms])
    y = np.log(fit_rows.amp.to_numpy())
    coef = np.linalg.lstsq(X, y, rcond = None)[0]
    s = coef[0]; Tvib = 1/(1/1e4 - s/HC_K) if (1/1e4 - s/HC_K) > 0 else np.inf
    c_arm = dict(zip(arms, coef[1:]))
    for _, r in g.iterrows():
        if r.arm in c_arm:
            pred = c_arm[r.arm] + s*Ev[r.v]
            res.append(dict(visit = visit, sm = sm, arm = r.arm, v = r.v, ratio_to_line = np.exp(np.log(r.amp) - pred), Tvib = Tvib))
R = pd.DataFrame(res)
if len(R):
    say(f'  Tvib from v\' = 3-7, per stack: median {R.groupby(["visit","sm"]).Tvib.first().median():.0f} K (16-84% {R.groupby(["visit","sm"]).Tvib.first().quantile(.16):.0f}-{R.groupby(["visit","sm"]).Tvib.first().quantile(.84):.0f} K)')
    for v in sorted(R.v.unique()):
        g = R[R.v == v]
        say(f'  v\'={v}: observed / (Boltzmann line through v\'=3-7) = {g.ratio_to_line.median():.3f} (16-84% {g.ratio_to_line.quantile(.16):.3f}-{g.ratio_to_line.quantile(.84):.3f}; ' +
            ', '.join(f'{arm} {gg.ratio_to_line.median():.3f}' for arm, gg in g.groupby('arm')) + ')' +
            (f'  -> f{v} = {1.6 if v == 8 else 1.4} x {g.ratio_to_line.median():.3f} = {(1.6 if v == 8 else 1.4)*g.ratio_to_line.median():.2f}' if v in (8, 9) else ''))

say('\n## 3. Other species relative to OH (observed / model, normalized to the arm\'s brightest OH level)')
for arm, g in Am[Am['mode'] == 'vup'].groupby('arm'):
    refc = g[g.component.str.startswith('OH')].groupby('component').flux.median().idxmax()
    say(f'  {arm}: reference {refc}')
    ref = g[g.component == refc].set_index(['visit', 'sm']).amp
    for c, gc in g[~g.component.str.startswith('OH')].groupby('component'):
        r = gc.set_index(['visit', 'sm']).amp/ref
        r = r.dropna()
        if len(r):
            say(f'  {arm} {c}: ' + ', '.join(f'night {n} {r[r.index.get_level_values(0).map(night) == n].median():.2f}' for n in [1, 2]) + f' (spread between stacks {r.std():.2f})')

say('\n## 4. Per-band amplitudes within one upper level (band mode), observed / model normalized to the strongest band of that v\' in the arm')
B = Am[Am['mode'] == 'band']
B = B[B.component.str.startswith('OH(')].copy()
if len(B):
    B['v'] = B.component.str.extract(r'OH\((\d)-').astype(int)
    for (arm, v), g in B.groupby(['arm', 'v']):
        med = g.groupby('component').amp.median()
        flux = g.groupby('component').flux.median()
        if len(med) < 2: continue
        refc = flux.idxmax()
        say(f'  {arm} v\'={v}: ' + ', '.join(f'{c} {med[c]/med[refc]:.2f}' for c in med.index))

open(osp.join(HERE, 'TIER2_fit_summary.txt'), 'w').write('\n'.join(out) + '\n')
P.to_csv(osp.join(HERE, 'fit_parameters.csv'), index = False); Am.to_csv(osp.join(HERE, 'fit_amplitudes.csv'), index = False)
