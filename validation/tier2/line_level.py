'''
Line-by-line comparison of the RAGNAR line lists with the PFS sky, using the forward-model fits (fit_all.py, per-v' mode).

For every line in the _fix1 centroider lists (cent = 1):
  - observed and model flux within +-1 sigma of the line, above the fitted continuum/offset polynomial -> observed / model ratio
  - identified: a local maximum of the observed spectrum within +-0.5 sigma of the (shifted) line position, with S/N > 5
Then features in observed - model that the lists don't explain, and whether they show up in every spectrograph (sky) or only some
(persistence from earlier arcs or sky exposures).
Writes TIER2_line_summary.txt and line_level.csv
'''
import glob, re, sys, os.path as osp
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

HERE = osp.dirname(osp.abspath(__file__))
LISTS = osp.join(osp.dirname(osp.dirname(HERE)), 'lineLists')
LIST_SETS = {'fix1': {'b': 'sweep_0.091_1_b_fix1_20261005.pkl', 'r': 'sweep_0.108_1_r_fix1_20261005.pkl', 'n': 'sweep_0.122_1_n_fix1_20261005.pkl'},
             'fix2': {'b': 'sweep_0.088_1_b_fix2_20261007.pkl', 'r': 'sweep_0.112_1_r_fix2_20261007.pkl', 'n': 'sweep_0.105_1_n_fix2_20261007.pkl'}}
LIST_TAG = sys.argv[1] if len(sys.argv) > 1 else 'fix1' # usage: python line_level.py [fix1|fix2]
LIST_FILES = LIST_SETS[LIST_TAG]
out = []
def say(s = ''):
    print(s); out.append(s)

rows, feats = [], []
for f in sorted(glob.glob(osp.join(HERE, 'fits', 'fit_*.npz'))):
    m = re.match(r'fit_(\d+)_sm(\d)_([brn])\.npz$', osp.basename(f))
    if not m: continue
    visit, sm, arm = int(m[1]), int(m[2]), m[3]
    d = np.load(f)
    wav, sky, err, model, cont = d['wav'], d['sky'], d['err'], d['model'], d['continuum']
    shift, sig = float(d['shift']), float(d['sig0'])
    obs, mod = sky - cont, model - cont
    peaks, _ = find_peaks(obs)
    L = pd.read_pickle(osp.join(LISTS, LIST_FILES[arm]))
    L = L[(L.wav > wav[0] + 3*sig) & (L.wav < wav[-1] - 3*sig)]
    for i, r in L.iterrows():
        w0 = r.wav + shift
        s = np.abs(wav - w0) < sig
        if s.sum() < 2: continue
        O, M, E = obs[s].sum(), mod[s].sum(), np.sqrt(np.sum(err[s]**2))
        near = peaks[np.abs(wav[peaks] - w0) < 0.5*sig]
        rows.append(dict(visit = visit, sm = sm, arm = arm, wav = r.wav, label = r.label, robust = r.robust, clean = r.clean, detection_prob = r.detection_prob,
                         I_list = r.intensity, obs = O, model = M, snr = O/E, ratio = O/M if M > 0 else np.nan, peak = len(near) > 0,
                         peak_offset_pm = 1e3*(wav[near[np.argmin(np.abs(wav[near] - w0))]] - w0) if len(near) else np.nan))
    # unexplained features: peaks in the residual that are significant and a sizeable fraction of the local line signal.
    # Only on the moon-down night: with the moon up, the solar spectrum in the continuum leaves structure the polynomial can't follow
    if visit not in (129806, 129807): continue
    res = sky - model
    rp, _ = find_peaks(res, distance = 2)
    local = pd.Series(np.abs(mod)).rolling(int(5/np.median(np.diff(wav))), center = True, min_periods = 1).max().to_numpy()
    for p in rp:
        snr = res[p]/err[p]
        if snr > 20 and res[p] > 0.05*local[p]:
            feats.append(dict(visit = visit, sm = sm, arm = arm, wav = wav[p] - shift, resid = res[p], frac_of_local = res[p]/local[p], snr = snr))
D = pd.DataFrame(rows); F = pd.DataFrame(feats)
D.to_csv(osp.join(HERE, f'line_level_{LIST_TAG}.csv'), index = False)
night = {129688: 1, 129689: 1, 129806: 2, 129807: 2}
D['night'] = D.visit.map(night)

say(f'# Tier 2: line by line, _{LIST_TAG} centroider lists against the PFS sky stacks')
say('\n## Identification (local peak within 0.5 sigma and S/N > 5), fraction of stacks per line, then over lines')
D['ident'] = D.peak & (D.snr > 5)
per_line = D.groupby(['arm', 'wav']).agg(ident = ('ident', 'mean'), robust = ('robust', 'first'), clean = ('clean', 'first'), dp = ('detection_prob', 'first'),
                                         I = ('I_list', 'first'), ratio = ('ratio', 'median'), snr = ('snr', 'median'), label = ('label', 'first'))
for arm, g in per_line.groupby(level = 0):
    rob = g[g.robust]; non = g[~g.robust]
    say(f'  {arm}: robust lines identified in >= 90% of stacks: {np.mean(rob.ident >= 0.9):.2f} of {len(rob)}; identified in none: {np.sum(rob.ident == 0)}'
        f' | non-robust: ' + ', '.join(f'detection_prob {lo}-{hi}: {g2.ident.mean():.2f} (n={len(g2)})' for lo, hi in [(0, 0.5), (0.5, 0.9), (0.9, 1.01)] for g2 in [non[(non.dp >= lo) & (non.dp < hi)]] if len(g2)))
    bad = rob[rob.ident < 0.5].sort_values('I', ascending = False).head(8)
    if len(bad):
        say('     brightest robust lines not identified: ' + '; '.join(f'{w[1]:.3f} nm I={r.I:.3g} S/N={r.snr:.1f} {r.label[:40]}' for w, r in bad.iterrows()))
    say(f'     robust lines identified, by list intensity: ' + ', '.join(f'I {lo:.0e}-{hi:.0e}: {rob[(rob.I >= lo) & (rob.I < hi)].ident.mean():.2f} (n={((rob.I >= lo) & (rob.I < hi)).sum()})' for lo, hi in [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e9)]))

say('\n## Intensities: observed / model flux per line (median over stacks), robust lines with S/N > 20, after the fit\'s per-v\' and per-species scaling')
say('   (this tests the line-to-line pattern within a band: rotational distribution, Lambda doubling, blends, transmission; band totals are in TIER2_fit_summary.txt)')
G = per_line[per_line.robust & (per_line.snr > 20)].copy()
G['band'] = G.label.str.extract(r'^\((\d-\d)\)')[0]
G['J'] = G.label.str.extract(r'\((\d+\.5)\)')[0].astype(float)
G['branch'] = G.label.str.extract(r'\)([opqrs][OPQRS]\d+)')[0]
for arm, g in G.groupby(level = 0):
    say(f'  {arm}: {len(g)} lines, median ratio {g.ratio.median():.3f}, 16-84% {g.ratio.quantile(.16):.3f}-{g.ratio.quantile(.84):.3f}; clean lines only: {g[g.clean].ratio.median():.3f} '
        f'({g[g.clean].ratio.quantile(.16):.3f}-{g[g.clean].ratio.quantile(.84):.3f}, n={g.clean.sum()})')
    gj = g.dropna(subset = ['J'])
    if len(gj) > 10:
        say('     by upper J\': ' + ', '.join(f'J {lo}-{hi}: {gj[(gj.J >= lo) & (gj.J < hi)].ratio.median():.3f} (n={((gj.J >= lo) & (gj.J < hi)).sum()})' for lo, hi in [(0, 3), (3, 6), (6, 9), (9, 20)] if ((gj.J >= lo) & (gj.J < hi)).sum()))
    out_l = g[(g.ratio < 0.7) | (g.ratio > 1.4)].sort_values('I', ascending = False).head(10)
    if len(out_l):
        say('     strongest outliers (ratio): ' + '; '.join(f'{w[1]:.3f} {r.ratio:.2f} {r.label[:35]}' for w, r in out_l.iterrows()))
say('\n  Night 1 (moon up) vs night 2 (moon down), median ratio over robust S/N>20 lines: ' + ', '.join(f'{arm}: {g[g.night == 1].ratio.median():.3f} vs {g[g.night == 2].ratio.median():.3f}' for arm, g in D[D.robust & (D.snr > 20)].groupby('arm')))

say('\n## Features in observed - model not explained by the lists (moon-down night only; residual peak S/N > 20 and > 5% of the local line signal)')
if len(F):
    F['wbin'] = (F.wav/0.1).round()*0.1
    for arm, g in F.groupby('arm'):
        n_stacks = D[D.arm == arm].groupby(['visit', 'sm']).ngroups
        agg = g.groupby('wbin').agg(n = ('sm', 'size'), sms = ('sm', lambda s: ''.join(map(str, sorted(set(s))))), visits = ('visit', lambda s: len(set(s))),
                                    frac = ('frac_of_local', 'median'))
        common = agg[agg.n >= 0.75*n_stacks]; rare = agg[agg.n <= 0.25*n_stacks]
        say(f'  {arm}: {len(agg)} distinct features; in >= 75% of the {n_stacks} stacks (sky, missing from or mismodelled in the list): {len(common)}; in <= 25% (likely persistence or noise): {len(rare)}')
        if len(common):
            say('     common: ' + ', '.join(f'{w:.1f} ({r.frac:.2f})' for w, r in common.sort_values('frac', ascending = False).head(15).iterrows()))
        if len(rare):
            say('     rare: ' + ', '.join(f'{w:.1f} (SM{r.sms}, {r.visits} visit{"s" if r.visits > 1 else ""})' for w, r in rare.sort_values('frac', ascending = False).head(15).iterrows()))
open(osp.join(HERE, f'TIER2_line_summary_{LIST_TAG}.txt'), 'w').write('\n'.join(out) + '\n')
