'''
Do the lines RAGNAR groups as varying together (dev/covariation.py) vary together in the PFS sky?

RAGNAR side: the cent=0 fix4 runs of b, r, n, m (same grid and seed, so the same skies) grouped together with combine_covariation:
S_ij = scatter of ln(I_i/I_j) over the 360 simulated skies, groups = complete linkage at THRESH. S is also computed from the simulated
spectra measured like the data (S_op, measure.py), to see what the measurement and its baseline do to the groups.

Data side (measure.py amplitudes):
  LR (b, r, n): two nights, two visits each. Per line and spectrograph, the night-to-night change d_i = <ln A>_night2 - <ln A>_night1,
  averaged over the spectrographs. Lines that vary together keep their ratio, so d_i - d_j ~ 0; the observed matrix is |d_i - d_j|.
  The error of d_i comes from the scatter between the spectrographs (same sky, independent detectors), modelled per arm as
  sqrt(a^2 + (b/snr)^2)/sqrt(n_sm), so the systematic floor a is included.
  MR (m): six 60 s visits within 9 minutes. The observed matrix is the scatter over the visits of ln(A_i/A_j), per spectrograph,
  against the scatter expected from the fibre-to-fibre noise.
Tests: per group, chi2 of its members' d_i around their mean; |d_i - d_j| against S_ij for all pairs; for each baseline variant
(the continuum treatment, see measure.py).

Run from RAGNAR_new/: python pfs_covar/analyze.py  ->  pfs_covar/out/covar_*.csv, covar_summary.json, figs/*.png, report_data.json
'''
import sys, json, base64, os
import numpy as np
import pandas as pd
import os.path as osp
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = osp.dirname(osp.abspath(__file__)); ROOT = osp.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from dev.covariation import combine_covariation, ratio_scatter
from measure import RUNS, DATE, VARIANTS, OUT

THRESH = 0.01 # ratio scatter that counts as varying together
SNR_LR, SNR_M = 30, 20 # median amplitude S/N per stack (LR) or per visit and spectrograph (m) for a line to enter the data test
NIGHT = {129688: 1, 129689: 1, 129806: 2, 129807: 2}
ARMS = 'brnm'
FIGS = osp.join(OUT, 'figs'); os.makedirs(FIGS, exist_ok = True)

def leaf_order(S):
    ''' complete-linkage leaf order, so every group (a subtree at THRESH) is a contiguous block '''
    if len(S) < 3:
        return np.arange(len(S))
    return leaves_list(linkage(squareform(S, checks = False), method = 'complete'))

def short_label(lab):
    parts = str(lab).split('|')
    base = parts[0].replace('e(', '(').replace('f(', '(')
    return base + (f' +{len(parts) - 1}' if len(parts) > 1 else '')

###################
### RAGNAR side ###
###################

T, S = combine_covariation([osp.join(ROOT, 'lineLists', f'{RUNS[a][0]}_{DATE}_covar.npz') for a in ARMS], threshold = THRESH, names = list(ARMS))
T['short'] = T.label.map(short_label)
# line-specific absorption: the H2O, O2, CO2 or CH4 bit of transmission_origin ([combin, aercld, molec, H2O, O3, O2, CO2, CH4]); the others are smooth
ORIGIN = {a: pd.read_pickle(osp.join(ROOT, 'lineLists', f'{RUNS[a][0]}_{DATE}.pkl')).transmission_origin.astype(str).to_numpy() for a in ARMS}
T['origin'] = [ORIGIN[a][i] for a, i in zip(T.run, T.line_index)]
T['absorbed'] = T.origin.map(lambda o: o[3] == '1' or '1' in o[5:])
key = list(zip(T.run, T.line_index))
row_of = {k: i for i, k in enumerate(key)}
print(f'{len(T)} robust lines in b/r/n/m, {T.covar_group_all.nunique()} groups, {np.sum(T.covar_n_all > 1)} lines in groups of 2 or more')
G = T.groupby('covar_group_all')
gsum = pd.DataFrame({'n': G.size(), 'arms': G.run.agg(lambda r: ''.join(a for a in ARMS if a in set(r))),
                     'n_upper': G.upper_state.nunique(), 'vgroup': G.vgroup.agg(lambda v: v.value_counts().index[0]),
                     'scatter': G.covar_scatter_all.max(), 'wav_min': G.wav.min(), 'wav_max': G.wav.max()})
print(gsum[gsum.n > 1].arms.value_counts().to_string())

# the simulated spectra measured like the data
op = {a: np.load(osp.join(OUT, f'model_op_{a}.npz')) for a in ARMS}
def op_matrix(rows, variant = 'base'):
    cols = []
    for i in rows:
        a, li = T.run.iat[i], T.line_index.iat[i]
        d = op[a]; k = np.flatnonzero(d['idx'] == li)
        cols.append(d[f'A_{variant}'][:, k[0]].astype(float) if len(k) else np.full(len(d[f'A_{variant}']), np.nan))
    return np.vstack(cols).T

#################
### LR side ###
#################

R = pd.read_csv(osp.join(OUT, 'obs_lr.csv'))
snr_base = R[R.variant == 'base'].assign(snr = lambda r: r.A/r.E).groupby(['arm', 'idx']).snr.median()

def lr_changes(variant):
    r = R[R.variant == variant].copy()
    r['lnA'] = np.log(r.A.where(r.A > 0))
    r['night'] = r.visit.map(NIGHT)
    g = r.groupby(['arm', 'idx', 'sm', 'night']).lnA.mean().unstack('night')
    d = (g[2] - g[1]).rename('d')
    w = r.pivot_table(index = ['arm', 'idx', 'sm'], columns = 'visit', values = 'lnA')
    within = (((w[129689] - w[129688]) + (w[129807] - w[129806]))/2).groupby(['arm', 'idx']).mean()
    per = d.groupby(['arm', 'idx']).agg(['mean', 'std', 'count']).rename(columns = {'mean': 'delta', 'std': 'sd', 'count': 'n_sm'})
    per['snr'] = snr_base.reindex(per.index)
    per['within'] = within.reindex(per.index)
    return per

def error_model(per):
    ''' per arm: var(d per spectrograph) = a^2 + b^2/snr^2, from the median scatter between spectrographs in S/N bins '''
    em = {}
    for arm, p in per.groupby(level = 'arm'):
        q = p[(p.n_sm >= 3) & np.isfinite(p.sd) & (p.snr > 5)]
        edges = np.quantile(q.snr, np.linspace(0, 1, 9))
        x, y = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = q[(q.snr >= lo) & (q.snr <= hi)]
            x.append(np.median(1/s.snr**2)); y.append(np.median(s.sd**2)/stats.chi2(2).median()*2) # median of chi2_2/2 -> mean, for 3 spectrographs
        b2, a2 = np.polyfit(x, y, 1)
        em[arm] = (np.sqrt(max(a2, 1e-6)), np.sqrt(max(b2, 0.0)))
    return em

def group_chi2(sel, delta, err, use = None):
    ''' per group with >= 2 measured lines (among use): chi2 of the members' d around their weighted mean '''
    out = []
    use = (np.ones(len(sel), dtype = bool) if use is None else use) & np.isfinite(delta)
    for g, idx in pd.Series(np.arange(len(sel))[use]).groupby(T.covar_group_all.to_numpy()[sel][use]):
        idx = idx.to_numpy()
        if len(idx) < 2:
            continue
        w = 1/err[idx]**2
        m = np.sum(w*delta[idx])/np.sum(w)
        c2 = float(np.sum(w*(delta[idx] - m)**2))
        out.append(dict(group = int(g), n = len(idx), chi2 = c2, dof = len(idx) - 1, p = float(stats.chi2(len(idx) - 1).sf(c2)),
                        spread = float(np.ptp(delta[idx])), arms = ''.join(a for a in ARMS if a in set(T.run.to_numpy()[sel][idx]))))
    return pd.DataFrame(out)

SBINS = [0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 10]
def pair_bins(Sx, D, N):
    iu = np.triu_indices(len(Sx), 1)
    s, d, n = Sx[iu], D[iu], N[iu]
    rows = []
    for lo, hi in zip(SBINS[:-1], SBINS[1:]):
        m = (s >= lo) & (s < hi) & np.isfinite(d)
        if m.sum():
            rows.append(dict(S_lo = lo, S_hi = hi, n_pairs = int(m.sum()), med_absdiff = float(np.median(d[m])), med_noise = float(np.median(n[m])),
                             frac_3sig = float(np.mean(d[m] > 3*n[m]))))
    return pd.DataFrame(rows)

CMIN = 2.0 # a line's peak over its local baseline, in both nights, for its amplitude to be trusted (night 1 had the Moon up)
rb = R[R.variant == 'base']
contrast_lr = (rb.A/rb.B.where(rb.B > 0)).groupby([rb.arm, rb.idx, rb.visit.map(NIGHT)]).median().unstack().min(axis = 1)

def subset_stats(sel, delta, err, use):
    gc = group_chi2(sel, delta, err, use)
    D = np.abs(delta[:, None] - delta[None, :]); N = np.hypot(err[:, None], err[None, :])
    g = T.covar_group_all.to_numpy()[sel]
    both = np.outer(use, use); iu = np.triu_indices(len(sel), 1)
    same, other = ((g[:, None] == g[None, :]) & both)[iu], ((g[:, None] != g[None, :]) & both)[iu]
    by = gc.groupby('arms').agg(n = ('chi2', 'size'), chi2 = ('chi2', 'sum'), dof = ('dof', 'sum'), ok = ('p', lambda p: np.mean(p > 0.01)), spread = ('spread', 'median'))
    return gc, dict(n_lines = int(use.sum()), n_groups_tested = len(gc), n_lines_in_groups = int(gc.n.sum()) if len(gc) else 0,
                    chi2_dof = float(gc.chi2.sum()/max(1, gc.dof.sum())), median_red_chi2 = float(np.median(gc.chi2/gc.dof)) if len(gc) else np.nan,
                    frac_groups_p01 = float(np.mean(gc.p > 0.01)) if len(gc) else np.nan, med_spread = float(gc.spread.median()) if len(gc) else np.nan,
                    med_absdiff_same = float(np.nanmedian(D[iu][same])), med_noise_same = float(np.nanmedian(N[iu][same])),
                    med_absdiff_other = float(np.nanmedian(D[iu][other])),
                    by_arms = {a: dict(n = int(r.n), chi2_dof = float(r.chi2/r.dof), frac_ok = float(r.ok), spread = float(r.spread)) for a, r in by.iterrows()})

lr_res, lr_keep = {}, None
for v in VARIANTS:
    per = lr_changes(v)
    if v == 'base': # the noise is the same for every variant, so all of them are judged with the errors of the base measurement
        em = error_model(per)
        err_base = pd.Series([np.hypot(em[a][0], em[a][1]/s)/np.sqrt(n) for (a, _), s, n in zip(per.index, per.snr, per.n_sm)], index = per.index)
    per['err'] = err_base.reindex(per.index)
    if lr_keep is None: # the same lines for every variant: chosen on the base measurement
        ok = (per.snr >= SNR_LR) & (per.n_sm >= 3) & np.isfinite(per.delta)
        lr_keep = [row_of[(a, i)] for (a, i) in per.index[ok] if (a, i) in row_of]
    sel = np.array(lr_keep)
    pidx = pd.MultiIndex.from_arrays([T.run.to_numpy()[sel], T.line_index.to_numpy()[sel]])
    if v == 'base':
        contrast = contrast_lr.reindex(pidx).to_numpy()
        SUBSETS = {'all': np.ones(len(sel), dtype = bool), 'contrast': contrast >= CMIN, 'clean': (contrast >= CMIN) & ~T.absorbed.to_numpy()[sel]}
    delta, err = per.delta.reindex(pidx).to_numpy(), per.err.reindex(pidx).to_numpy()
    D = np.abs(delta[:, None] - delta[None, :]); N = np.hypot(err[:, None], err[None, :])
    sub = {k: subset_stats(sel, delta, err, u) for k, u in SUBSETS.items()}
    lr_res[v] = dict(per = per, em = em, delta = delta, err = err, D = D, N = N, gc = sub['contrast'][0], gc_all = sub['all'][0],
                     bins = {k: pair_bins(S[np.ix_(sel[u], sel[u])], D[np.ix_(u, u)], N[np.ix_(u, u)]) for k, u in SUBSETS.items()},
                     within = per.within.reindex(pidx).to_numpy(),
                     stats = {k: st for k, (_, st) in sub.items()} | dict(error_model = {a: [round(x, 5) for x in em[a]] for a in em}))
    for k in SUBSETS:
        st = lr_res[v]['stats'][k]
        print(f"LR {v:7s} {k:8s}: {st['n_lines']:3d} lines, {st['n_groups_tested']:2d} groups, chi2/dof {st['chi2_dof']:6.2f} (median {st['median_red_chi2']:5.2f}), "
              f"p > 0.01: {st['frac_groups_p01']:.2f}, median group spread {100*st['med_spread']:.1f}%, |d_i - d_j| same group {100*st['med_absdiff_same']:.2f}% "
              f"(noise {100*st['med_noise_same']:.2f}%), other {100*st['med_absdiff_other']:.1f}%")
sel_lr = np.array(lr_keep)

# r-n groups: a change of the r/n response between the nights shows up as one offset between the arms, common to all groups
b = lr_res['base']; arm_sel = T.run.to_numpy()[sel_lr]; g_sel = T.covar_group_all.to_numpy()[sel_lr]
diffs = []
for g in np.unique(g_sel[SUBSETS['clean']]): # absorption-free lines, so a change of the water vapour does not enter
    m = (g_sel == g) & SUBSETS['clean']
    if not ({'r', 'n'} <= set(arm_sel[m])):
        continue
    w = 1/b['err']**2
    mr, mn = m & (arm_sel == 'r'), m & (arm_sel == 'n')
    diffs.append((g, np.sum(w[mr]*b['delta'][mr])/np.sum(w[mr]) - np.sum(w[mn]*b['delta'][mn])/np.sum(w[mn]), np.sqrt(1/np.sum(w[mr]) + 1/np.sum(w[mn]))))
dd = pd.DataFrame(diffs, columns = ['group', 'r_minus_n', 'err'])
wd = 1/dd.err**2
rn_off = float(np.sum(wd*dd.r_minus_n)/np.sum(wd)); rn_err = float(1/np.sqrt(np.sum(wd)))
rn_chi = float(np.sum(wd*(dd.r_minus_n - rn_off)**2)/(len(dd) - 1))
delta_c = b['delta'] + np.where(arm_sel == 'n', rn_off, 0.0)
gc_c, st_c = subset_stats(sel_lr, delta_c, b['err'], SUBSETS['clean'])
rn = dict(n_groups = len(dd), offset = rn_off, offset_err = rn_err, chi2_dof_about_offset = rn_chi, scatter = float(dd.r_minus_n.std()), corrected = st_c)
print(f"r-n: {len(dd)} groups, night-to-night change of the r/n response {100*rn_off:+.2f} +- {100*rn_err:.2f}% (chi2/dof about it {rn_chi:.2f}); "
      f"after it, clean subset chi2/dof {st_c['chi2_dof']:.2f}, p > 0.01: {st_c['frac_groups_p01']:.2f}, by arms {st_c['by_arms']}")

#################
### m side ###
#################

M = pd.read_csv(osp.join(OUT, 'obs_mr.csv'))
snr_m = M[M.variant == 'base'].assign(snr = lambda r: r.A/r.E).groupby('idx').snr.median()
keep_m = [row_of[('m', i)] for i in snr_m.index[snr_m >= SNR_M] if ('m', i) in row_of]
sel_m = np.array(keep_m)
m_res = {}
for v in VARIANTS:
    r = M[M.variant == v].copy()
    r['lnA'] = np.log(r.A.where(r.A > 0)); r['e'] = r.E/r.A
    li = T.line_index.to_numpy()[sel_m]
    L = r.pivot_table(index = ['sm', 'visit'], columns = 'idx', values = 'lnA').reindex(columns = li)
    E = r.pivot_table(index = ['sm', 'visit'], columns = 'idx', values = 'e').reindex(columns = li)
    sms = sorted(r.sm.unique())
    dev = np.stack([L.loc[s].to_numpy() - np.nanmean(L.loc[s].to_numpy(), axis = 0) for s in sms]) # (sm, visit, line)
    ee = np.stack([E.loc[s].to_numpy() for s in sms])
    nv = dev.shape[1]
    Dm = dev[:, :, :, None] - dev[:, :, None, :]
    O = np.sqrt(np.nanmean(np.nanvar(Dm, axis = 1, ddof = 1), axis = 0)) # scatter over visits of ln(A_i/A_j), averaged over spectrographs
    Nm = np.sqrt(np.nanmean(ee[:, :, :, None]**2 + ee[:, :, None, :]**2, axis = (0, 1))) # expected from the noise
    np.fill_diagonal(O, 0)
    single = np.sqrt(np.nanmean(np.nanvar(dev, axis = 1, ddof = 1), axis = 0)) # each line's own scatter over the visits
    m_res[v] = dict(O = O, N = Nm, single = single, bins = pair_bins(S[np.ix_(sel_m, sel_m)], O, Nm),
                    stats = dict(n_lines = len(sel_m), med_ratio_scatter = float(np.median(O[np.triu_indices(len(sel_m), 1)])),
                                 med_noise = float(np.median(Nm[np.triu_indices(len(sel_m), 1)])), med_line_scatter = float(np.median(single))))
    st = m_res[v]['stats']
    print(f"m  {v:7s}: {st['n_lines']} lines, median ratio scatter over visits {100*st['med_ratio_scatter']:.2f}% vs noise {100*st['med_noise']:.2f}%, "
          f"single-line scatter {100*st['med_line_scatter']:.2f}%")

#####################################
### the measurement on the model ###
#####################################

def op_effect(sel, variant = 'base'):
    A = op_matrix(sel, variant)
    good = np.all(A > 0, axis = 0)
    Sop = np.full((len(sel), len(sel)), np.nan)
    Sop[np.ix_(good, good)] = ratio_scatter(A[:, good])
    return Sop
Sop_lr, Sop_m = op_effect(sel_lr), op_effect(sel_m)
same_lr = T.covar_group_all.to_numpy()[sel_lr][:, None] == T.covar_group_all.to_numpy()[sel_lr][None, :]
iu = np.triu_indices(len(sel_lr), 1)
op_stats = {}
for k, u in SUBSETS.items():
    sp = Sop_lr[iu][(same_lr & np.outer(u, u))[iu]]; sp = sp[np.isfinite(sp)]
    op_stats[k] = dict(n_pairs_same = len(sp), frac_below_thresh = float(np.mean(sp < THRESH)), frac_below_2thresh = float(np.mean(sp < 2*THRESH)), median = float(np.median(sp)))
    print(f"measured like the data ({k}): same-group pairs median S_op {100*op_stats[k]['median']:.2f}%, {op_stats[k]['frac_below_thresh']:.2f} below {THRESH:.0%}, {op_stats[k]['frac_below_2thresh']:.2f} below {2*THRESH:.0%}")

###############
### tables ###
###############

base = lr_res['base']
T['tested'] = ''
T.loc[T.index[sel_lr], 'tested'] = 'LR'; T.loc[T.index[sel_m], 'tested'] = 'm'
for c in ['night_change', 'night_change_err', 'within_night', 'contrast', 'm_scatter']:
    T[c] = np.nan
T.loc[T.index[sel_lr], 'night_change'] = base['delta']; T.loc[T.index[sel_lr], 'night_change_err'] = base['err']
T.loc[T.index[sel_lr], 'within_night'] = base['within']; T.loc[T.index[sel_lr], 'contrast'] = contrast
T.loc[T.index[sel_m], 'm_scatter'] = m_res['base']['single']
cols = ['run', 'line_index', 'wav', 'intensity', 'short', 'upper_state', 'upper_frac', 'vgroup', 'vgroup_frac', 'transmission', 'origin', 'absorbed',
        'covar_group', 'covar_group_all', 'covar_n_all', 'covar_scatter_all', 'covar_nearest_all', 'tested', 'contrast', 'night_change', 'night_change_err', 'within_night', 'm_scatter']
T[cols].to_csv(osp.join(OUT, 'covar_lines.csv'), index = False)
gsum = gsum.join(base['gc_all'].set_index('group')[['n', 'chi2', 'dof', 'p', 'spread']].rename(columns = {'n': 'n_tested'}))
gsum = gsum.join(base['gc'].set_index('group')[['n', 'chi2', 'p']].rename(columns = {'n': 'n_tested_c', 'chi2': 'chi2_c', 'p': 'p_c'}))
gsum['members'] = G.apply(lambda g: '; '.join(f'{a}:{w:.3f}' for a, w in zip(g.run, g.wav)))
gsum['upper_states'] = G.upper_state.agg(lambda u: ' | '.join(sorted(set(u))))
gsum.to_csv(osp.join(OUT, 'covar_groups.csv'))
summary = dict(threshold = THRESH, snr_lr = SNR_LR, snr_m = SNR_M, contrast_min = CMIN, n_lines = len(T), n_groups = int(T.covar_group_all.nunique()),
               n_multi_groups = int((gsum.n > 1).sum()), n_lines_in_multi = int((T.covar_n_all > 1).sum()),
               lines_by_arm = T.run.value_counts().to_dict(), multi_by_arms = gsum[gsum.n > 1].arms.value_counts().to_dict(),
               mixed_upper = int(((gsum.n > 1) & (gsum.n_upper > 1)).sum()),
               lr = {v: lr_res[v]['stats'] for v in VARIANTS}, rn = rn, m = {v: m_res[v]['stats'] for v in VARIANTS}, operator = op_stats,
               lr_bins = {v: {k: lr_res[v]['bins'][k].to_dict('records') for k in SUBSETS} for v in VARIANTS},
               m_bins = {v: m_res[v]['bins'].to_dict('records') for v in VARIANTS})
json.dump(summary, open(osp.join(OUT, 'covar_summary.json'), 'w'), indent = 1, default = float)

###############
### figures ###
###############

NORM = LogNorm(1e-3, 1)
CMAP = 'magma_r'
def mat(ax, A, title, order, norm = NORM, cmap = CMAP):
    B = A[np.ix_(order, order)].copy()
    np.fill_diagonal(B, norm.vmin)
    im = ax.imshow(np.clip(np.nan_to_num(B, nan = norm.vmax), norm.vmin, norm.vmax), norm = norm, cmap = cmap, interpolation = 'nearest')
    ax.set_title(title, fontsize = 10); ax.set_xticks([]); ax.set_yticks([])
    return im

order_all = leaf_order(S)
fig, ax = plt.subplots(figsize = (9, 8))
im = mat(ax, S, f'RAGNAR: scatter of ln(I_i/I_j) over 360 simulated skies\n{len(T)} robust lines, b/r/n/m, complete-linkage order', order_all)
fig.colorbar(im, ax = ax, label = 'ratio scatter (0.01 = 1%)', shrink = 0.8)
fig.savefig(osp.join(FIGS, 'matrix_ragnar_all.png'), dpi = 130, bbox_inches = 'tight'); plt.close(fig)

for k, fname in [('contrast', 'matrix_lr.png'), ('all', 'matrix_lr_all.png')]:
    u = np.flatnonzero(SUBSETS[k]); su = sel_lr[u]
    o = leaf_order(S[np.ix_(su, su)])
    fig, axs = plt.subplots(1, 4, figsize = (22, 5.6))
    mat(axs[0], S[np.ix_(su, su)], 'RAGNAR S_ij (line centre)', o)
    mat(axs[1], Sop_lr[np.ix_(u, u)], 'RAGNAR S_ij, measured like the data', o)
    im = mat(axs[2], base['D'][np.ix_(u, u)], 'PFS: |d_i - d_j|, ln-ratio change night 1 -> 2', o)
    fig.colorbar(im, ax = axs[:3], label = 'ln-ratio scatter / change', shrink = 0.85)
    im = mat(axs[3], base['D'][np.ix_(u, u)]/base['N'][np.ix_(u, u)], 'PFS: |d_i - d_j| / error', o, norm = LogNorm(0.1, 100), cmap = 'RdBu_r')
    fig.colorbar(im, ax = axs[3], shrink = 0.85)
    fig.suptitle(f'b/r/n, {"lines with peak/baseline >= %g in both nights" % CMIN if k == "contrast" else "all lines"} and S/N >= {SNR_LR}: {len(u)} lines (same order in all panels)', fontsize = 11)
    fig.savefig(osp.join(FIGS, fname), dpi = 130, bbox_inches = 'tight'); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize = (17, 5.6))
o = leaf_order(S[np.ix_(sel_m, sel_m)])
mat(axs[0], S[np.ix_(sel_m, sel_m)], 'RAGNAR S_ij (line centre)', o)
mat(axs[1], m_res['base']['O'], 'PFS m: scatter of ln ratio over 6 visits', o)
im = mat(axs[2], m_res['base']['O']/m_res['base']['N'], 'PFS m: observed / noise-expected scatter', o, norm = LogNorm(0.3, 10), cmap = 'RdBu_r')
fig.colorbar(im, ax = axs[2], shrink = 0.85)
fig.suptitle(f'm: {len(sel_m)} lines with S/N >= {SNR_M} per visit and spectrograph, 724-767 nm', fontsize = 11)
fig.savefig(osp.join(FIGS, 'matrix_m.png'), dpi = 130, bbox_inches = 'tight'); plt.close(fig)

fig, ax = plt.subplots(figsize = (11, 4.5))
sub = T.iloc[sel_lr]
ref = np.median(base['delta'])
for vg in sorted(sub.vgroup.unique()):
    m = (sub.vgroup == vg).to_numpy()
    ax.errorbar(sub.wav[m], 100*(base['delta'][m] - ref), 100*base['err'][m], fmt = 'o', ms = 3, label = vg, lw = 0.6, alpha = 0.35)
    m &= SUBSETS['contrast']
    ax.errorbar(sub.wav[m], 100*(base['delta'][m] - ref), 100*base['err'][m], fmt = 'o', ms = 4, lw = 0.6, color = ax.lines[-1].get_color())
ax.set_ylim(-120, 150)
ax.set_xlabel('wavelength [nm]'); ax.set_ylabel('change of ln A, night 1 -> 2, minus median [%]'); ax.legend(ncol = 6, fontsize = 7)
ax.set_title(f'faint points: peak/baseline < {CMIN:g} in one of the nights', fontsize = 9)
fig.savefig(osp.join(FIGS, 'night_change.png'), dpi = 130, bbox_inches = 'tight'); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize = (12, 4.5), sharey = True)
for ax, k in zip(axs, ['all', 'contrast']):
    for v in VARIANTS:
        bb = lr_res[v]['bins'][k]
        ax.plot(np.sqrt(bb.S_lo.clip(5e-4)*bb.S_hi.clip(upper = 1)), 100*bb.med_absdiff, 'o-', label = f'{v}')
    bb = lr_res['base']['bins'][k]
    ax.plot(np.sqrt(bb.S_lo.clip(5e-4)*bb.S_hi.clip(upper = 1)), 100*bb.med_noise, 'k--', label = 'noise (base)')
    ax.axvline(THRESH, color = 'gray', lw = 0.8)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('RAGNAR ratio scatter S_ij'); ax.set_title(k, fontsize = 10)
axs[0].set_ylabel('median |d_i - d_j| [%]'); axs[0].legend(fontsize = 8, title = 'baseline')
fig.savefig(osp.join(FIGS, 'pairs_vs_S.png'), dpi = 130, bbox_inches = 'tight'); plt.close(fig)
print('figures in', FIGS)

##########################
### data for the report ###
##########################

import zlib
def q8(A, lo = -3.0, hi = 0.0, tri = False):
    """ log10 of a matrix as bytes (0 = 10^lo or below, 254 = 10^hi or above, 255 = missing), upper triangle only if tri, zlib + base64 """
    L = np.log10(np.clip(A, 10**lo, 10**hi))
    B = np.round((L - lo)/(hi - lo)*254).astype(np.uint8)
    B[~np.isfinite(A)] = 255
    if tri:
        B = B[np.triu_indices(len(B), 1)]
    return base64.b64encode(zlib.compress(B.tobytes(), 9)).decode()

def line_rows(sel):
    t = T.iloc[sel]
    return [dict(arm = a, wav = round(w, 4), I = round(i, 4), lab = s, up = u, uf = round(uf, 3), vg = vg, g = int(g), gn = int(n), gs = round(float(gs), 4), ab = int(ab))
            for a, w, i, s, u, uf, vg, g, n, gs, ab in zip(t.run, t.wav, t.intensity, t.short, t.upper_state, t.upper_frac, t.vgroup, t.covar_group_all, t.covar_n_all,
                                                          t.covar_scatter_all, t.absorbed)]

rep = dict(summary = summary, all = dict(lines = line_rows(order_all), S = q8(S[np.ix_(order_all, order_all)], tri = True)))
o = leaf_order(S[np.ix_(sel_lr, sel_lr)]); s_o = sel_lr[o]
fin = lambda x: [None if not np.isfinite(y) else round(float(y), 5) for y in x]
rep['lr'] = dict(lines = line_rows(s_o), S = q8(S[np.ix_(s_o, s_o)], tri = True), Sop = q8(Sop_lr[np.ix_(o, o)], tri = True),
                 D = {v: q8(lr_res[v]['D'][np.ix_(o, o)], tri = True) for v in ['base', 'none']},
                 Z = q8(base['D'][np.ix_(o, o)]/base['N'][np.ix_(o, o)], -1, 2, tri = True),
                 delta = fin(base['delta'][o]), err = fin(base['err'][o]), within = fin(base['within'][o]), contrast = fin(contrast[o]),
                 delta_none = fin(lr_res['none']['delta'][o]))
o = leaf_order(S[np.ix_(sel_m, sel_m)]); s_o = sel_m[o]
rep['m'] = dict(lines = line_rows(s_o), S = q8(S[np.ix_(s_o, s_o)], tri = True), O = q8(m_res['base']['O'][np.ix_(o, o)], tri = True),
                R = q8(m_res['base']['O'][np.ix_(o, o)]/m_res['base']['N'][np.ix_(o, o)], -1, 2, tri = True), single = fin(m_res['base']['single'][o]))
gtab = gsum[gsum.n > 1].reset_index()
rep['groups'] = json.loads(gtab[['covar_group_all', 'n', 'arms', 'n_upper', 'vgroup', 'scatter', 'n_tested', 'chi2', 'dof', 'p', 'spread', 'n_tested_c', 'chi2_c', 'p_c',
                                 'members', 'upper_states']].rename(columns = {'covar_group_all': 'g'}).round(5).to_json(orient = 'records'))
rep['rn'] = dd.round(5).to_dict('records')
json.dump(rep, open(osp.join(OUT, 'report_data.json'), 'w'), separators = (',', ':'), default = float)
print('report data', round(osp.getsize(osp.join(OUT, 'report_data.json'))/1e6, 2), 'MB')
