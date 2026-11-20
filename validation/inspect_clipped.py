'''
Per-line performance of the line lists on the real PFS sky (from compare_linelists.py), and a closer look at the robust lines that the
per-fibre wavelength solutions clip.

1. Real-data bias per line, the analogue of the simulated test in compare_linelists.py: the median over fibres of measured - reference
   wavelength after each fibre's own correction (unclipped), for lines measured in at least half of the fibres.
2. RAGNAR cent=1 robust lines: clip fraction per line (clipped / measured with S/N > 10), and which line properties go with clipping:
   line-specific absorption (MODTRAN relative to its smooth envelope), the rotational energy of the dominant subline (hot population),
   distance to the nearest other list line, blends of several upper levels (not clean), S/N, agreement between spectrographs.
3. Figure of the worst clipped robust lines against the observed sky (moon-down stack).
Usage: python inspect_clipped.py   (after compare_linelists.py for b, r, n)
Writes clipped_lines_report.txt, clipped_robust_lines.csv and figs/clipped_robust_lines_{arm}.png
'''
import sys, os, os.path as osp
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(HERE))
from compare_linelists import ARMS, ragnar_list, LISTS_DIR
from dev.species_intensities import get_pgopher_linelist
from dev.make_spectra import load_transmission, get_transmission_at_lines

out = []
def say(s = ''):
    print(s, flush = True); out.append(s)

A = get_pgopher_linelist()
Ev = A.groupby("v'")['Eupper'].min()
A['Erot'] = A['Eupper'] - A["v'"].map(Ev)
A_pos = A['Position'].to_numpy()

def dominant_subline(sub_wav, sub_I):
    '''pgopher row of the strongest subline (by its fractional intensity in the list)'''
    k = int(np.argmax(sub_I))
    i = np.argmin(np.abs(A_pos - sub_wav[k]))
    return A.iloc[i] if abs(A_pos[i] - sub_wav[k]) < 1e-4 else None

def line_specific_transmission(wav, lo, hi):
    '''MODTRAN (background, PWV 2 mm, ZD 40) at each line, relative to the curve's upper envelope over 1 nm'''
    tr = load_transmission('background', 2.0, 40, basedir = '/tigress/cj1223/', min_wav = lo - 3, max_wav = hi + 3)
    L = pd.DataFrame(dict(Position = wav, Species = 'OH'))
    t = get_transmission_at_lines(L, tr['nm'].to_numpy(), tr['combin'].to_numpy(), Trot = 195)
    nm, c = tr['nm'].to_numpy()[::-1], tr['combin'].to_numpy()[::-1]
    env = maximum_filter1d(c, size = max(3, int(1.0/np.median(np.diff(nm)))))
    return t/np.interp(wav, nm, env)

say('# Line lists on the real PFS sky, line by line')
rows_all = []
for arm in ['b', 'r', 'n']:
    f = osp.join(HERE, f'linelist_comparison_{arm}_lines.csv')
    if not osp.exists(f): continue
    D = pd.read_csv(f)
    nf = D.groupby('list').apply(lambda d: d[['visit', 'sm', 'fiber']].drop_duplicates().shape[0], include_groups = False)
    say(f'\n## {arm} arm: real-data bias per line (median over fibres of the unclipped residual, lines measured in >= 50% of fibres), pm')
    for name, d in D.groupby('list', sort = False):
        g = d.groupby('line').agg(bias = ('resid_all', 'median'), n = ('measured', 'sum'))
        g = g[g.n >= 0.5*nf[name]]
        b = 1e3*g.bias
        say(f'  {name:28s}: lines {len(g):4d}, rms bias {np.sqrt(np.mean(b**2)):6.2f}, median |bias| {np.median(np.abs(b)):5.2f}, 90th pct |bias| {np.percentile(np.abs(b), 90):6.2f}, lines with |bias| > 10 pm: {np.sum(np.abs(b) > 10)}')

    # RAGNAR cent=1 robust: clipping per line, joined with the list's own columns
    name = 'RAGNAR cent=1, robust'
    d = D[D['list'] == name]
    full = pd.read_pickle(osp.join(LISTS_DIR, ARMS[arm]['tag'].format(c = 1) + '.pkl'))
    lo, hi = ARMS[arm]['range']; sigma = ARMS[arm]['sigma']
    full = full[(full.wav > lo) & (full.wav < hi) & full.robust].reset_index(drop = True) # same rows, same order as ragnar_list(arm, 1, True)
    assert np.allclose(full.wav.to_numpy(), ragnar_list(arm, 1, True).wav.to_numpy())
    g = d.groupby('line').agg(n_meas = ('measured', 'sum'), n_used = ('used', 'sum'), bias = ('resid_all', 'median'),
                              scatter = ('resid_all', lambda r: 1.4826*np.nanmedian(np.abs(r - np.nanmedian(r)))), snr = ('snr', 'median'))
    # agreement between spectrographs: spread of the per-spectrograph median residual
    per_sm = d.groupby(['line', 'sm']).resid_all.median().groupby('line').agg(lambda r: np.nanmax(r) - np.nanmin(r))
    g['sm_spread'] = per_sm
    g = g.join(full, how = 'left')
    g['clip_frac'] = 1 - g.n_used/g.n_meas.where(g.n_meas > 0)
    g['t_rel'] = line_specific_transmission(g.wav.to_numpy(), lo, hi)
    dom = [dominant_subline(np.atleast_1d(sw), np.atleast_1d(si)) for sw, si in zip(g.subline_wav, g.subline_I)]
    g['dom_label'] = [None if r is None else f"({int(r[chr(118)+chr(39)])}-{int(r[chr(118)+chr(34)])}){r.Branch}" for r in dom]
    g['dom_Erot'] = [np.nan if r is None else r.Erot for r in dom]
    g['dom_frac'] = [np.max(np.atleast_1d(si)) for si in g.subline_I]
    g['nearest_list_line_sigma'] = g.separation/sigma
    g['arm'] = arm
    rows_all.append(g)

    measured = g[g.n_meas >= 0.5*nf[name]]
    clipped = measured[measured.clip_frac > 0.5]
    say(f'\n## {arm} arm, {name}: {len(measured)} lines measured in >= 50% of fibres, {len(clipped)} clipped in more than half of those')
    def frac(sel): return f'{(sel & (measured.clip_frac > 0.5)).sum()}/{sel.sum()} clipped'
    say('   what goes with clipping (lines clipped in > 50% of fibres / lines in the group):')
    say(f'     line-specific absorption, t/envelope < 0.97: {frac(measured.t_rel < 0.97)} | unabsorbed: {frac(measured.t_rel >= 0.97)}')
    say(f'     dominant subline E_rot > 1000 cm-1 (hot population): {frac(measured.dom_Erot > 1000)} | below: {frac(measured.dom_Erot <= 1000)}')
    say(f'     nearest list line < 3 sigma: {frac(measured.nearest_list_line_sigma < 3)} | >= 3 sigma: {frac(measured.nearest_list_line_sigma >= 3)}')
    say(f'     not clean (several upper levels): {frac(~measured.clean.astype(bool))} | clean: {frac(measured.clean.astype(bool))}')
    say(f'     dominant subline < 60% of the line: {frac(measured.dom_frac < 0.6)} | >= 60%: {frac(measured.dom_frac >= 0.6)}')
    say(f'     median S/N < 30: {frac(measured.snr < 30)} | >= 30: {frac(measured.snr >= 30)}')
    say(f'     spread between spectrographs > 10 pm: {frac(1e3*measured.sm_spread > 10)} | <= 10 pm: {frac(1e3*measured.sm_spread <= 10)}')
    if len(clipped):
        cols = ['wav', 'bias', 'scatter', 'clip_frac', 'snr', 't_rel', 'dom_label', 'dom_Erot', 'dom_frac', 'number_of_lines', 'clean', 'nearest_list_line_sigma', 'sm_spread', 'sigma_wav', 'label']
        C = clipped[cols].copy()
        C['bias'] *= 1e3; C['scatter'] *= 1e3; C['sm_spread'] *= 1e3; C['sigma_wav'] *= 1e3
        C['label'] = C.label.str.slice(0, 60)
        say('   clipped robust lines (bias, scatter, sm_spread, sigma_wav in pm):')
        say(C.sort_values('wav').to_string(index = False, float_format = lambda v: f'{v:.3f}' if abs(v) < 10 else f'{v:.1f}'))

R = pd.concat(rows_all)
R.to_csv(osp.join(HERE, 'clipped_robust_lines.csv'), index = False)
open(osp.join(HERE, 'clipped_lines_report.txt'), 'w').write('\n'.join(out) + '\n')

#############################################
### Figure: worst clipped robust lines ###
#############################################
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
INK, INK2, GRID, BLUE, GREY = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6', '#8f8e8a'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.8})
SHIFT = {'b': -0.015, 'r': -0.024, 'n': -0.035} # observed - RAGNAR wavelength offset per arm, from the forward-model fits (validation/tier2)
os.makedirs(osp.join(HERE, 'figs'), exist_ok = True)
stack = np.load(osp.join(HERE, 'tier2', 'sky_stack_129806.npz'))
for arm in ['b', 'r', 'n']:
    Ra = R[R.arm == arm]
    g = Ra[(Ra.clip_frac > 0.5) & (Ra.n_meas >= 0.5*Ra.n_meas.max())].copy() # well-measured lines only, as in the report
    if not len(g): continue
    g = g.reindex(g.bias.abs().sort_values(ascending = False).index).head(12)
    sigma = ARMS[arm]['sigma']
    n = len(g); ncol = 4; nrow = int(np.ceil(n/ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize = (3.2*ncol, 2.5*nrow + 0.4), squeeze = False)
    for ax, (_, r) in zip(axs.flat, g.iterrows()):
        w0 = r.wav + SHIFT[arm]
        m = np.abs(stack['wav'] - w0) < 6*sigma
        ax.grid(True, color = GRID, linewidth = 0.8); ax.set_axisbelow(True)
        ax.plot(stack['wav'][m] - w0, stack['sky'][m], color = INK, linewidth = 1.5, drawstyle = 'steps-mid', label = 'PFS sky (129806)')
        top = np.nanmax(stack['sky'][m]); base = np.nanmin(stack['sky'][m])
        for sw, si in zip(np.atleast_1d(r.subline_wav), np.atleast_1d(r.subline_I)):
            ax.plot([sw + SHIFT[arm] - w0]*2, [base, base + si*(top - base)], color = GREY, linewidth = 2, solid_capstyle = 'round')
        ax.axvline(0, color = BLUE, linewidth = 2, label = 'list wavelength')
        ax.axvline(r.bias, color = INK2, linewidth = 1, linestyle = (0, (3, 2)), label = 'measured centroid (median)')
        ax.set_title(f'{r.wav:.3f} nm  bias {1e3*r.bias:+.0f} pm', loc = 'left', fontsize = 9, color = INK)
        lab = str(r.dom_label) + (f'  +{int(r.number_of_lines) - 1} more' if r.number_of_lines > 1 else '')
        ax.text(0.02, 0.96, f'{lab}\nclipped {100*r.clip_frac:.0f}%, t/env {r.t_rel:.2f}\nE$_{{rot}}$ {r.dom_Erot:.0f} cm$^{{-1}}$, nearest {r.nearest_list_line_sigma:.1f}$\\sigma$',
                transform = ax.transAxes, va = 'top', fontsize = 7.5, color = INK2)
        ax.set_xlabel('wavelength - list wavelength [nm]', color = INK2)
    for ax in list(axs.flat)[n:]:
        ax.axis('off')
    h, l = axs.flat[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color = GREY, linewidth = 2)); l.append('sublines (height = fraction of the line)')
    fig.suptitle(f'Robust RAGNAR lines (cent=1, _fix3) clipped in most fibres, {arm} arm: largest biases', x = 0.01, y = 0.995, ha = 'left', fontsize = 10.5, color = INK)
    fig.legend(h, l, loc = 'upper left', ncol = 4, frameon = False, fontsize = 8.5, bbox_to_anchor = (0.005, 0.965))
    fig.tight_layout(rect = (0, 0, 1, 0.90))
    fig.savefig(osp.join(HERE, 'figs', f'clipped_robust_lines_{arm}.png'), dpi = 150)
    print('saved', osp.join(HERE, 'figs', f'clipped_robust_lines_{arm}.png'))
