'''
Rotational level populations of each upper vibrational level v' from the PFS sky stacks, fitted with a cold + hot double Boltzmann distribution.

For every stack of one night, the forward model is refitted with one free amplitude per upper rotational level (v', J', F), with Trot, LSF
and wavelength shift fixed at that stack's per-v' fit (fits/) and pixels with line-specific absorption masked. Every line from a level, in any
branch or band in the arm, constrains its amplitude, and blends are shared out by the fit. The level population is then
    n/g = amplitude x (RAGNAR's model n/g for that level),  g = 2(2J'+1) as in RAGNAR
Levels with amplitude S/N > 5 are normalized per stack, arm and v', then averaged over stacks (error = scatter / sqrt(N)).
Per v', all arms are fitted together with a free scale per arm (absorbing the response):
    n/g = S_band [ (1 - r_hot) exp(-E/k T_cold)/Q_cold + r_hot exp(-E/k T_hot)/Q_hot ]
with the partition functions summed over the unique rotational levels of that v', as in RAGNAR's make_OH_intensities.
Usage: python hot_population.py NIGHT   (1 = 129688/129689, moon up; 2 = 129806/129807, moon down)
Writes hot_population_night{N}.json and figs/hot_population_night{N}.png
'''
import sys, glob, json, re, os, os.path as osp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import erf

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(osp.dirname(HERE)))
from sky_model import ArmModel, tapered_profile, R_SCAT, WF, HALF_WIDTH
from dev.species_intensities import get_pgopher_linelist

HC_K = 1.4387769
night = int(sys.argv[1]) if len(sys.argv) > 1 else 2
VISITS = {1: [129688, 129689], 2: [129806, 129807]}[night]

A_full = get_pgopher_linelist()
Ev = A_full.groupby("v'")['Eupper'].min()
LEVELS = {v: np.unique(np.round(g['Eupper'].to_numpy() - Ev[v], 4)) for v, g in A_full.groupby("v'")} # unique rotational levels per v'

def window_fraction(h, sig, r = R_SCAT):
    '''fraction of the tapered RAGNAR profile within +-h'''
    gam = WF*sig; H = HALF_WIDTH*sig
    cdf = lambda x: (0.5*(1 + erf(x/(np.sqrt(2)*sig))) + np.where(x < 0, r*np.exp(x/gam), 2*r - r*np.exp(-x/gam)))/(1 + 2*r)
    edge = (np.exp(-0.5*(H/sig)**2)/(np.sqrt(2*np.pi)*sig) + r/gam*np.exp(-H/gam))/(1 + 2*r)
    return (cdf(h) - cdf(-h) - edge*2*h)/(cdf(H) - cdf(-H) - edge*2*H)

from sky_model import line_table
rows = []
for f in sorted(glob.glob(osp.join(HERE, 'fits', 'fit_*_sm?_[brn].npz'))):
    visit, sm, arm = re.match(r'.*fit_(\d+)_sm(\d)_([brn])\.npz', f).groups(); visit, sm = int(visit), int(sm)
    if visit not in VISITS: continue
    fit = np.load(f)
    st = np.load(osp.join(HERE, f'sky_stack_{visit}_sm{sm}.npz'))
    # one amplitude per upper rotational level, with Trot, LSF and shift fixed at this stack's per-v' fit and the absorbed-line pixels masked
    M = ArmModel(arm, st['wav'], st['sky'], st['err'], float(st['airmass']), pwv = float(fit['pwv']), mask_t = 0.97, mode = 'level')
    Trot = float(fit['Trot'])
    B = M.basis(Trot, float(fit['sig0']), float(fit['sig1']), float(fit['shift']))
    X = np.vstack([B, M.poly.T]).T
    w = M.use/np.sqrt(M.err**2 + (0.01*np.abs(M.sky))**2)
    Xw = X*w[:, None]
    coef, *_ = np.linalg.lstsq(Xw, M.sky*w, rcond = None)
    cov = np.linalg.pinv(Xw.T @ Xw)*np.sum((M.sky*w - Xw @ coef)**2)/max(w.astype(bool).sum() - X.shape[1], 1)
    ecoef = np.sqrt(np.clip(np.diag(cov), 0, None))
    # RAGNAR's model population of each level (no transmission): I/(A g) of its lines at this Trot
    Lt = line_table(Trot).loc[M.L.index]
    popm = (Lt['Intensity']/(Lt['A']*2*(2*Lt["J'"] + 1))).to_numpy()
    for k, c in enumerate(M.components):
        if not c.startswith('L') or coef[k] <= 0 or coef[k] < 5*ecoef[k]: continue
        v, J, N = c[1:].split('_'); v, J, N = int(v), float(J), int(N)
        sel = (M.comp == c) & (M.L['Species'].to_numpy() == 'OH') & (M.L["v'"].to_numpy() == v)
        rows.append(dict(visit = visit, sm = sm, arm = arm, key = c, band = arm, v = v, J = J, N = N,
                         E = float(np.median(M.L['Eupper'].to_numpy()[sel]) - Ev[v]), n = coef[k]*np.median(popm[sel]), snr = coef[k]/ecoef[k]))
D = pd.DataFrame(rows)
D['stk'] = D.visit.astype(str) + '_' + D.sm.astype(str)
# normalize each stack, arm and v' by the summed population of the levels found in every stack, then average over stacks
P = []
for (arm, v), g in D.groupby(['arm', 'v']):
    nst = g.stk.nunique()
    common = g.groupby('key').stk.nunique(); common = common[common == nst].index
    if len(common) < 2: continue
    norm = g[g.key.isin(common)].groupby('stk').n.sum()
    g = g.assign(nn = g.n/g.stk.map(norm))
    for key, gk in g.groupby('key'):
        if len(gk) < max(3, 0.5*nst): continue
        P.append(dict(band = arm, key = key, v = v, E = gk.E.iloc[0], J = gk.J.iloc[0], N = gk.N.iloc[0], n = gk.nn.mean(),
                      err = max(gk.nn.std()/np.sqrt(len(gk)), 0.01*gk.nn.mean()), nstack = len(gk)))
P = pd.DataFrame(P)
print(f'{len(D)} level measurements from {D.stk.nunique()} stacks -> {len(P)} levels (v\', arm)')

def dist(E, v, Tc, r, Th):
    lev = LEVELS[v]
    Qc = np.sum(np.exp(-lev*HC_K/Tc)); Qh = np.sum(np.exp(-lev*HC_K/Th))
    return (1 - r)*np.exp(-E*HC_K/Tc)/Qc + r*np.exp(-E*HC_K/Th)/Qh

from dev.species_intensities import R_HOT as R_HOT_RAGNAR, T_HOT as T_HOT_RAGNAR # the values RAGNAR uses now (adopted 2026-10-07)
results = {}
for v, g in P.groupby('v'):
    g = g[np.isfinite(g.E) & np.isfinite(g.n) & (g.n > 0) & np.isfinite(g.err) & (g.err > 0)]
    bad = P[(P.v == v) & ~P.index.isin(g.index)]
    if len(bad): print(f"v'={v}: dropping {len(bad)} levels with non-finite values: {bad[['key','E','n','err']].to_dict('records')[:3]}")
    bands = sorted(g.band.unique())
    bi = g.band.map({b: i for i, b in enumerate(bands)}).to_numpy()
    E, y, ey = g.E.to_numpy(), np.log(g.n.to_numpy()), g.err.to_numpy()/g.n.to_numpy()
    if len(g) < 6 or E.max() < 1000:
        print(f"v'={v}: only {len(g)} levels up to E = {E.max():.0f} cm-1, not enough to constrain a hot population"); continue
    def resid(p):
        Tc, lr, Th = p[:3]; s = p[3:]
        return (np.log(dist(E, v, Tc, 10**lr, Th)) + s[bi] - y)/ey
    best = None
    for Th0 in [1000, 3000, 6000, 10000]:
        for lr0 in [-2.5, -1.5]:
            sol = least_squares(resid, [195, lr0, Th0] + [0]*len(bands), bounds = ([120, -5, 400] + [-50]*len(bands), [320, np.log10(0.5), 30000] + [50]*len(bands)))
            if best is None or sol.cost < best.cost: best = sol
    J = best.jac; dof = max(len(y) - len(best.x), 1)
    cov = np.linalg.pinv(J.T @ J)*max(2*best.cost/dof, 1)
    Tc, lr, Th = best.x[:3]; e = np.sqrt(np.diag(cov))
    # single Boltzmann for comparison
    sol1 = least_squares(lambda p: (np.log(dist(E, v, p[0], 0, 1000)) + p[1:][bi] - y)/ey, [195] + [0]*len(bands), bounds = ([100] + [-50]*len(bands), [400] + [50]*len(bands)))
    results[int(v)] = dict(T_cold = Tc, T_cold_err = e[0], r_hot = 10**lr, r_hot_err = 10**lr*np.log(10)*e[1], T_hot = Th, T_hot_err = e[2],
                           chi2_dof = 2*best.cost/dof, chi2_dof_single = 2*sol1.cost/max(len(y) - len(sol1.x), 1), n_levels = len(g), bands = bands,
                           E_max = float(E.max()), r_hot_RAGNAR = float(R_HOT_RAGNAR[v]), T_hot_RAGNAR_at_Tvib_1e4 = float(T_HOT_RAGNAR[v]), scales = list(best.x[3:]))
    print(f"v'={v}: {len(g)} levels (arms {', '.join(bands)}), E up to {E.max():.0f} cm-1 | T_cold {Tc:.1f}+-{e[0]:.1f} K, r_hot {10**lr:.4f}+-{10**lr*np.log(10)*e[1]:.4f}, "
          f"T_hot {Th:.0f}+-{e[2]:.0f} K, chi2/dof {2*best.cost/dof:.2f} (single Boltzmann {2*sol1.cost/max(len(y)-len(sol1.x),1):.2f}) | RAGNAR: r_hot {R_HOT_RAGNAR[v]:.4f}, T_hot {T_HOT_RAGNAR[v]} K x Tvib/1e4")
json.dump({str(k): v for k, v in results.items()}, open(osp.join(HERE, f'hot_population_night{night}.json'), 'w'), indent = 1)
P.to_csv(osp.join(HERE, f'level_populations_night{night}.csv'), index = False)

#############
### Plot ###
#############
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
INK, INK2, GRID, BLUE, GREY = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6', '#8f8e8a'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.8})
vs = sorted(results)
ncol = 4; nrow = int(np.ceil(len(vs)/ncol))
fig, axs = plt.subplots(nrow, ncol, figsize = (3.4*ncol, 3.0*nrow + 0.5), squeeze = False)
for ax, v in zip(axs.flat, vs):
    g = P[P.v == v]; r = results[v]
    bands = r['bands']; s = np.array(r['scales'])
    shift_ = np.array([s[bands.index(b)] for b in g.band])
    yv = g.n.to_numpy()*np.exp(-shift_) # remove the per-band scale so all bands sit on the fitted distribution
    ye = g.err.to_numpy()*np.exp(-shift_)
    Eg = np.linspace(0, max(g.E.max()*1.05, 100), 300)
    ax.grid(True, color = GRID, linewidth = 0.8); ax.set_axisbelow(True)
    # RAGNAR's adopted hot population (at the fitted cold temperature and Tvib = 1e4 K), scaled to the data at low energy
    rag = dist(Eg, v, r['T_cold'], r['r_hot_RAGNAR'], r['T_hot_RAGNAR_at_Tvib_1e4'])
    low = g.E.to_numpy() < 600
    k = np.exp(np.median(np.log(yv[low]) - np.log(dist(g.E.to_numpy()[low], v, r['T_cold'], r['r_hot_RAGNAR'], r['T_hot_RAGNAR_at_Tvib_1e4'])))) if low.any() else 1
    ax.plot(Eg, k*rag, color = GREY, linewidth = 1.2, label = 'RAGNAR (adopted)')
    ax.plot(Eg, dist(Eg, v, r['T_cold'], r['r_hot'], r['T_hot']), color = BLUE, linewidth = 2, solid_capstyle = 'round', label = 'double Boltzmann fit')
    ax.errorbar(g.E, yv, yerr = ye, fmt = 'o', ms = 4, color = INK, ecolor = INK, elinewidth = 1, capsize = 0, mec = 'white', mew = 0.8, zorder = 3, label = 'PFS sky')
    ax.set_yscale('log')
    ax.set_title(f"v' = {v}", color = INK, fontsize = 10, loc = 'left')
    ax.text(0.97, 0.95, f"r$_{{hot}}$ = {r['r_hot']:.3f} ± {r['r_hot_err']:.3f}\nT$_{{hot}}$ = {r['T_hot']:.0f} ± {r['T_hot_err']:.0f} K\nT$_{{cold}}$ = {r['T_cold']:.1f} ± {r['T_cold_err']:.1f} K",
            transform = ax.transAxes, ha = 'right', va = 'top', fontsize = 8.5, color = INK, bbox = dict(boxstyle = 'round,pad=0.3', fc = 'white', ec = GRID, lw = 0.8))
    ax.set_xlabel(r"rotational energy E$_{rot}$ [cm$^{-1}$]", color = INK2)
    ax.set_ylabel('level population n/g (scaled)', color = INK2)
for ax in list(axs.flat)[len(vs):]:
    ax.axis('off')
h, l = axs.flat[0].get_legend_handles_labels()
order = [l.index(x) for x in ['PFS sky', 'double Boltzmann fit', 'RAGNAR (adopted)']]
fig.legend([h[i] for i in order], [l[i] for i in order], loc = 'upper right', bbox_to_anchor = (0.99, 0.995), ncol = 3, frameon = False, fontsize = 9)
fig.suptitle(f"OH rotational level populations, PFS all-sky night {night} ({' + '.join(map(str, VISITS))}{', moon down' if night == 2 else ', moon up'})", color = INK, fontsize = 11, x = 0.01, ha = 'left')
fig.tight_layout(rect = (0, 0, 1, 0.93))
os.makedirs(osp.join(HERE, 'figs'), exist_ok = True)
fig.savefig(osp.join(HERE, 'figs', f'hot_population_night{night}.png'), dpi = 150)
print('saved', osp.join(HERE, 'figs', f'hot_population_night{night}.png'))
