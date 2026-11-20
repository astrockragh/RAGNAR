'''
Sections of the PFS sky spectrum in which everything changes linearly together, from the cent=0 fix4 runs (dev/covariation.linear_sections).

Within a section, every simulated sky is mean(lambda) + a(sky)*mode(lambda) to within THRESH (rms residual over pixels and skies, relative to
the rms of the mean spectrum). The skies are the 120 of each run at the nominal LSF: Trot, Tvib, hot population, chemistry and PWV vary,
the LSF is the instrument's and is held fixed. Pixels are sigma/2 of each arm, edges at minima of the mean spectrum, so no line is cut.
A continuum that is the same in every sky only shifts the intercept; a changing one (moonlight, twilight) adds a second mode in real data.

Families: sections whose total fluxes also lie on a straight line with each other (linear_scatter <= THRESH over the same skies, across
arms; the four runs simulated the same skies).
Drivers: the sky parameters the section flux correlates with most over the 120 skies.

Run from RAGNAR_new/: python pfs_covar/sections.py  ->  pfs_covar/out/sections_{1,2,5}pct.csv, sections_data.json, figs/sections_*.png
'''
import sys, json, pickle, time
import numpy as np
import pandas as pd
import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = osp.dirname(osp.abspath(__file__)); ROOT = osp.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from dev.covariation import linear_sections, bin_spectra, linear_scatter, covariation_groups, PARAM_NAMES
from measure import RUNS, DATE, SPEC_DIR, OUT

ARM_RANGE = {'b': (380.0, 650.0), 'r': (630.0, 970.0), 'n': (940.0, 1260.0), 'm': (701.0, 898.0)} # the detectors
THRESHOLDS = [0.01, 0.02, 0.05]
MIX = 0.02 # a section whose robust lines have more than this share of their intensity from a second species is flagged as mixed
DRIVERS = ['Trot', 'Tvib', 'r_hot_scale', 'X_OH', 'X_O2b', 'X_O2a', 'X_Ogreen', 'X_Ored', 'X_Na', 'pwv']

def species(vg):
    return 'OH' if vg.startswith('OH') else 'O2' if vg.startswith('O2') else vg

def section_lines(L, lo, hi):
    '''
    robust lines of the cent=0 list inside a section, the upper vibrational level / species with most of their intensity and its share,
    and the share of species other than the main one. RAGNAR draws the species' chemistry from one quantile, so OH, O2, OI, Na move along
    one curve in the simulations; a section that mixes species can look linear only because of that.
    '''
    s = L[(L.wav >= lo) & (L.wav <= hi) & L.robust.astype(bool)]
    if not len(s):
        return 0, '', np.nan, 0.0
    w = s.groupby('vgroup').intensity.sum().sort_values(ascending = False)
    sp = s.groupby(s.vgroup.map(species)).intensity.sum()
    return len(s), w.index[0], float(w.iloc[0]/w.sum()), float(1 - sp.max()/sp.sum())

out, spec_disp, F_all, keys = {th: [] for th in THRESHOLDS}, {}, {th: [] for th in THRESHOLDS}, {th: [] for th in THRESHOLDS}
P_ref = None
for arm in 'brnm':
    t = time.time()
    name, sig = RUNS[arm]
    L = pd.read_pickle(osp.join(ROOT, 'lineLists', f'{name}_{DATE}.pkl'))
    P = np.load(osp.join(ROOT, 'lineLists', f'{name}_{DATE}_covar.npz'))['params']
    nom = np.isclose(P[:, PARAM_NAMES.index('lsf_factor')], 1.0)
    Pn = P[nom]
    if P_ref is None:
        P_ref = Pn
    assert np.allclose(np.delete(Pn, PARAM_NAMES.index('lsf_factor'), 1), np.delete(P_ref, PARAM_NAMES.index('lsf_factor'), 1)), arm
    wr, specs = pickle.load(open(osp.join(SPEC_DIR, f'sweep_0_{sig}_{arm}_fix4_{DATE}.pkl'), 'rb'))
    lo, hi = max(ARM_RANGE[arm][0], wr[0] + sig), min(ARM_RANGE[arm][1], wr[-1] - sig)
    x, Y = bin_spectra(wr, [specs[i] for i in np.flatnonzero(nom)], lo, hi, sig/2)
    del specs
    m = Y.mean(axis = 0)
    spec_disp[arm] = dict(wav = np.round(x, 3).tolist(), mean = np.round(m*1e3, 3).tolist(), sd = np.round(Y.std(axis = 0)*1e3, 3).tolist(), sigma = sig)
    for th in THRESHOLDS:
        T, F = linear_sections(x, Y, threshold = th, max_width = 30.0)
        T.insert(0, 'arm', arm)
        info = [section_lines(L, a, b) for a, b in zip(T.lo, T.hi)]
        T['n_lines'], T['mostly'], T['mostly_frac'], T['other_species'] = zip(*info)
        T['mixed'] = T.other_species > MIX
        Fc = F - F.mean(axis = 0)
        Pc = np.column_stack([Pn[:, PARAM_NAMES.index(d)] for d in DRIVERS]); Pc = Pc - Pc.mean(axis = 0)
        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            R = (Fc.T @ Pc)/np.outer(np.linalg.norm(Fc, axis = 0), np.linalg.norm(Pc, axis = 0))
        R = np.nan_to_num(R)
        top = np.argsort(-np.abs(R), axis = 1)[:, :2]
        T['driver1'] = [f'{DRIVERS[k[0]]} {R[i, k[0]]:+.2f}' for i, k in enumerate(top)]
        T['driver2'] = [f'{DRIVERS[k[1]]} {R[i, k[1]]:+.2f}' for i, k in enumerate(top)]
        out[th].append(T)
        F_all[th].append(F); keys[th] += [(arm, i) for i in range(len(T))]
        lin = T[T.linear]
        print(f'{arm} {th:.0%}: {len(T)} sections, {len(lin)} linear ({np.sum(lin.n_feat > 1)} with 2+ features, {np.sum(lin.mixed)} mixing species) holding {lin.flux_frac.sum():.2f} of the flux, '
              f'median width {np.median(lin.hi - lin.lo):.2f} nm, widest {np.max(lin.hi - lin.lo):.1f} nm ({time.time() - t:.0f} s)', flush = True)

res = {}
for th in THRESHOLDS:
    T = pd.concat(out[th], ignore_index = True)
    F = np.hstack(F_all[th])
    # families: linear sections whose fluxes are on a straight line with each other, across the spectrum and the arms
    lin = np.flatnonzero(T.linear.to_numpy())
    Ls = linear_scatter(F[:, lin])
    fam = covariation_groups(Ls, th)
    T['family'] = -1
    T.loc[lin, 'family'] = fam
    nf = np.bincount(fam)
    T['family_n'] = 0
    T.loc[lin, 'family_n'] = nf[fam]
    T['width'] = T.hi - T.lo
    T.to_csv(osp.join(OUT, f'sections_{round(100*th)}pct.csv'), index = False)
    res[th] = (T, Ls, lin)
    print(f'{th:.0%}: {len(lin)} linear sections in all arms, {np.sum(nf > 1)} families of 2+ sections ({nf[nf > 1].sum()} sections), largest {nf.max()}')

# figure: each arm's mean spectrum with the linear sections at 1% and 2%
fig, axs = plt.subplots(4, 1, figsize = (15, 13))
for ax, arm in zip(axs, 'brnm'):
    d = spec_disp[arm]
    ax.semilogy(d['wav'], np.maximum(d['mean'], 1e-3), color = 'k', lw = 0.5)
    for th, c, yy in [(0.02, '#9ec5f4', 0.5), (0.01, '#2a78d6', 1.0)]:
        T = res[th][0]
        for _, s in T[(T.arm == arm) & T.linear].iterrows():
            ax.axvspan(s.lo, s.hi, ymin = 0, ymax = 0.04*yy, color = c, lw = 0)
    ax.set_xlim(d['wav'][0], d['wav'][-1]); ax.set_ylabel(f'{arm}: mean [R/nm-ish]')
axs[0].set_title('Mean simulated sky (line-only) with the sections that change linearly together: dark 1%, light 2% (bars along the bottom)', fontsize = 10)
fig.savefig(osp.join(OUT, 'figs', 'sections_spectra.png'), dpi = 120, bbox_inches = 'tight'); plt.close(fig)

# figure: section-to-section linear scatter (2%)
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
T, Ls, lin = res[0.02]
o = leaves_list(linkage(squareform(Ls, checks = False), method = 'complete'))
from matplotlib.colors import LogNorm
fig, ax = plt.subplots(figsize = (8, 7))
B = Ls[np.ix_(o, o)].copy(); np.fill_diagonal(B, 1e-3)
im = ax.imshow(np.clip(B, 1e-3, 1), norm = LogNorm(1e-3, 1), cmap = 'magma_r', interpolation = 'nearest')
ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'{len(lin)} sections linear at 2%: how linearly their fluxes move with each other', fontsize = 10)
fig.colorbar(im, ax = ax, label = 'fractional residual of a straight line', shrink = 0.8)
fig.savefig(osp.join(OUT, 'figs', 'sections_matrix.png'), dpi = 120, bbox_inches = 'tight'); plt.close(fig)

# data for the report
import base64, zlib
def q8tri(A, lo = -3.0, hi = 0.0):
    Lg = np.log10(np.clip(A, 10**lo, 10**hi)); Bq = np.round((Lg - lo)/(hi - lo)*254).astype(np.uint8)
    return base64.b64encode(zlib.compress(Bq[np.triu_indices(len(Bq), 1)].tobytes(), 9)).decode()
rep = dict(spec = spec_disp, sections = {}, families = {})
for th in THRESHOLDS:
    T, Ls, lin = res[th]
    cols = ['arm', 'lo', 'hi', 'n_feat', 'n_lines', 'mostly', 'mostly_frac', 'other_species', 'mixed', 'flux_frac', 'E_worst', 'E_affine', 'E_prop', 'mode_frac', 'linear',
            'family', 'family_n', 'driver1', 'driver2']
    rep['sections'][f'{round(100*th)}'] = json.loads(T[cols].round(5).to_json(orient = 'split', index = False))
    o = leaves_list(linkage(squareform(Ls, checks = False), method = 'complete')) if len(lin) > 2 else np.arange(len(lin))
    rep['families'][f'{round(100*th)}'] = dict(order = [int(lin[k]) for k in o], L = q8tri(Ls[np.ix_(o, o)]))
json.dump(rep, open(osp.join(OUT, 'sections_data.json'), 'w'), separators = (',', ':'))
print('sections data', round(osp.getsize(osp.join(OUT, 'sections_data.json'))/1e6, 2), 'MB')
