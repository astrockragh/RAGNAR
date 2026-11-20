'''
The best 5-10 nm windows of each PFS arm in which everything changes linearly together (dev/covariation.linear_windows), over the 120 skies
of the cent=0 fix4 runs at the nominal LSF, on sigma/2 pixels with edges at minima of the mean spectrum (as sections.py).
Ranked by E_worst (the worst feature's residual from the window's single pattern, relative to its own flux), among windows that
  - hold at least MIN_FLUX of the arm's line flux (the bluest b bands are linear but far too faint to measure), and
  - have at most MAX_ABS of their robust-line intensity in lines with H2O or O2 absorption (transmission_origin bits): the grid has one
    airmass and two PWVs, so it cannot see absorption changing the lines differently (e.g. 7-2 inside the O2 B band).
The five best windows that do not overlap are kept per arm ('usable'), the five best by the whole-window residual E_affine instead
('usable_window': for a test that uses the summed window flux, where faint features matter little), and the best ones without the two
cuts ('any'), with their robust
lines, the main upper v' or species, the share of other species (RAGNAR ties the species' chemistry together, so a mixed window is suspect)
and the sky parameters the window's flux follows.

Run from RAGNAR_new/: python pfs_covar/best_windows.py [min_width max_width]  ->  pfs_covar/out/best_windows_{min}-{max}nm.csv
'''
import sys, pickle
import numpy as np
import pandas as pd
import os.path as osp

HERE = osp.dirname(osp.abspath(__file__)); ROOT = osp.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from dev.covariation import linear_windows, bin_spectra, PARAM_NAMES
from measure import RUNS, DATE, SPEC_DIR, OUT
from sections import ARM_RANGE, DRIVERS, species

W0, W1 = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (5.0, 10.0)
NKEEP = 5
MIN_FLUX, MAX_ABS = 0.01, 0.10
out = []
for arm in 'brnm':
    name, sig = RUNS[arm]
    L = pd.read_pickle(osp.join(ROOT, 'lineLists', f'{name}_{DATE}.pkl'))
    P = np.load(osp.join(ROOT, 'lineLists', f'{name}_{DATE}_covar.npz'))['params']
    nom = np.isclose(P[:, PARAM_NAMES.index('lsf_factor')], 1.0)
    wr, specs = pickle.load(open(osp.join(SPEC_DIR, f'sweep_0_{sig}_{arm}_fix4_{DATE}.pkl'), 'rb'))
    lo, hi = max(ARM_RANGE[arm][0], wr[0] + sig), min(ARM_RANGE[arm][1], wr[-1] - sig)
    x, Y = bin_spectra(wr, [specs[i] for i in np.flatnonzero(nom)], lo, hi, sig/2)
    del specs
    T = linear_windows(x, Y, W0, W1)
    rob = L[L.robust.astype(bool)].sort_values('wav')
    lw, li = rob.wav.to_numpy(), rob.intensity.to_numpy()
    ab = rob.transmission_origin.astype(str).map(lambda o: o[3] == '1' or '1' in o[5:]).to_numpy()
    ci, ca = np.r_[0, np.cumsum(li)], np.r_[0, np.cumsum(li*ab)]
    a, z = np.searchsorted(lw, T.lo.to_numpy()), np.searchsorted(lw, T.hi.to_numpy(), side = 'right')
    T['absorbed'] = np.where(ci[z] > ci[a], (ca[z] - ca[a])/np.maximum(ci[z] - ci[a], 1e-30), 0.0)
    keep = []
    usable = (T.flux_frac >= MIN_FLUX) & (T.absorbed <= MAX_ABS)
    for sel, ok, by in [('usable', usable, 'E_worst'), ('usable_window', usable, 'E_affine'), ('any', np.ones(len(T), dtype = bool), 'E_worst')]:
        kk = []
        for _, w in T[ok].sort_values(by).iterrows(): # best first, no overlaps
            if all(w.hi < k.lo or w.lo > k.hi for k in kk):
                kk.append(w)
            if len(kk) == NKEEP:
                break
        keep += [(sel, w) for w in kk]
    Pn = P[nom]
    Pc = np.column_stack([Pn[:, PARAM_NAMES.index(d)] for d in DRIVERS]); Pc = Pc - Pc.mean(axis = 0)
    rank = {'usable': 0, 'usable_window': 0, 'any': 0}
    for sel, w in keep:
        rank[sel] += 1
        s = L[(L.wav >= w.lo) & (L.wav <= w.hi) & L.robust.astype(bool)].sort_values('intensity', ascending = False)
        vg = s.groupby('vgroup').intensity.sum().sort_values(ascending = False)
        sp = s.groupby(s.vgroup.map(species)).intensity.sum()
        f = Y[:, int(w.i0):int(w.i1)].sum(axis = 1); f = f - f.mean()
        r = (f @ Pc)/(np.linalg.norm(f)*np.linalg.norm(Pc, axis = 0))
        top = np.argsort(-np.abs(r))[:3]
        out.append(dict(arm = arm, set = sel, rank = rank[sel], lo = round(w.lo, 3), hi = round(w.hi, 3), width = round(w.width, 2), n_feat = int(w.n_feat), n_lines = len(s),
                        E_worst = w.E_worst, E_affine = w.E_affine, E_prop = w.E_prop, mode_frac = w.mode_frac, flux_frac = w.flux_frac,
                        absorbed = w.absorbed, mostly = vg.index[0] if len(vg) else '', mostly_frac = float(vg.iloc[0]/vg.sum()) if len(vg) else np.nan,
                        other_species = float(1 - sp.max()/sp.sum()) if len(sp) else 0.0,
                        drivers = ', '.join(f'{DRIVERS[k]} {r[k]:+.2f}' for k in top),
                        brightest = '; '.join(f'{a:.3f} {lab.split("|")[0]}' for a, lab in zip(s.wav[:4], s.label[:4]))))
    print(f'{arm}: {len(T)} windows of {W0:g}-{W1:g} nm, {int(((T.flux_frac >= MIN_FLUX) & (T.absorbed <= MAX_ABS)).sum())} usable', flush = True)
R = pd.DataFrame(out)
R.to_csv(osp.join(OUT, f'best_windows_{W0:g}-{W1:g}nm.csv'), index = False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
print(R.round(4).to_string())
