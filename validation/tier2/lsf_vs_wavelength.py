'''
Does the line spread change with wavelength within an arm?

Per stack (visit x spectrograph, per-v' forward-model fits in fits/): lines or tight blends (components within 0.5 sigma) that carry >= 85% of
the model flux in their +-3 sigma window, with no line-specific H2O/O2 absorption, are refitted on their own after subtracting all other lines
and the continuum with the forward-model fit:
    A * sum_i w_i P(x - p_i - d; sigma) + c
with the blend's internal structure (w_i, p_i) fixed at the model's, and A, d, sigma, c free (P = RAGNAR's profile, wing fixed at r_scat = 0.0125,
as in the fits). Then per arm: sigma(lambda) = s0_sm + s1 (lambda - centre) [+ s2 (lambda - centre)^2], with one offset per spectrograph,
fitted to the per-line medians over the stacks of each spectrograph; slope errors by bootstrapping the stacks.
Compared with a constant resolving power (sigma proportional to lambda).
Note: pfsMerged is resampled onto a 0.08 nm grid, which adds a roughly constant width in quadrature, so relative trends here are slightly
diluted compared with pfsArm.
Writes lsf_vs_wavelength.csv, lsf_vs_wavelength.txt and figs/lsf_vs_wavelength.png
'''
import sys, glob, re, os, os.path as osp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(osp.dirname(HERE)))
from sky_model import ArmModel, tapered_profile

import argparse
ap = argparse.ArgumentParser(); ap.add_argument('--arms', default = 'brn'); ap.add_argument('--own', type = float, default = 0.85); ap.add_argument('--snr', type = float, default = 30)
ap.add_argument('--tag', default = '')
ARGS = ap.parse_args()
OWN_MIN = ARGS.own # usage: python lsf_vs_wavelength.py [--arms b --own 0.75 --snr 20 --tag _b_relaxed]
out = []
def say(s = ''):
    print(s, flush = True); out.append(s)

def prof(x, pos, sig):
    return tapered_profile((x - pos)[None, :], np.array([sig]))[0]

rows = []
for f in sorted(glob.glob(osp.join(HERE, 'fits', 'fit_*_sm?_[brn].npz'))):
    visit, sm, arm = re.match(r'.*fit_(\d+)_sm(\d)_([brn])\.npz', f).groups(); visit, sm = int(visit), int(sm)
    if arm not in ARGS.arms: continue
    fit = np.load(f); st = np.load(osp.join(HERE, f'sky_stack_{visit}_sm{sm}.npz'))
    M = ArmModel(arm, st['wav'], st['sky'], st['err'], float(st['airmass']), pwv = float(fit['pwv']), mask_t = 0.97)
    assert np.allclose(M.wav, fit['wav'])
    cont = fit['continuum']
    Trot, sig0, sig1, shift = float(fit['Trot']), float(fit['sig0']), float(fit['sig1']), float(fit['shift'])
    amp = dict(zip(fit['components'], fit['coef'])); a = np.array([amp[c] for c in M.comp])
    I = M.intensities(Trot)*a
    pos = M.L['Position'].to_numpy() + shift
    sig = sig0*(1 + sig1*(pos - 0.5*(M.lo + M.hi))/(M.hi - M.lo))
    order = np.argsort(pos)
    # clusters of lines within 0.5 sigma of each other, keeping only lines that matter
    strong = order[I[order] > 1e-4*np.nanmax(I)]
    groups, cur = [], [strong[0]]
    for k in strong[1:]:
        if pos[k] - pos[cur[-1]] < 0.5*sig[k]: cur.append(k)
        else: groups.append(cur); cur = [k]
    groups.append(cur)
    for g in groups:
        g = np.array(g)
        if np.min(M.t_ref[g]) < 0.97: continue
        p0 = np.average(pos[g], weights = I[g]); s0 = np.mean(sig[g])
        w = np.abs(M.wav - p0) < 3*s0
        if w.sum() < 6 or not np.all(np.isfinite(M.sky[w])): continue
        x = M.wav[w]
        near = np.nonzero(np.abs(pos - p0) < 15*s0 + 1)[0]
        others = sum(I[k]*prof(x, pos[k], sig[k]) for k in near if k not in g)
        own = sum(I[k]*prof(x, pos[k], sig[k]) for k in g)
        if np.sum(own) < OWN_MIN*(np.sum(own) + np.sum(np.abs(others))): continue
        yd = M.sky[w] - cont[w] - others
        ew = np.sqrt(M.err[w]**2 + (0.01*np.abs(M.sky[w]))**2)
        wts = I[g]/I[g].sum(); offs = pos[g] - p0
        def model(p):
            A_, d, s, c = p
            return A_*sum(wi*prof(x, p0 + oi + d, s) for wi, oi in zip(wts, offs)) + c
        A0 = np.sum(yd)*np.median(np.diff(x))
        if not np.isfinite(A0) or A0 <= 0: continue
        sol = least_squares(lambda p: (model(p) - yd)/ew, [A0, 0.0, s0, 0.0], x_scale = [A0, 0.005, 0.005, A0/s0/20],
                            bounds = ([0, -0.05, 0.3*s0, -np.inf], [np.inf, 0.05, 3*s0, np.inf]))
        J = sol.jac; cov = np.linalg.pinv(J.T @ J)*max(np.sum(sol.fun**2)/(len(x) - 4), 1)
        snr = sol.x[0]/np.sqrt(cov[0, 0])
        if snr < ARGS.snr: continue
        # identify the line by its strongest component's model wavelength, which is the same in every stack
        rows.append(dict(arm = arm, visit = visit, sm = sm, wav = round(float(M.L['Position'].to_numpy()[g[np.argmax(I[g])]]), 5), sigma = sol.x[2], sigma_err = np.sqrt(cov[2, 2]), snr = snr,
                         n_comp = len(g), comp_spread = np.ptp(pos[g]), own_frac = np.sum(own)/(np.sum(own) + np.sum(np.abs(others)))))
D = pd.DataFrame(rows)
D.to_csv(osp.join(HERE, f'lsf_vs_wavelength{ARGS.tag}.csv'), index = False)

say(f'# Line spread vs wavelength within each arm (pfsMerged stacks, per-line sigma with the blend structure fixed; own flux >= {OWN_MIN}, S/N >= {ARGS.snr})')
fitres = {}
for arm, g in D.groupby('arm'):
    lo, hi = g.wav.min(), g.wav.max(); c = 0.5*(lo + hi)
    # per spectrograph and line: median over that spectrograph's stacks
    P = g.groupby(['sm', 'wav']).agg(sigma = ('sigma', 'median'), sd = ('sigma', 'std'), n = ('sigma', 'size')).reset_index()
    P = P[P.n >= 2]
    sms = sorted(P.sm.unique())
    def design(df, quad):
        cols = [(df.sm == s).astype(float) for s in sms] + [df.wav - c] + ([(df.wav - c)**2] if quad else [])
        return np.column_stack(cols)
    def fit(df, quad = False):
        X = design(df, quad); coef = np.linalg.lstsq(X, df.sigma.to_numpy(), rcond = None)[0]
        return coef, df.sigma.to_numpy() - X @ coef
    coef, res = fit(P); coefq, resq = fit(P, quad = True)
    # bootstrap over the stacks: resample stacks within each spectrograph, recompute the medians and the slope
    rng = np.random.default_rng(0); boots = []
    stacks = g[['visit', 'sm']].drop_duplicates().to_numpy()
    for _ in range(300):
        pick = stacks[rng.integers(0, len(stacks), len(stacks))]
        gb = pd.concat([g[(g.visit == v) & (g.sm == s)] for v, s in pick])
        Pb = gb.groupby(['sm', 'wav']).sigma.median().reset_index()
        Pb = Pb[Pb.sm.isin(sms)]
        if Pb.sm.nunique() < len(sms): continue
        boots.append(fit(Pb)[0][len(sms)])
    s1, es1 = coef[len(sms)], np.std(boots)
    s_mid = np.mean(coef[:len(sms)])
    frac_change = s1*(hi - lo)/s_mid
    constR = s_mid/c # slope for sigma proportional to lambda
    scatter = 1.4826*np.median(np.abs(res))
    say(f'\n## {arm} arm: {P.wav.nunique()} lines, {lo:.0f}-{hi:.0f} nm, spectrographs {sms}')
    say(f'  sigma at {c:.0f} nm, per spectrograph: ' + ', '.join(f'SM{s} {coef[i]:.4f}' for i, s in enumerate(sms)) + ' nm')
    say(f'  linear slope {1e3*s1:+.4f} +- {1e3*es1:.4f} pm/nm ({s1/es1:+.1f} sigma): sigma changes by {100*frac_change:+.1f}% across the measured range')
    say(f'  for constant resolving power it would be {1e3*constR:+.4f} pm/nm ({100*constR*(hi - lo)/s_mid:+.1f}%); measured / constant-R slope = {s1/constR:.2f}')
    say(f'  quadratic term {1e3*coefq[-1]:+.6f} pm/nm^2; scatter of lines about the linear fit {1e3*scatter:.2f} pm ({100*scatter/s_mid:.1f}% of sigma), '
        f'rms reduction from the quadratic term {100*(1 - np.std(resq)/np.std(res)):.1f}%')
    per_sm = []
    for s in sms:
        Ps = P[P.sm == s]
        if len(Ps) > 5:
            cs = np.polyfit(Ps.wav - c, Ps.sigma, 1); per_sm.append(f'SM{s} {100*cs[0]*(hi - lo)/cs[1]:+.1f}%')
    say('  change across the range, each spectrograph on its own: ' + ', '.join(per_sm))
    fitres[arm] = dict(P = P, coef = coef, sms = sms, c = c, lo = lo, hi = hi, s1 = s1, es1 = es1, s_mid = s_mid)
open(osp.join(HERE, f'lsf_vs_wavelength{ARGS.tag}.txt'), 'w').write('\n'.join(out) + '\n')

#############
### Plot ###
#############
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
INK, INK2, GRID, BLUE, GREY = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6', '#8f8e8a'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.8})
fig, axs = plt.subplots(1, len(fitres), figsize = (4.2*len(fitres), 3.6), squeeze = False)
for ax, (arm, r) in zip(axs[0], fitres.items()):
    P = r['P']; sms = r['sms']
    # remove each spectrograph's offset so all lines sit on one trend
    off = dict(zip(sms, r['coef'][:len(sms)] - r['s_mid']))
    y = P.sigma - P.sm.map(off)
    L = P.assign(y = y).groupby('wav').agg(y = ('y', 'mean'), e = ('y', lambda v: np.std(v)/np.sqrt(len(v)) if len(v) > 1 else np.nan)).reset_index()
    ax.grid(True, color = GRID, linewidth = 0.8); ax.set_axisbelow(True)
    xx = np.linspace(r['lo'], r['hi'], 100)
    ax.plot(xx, 1e3*r['s_mid']*xx/r['c'], color = GREY, linewidth = 1.4, label = 'constant resolving power')
    ax.plot(xx, 1e3*(r['s_mid'] + r['s1']*(xx - r['c'])), color = BLUE, linewidth = 2, label = 'linear fit')
    ax.errorbar(L.wav, 1e3*L.y, yerr = 1e3*L.e, fmt = 'o', ms = 3.5, color = INK, ecolor = INK, elinewidth = 0.8, capsize = 0, mec = 'white', mew = 0.6, zorder = 3, label = 'lines (mean over spectrographs)')
    ax.set_title(f"{arm} arm: {100*r['s1']*(r['hi'] - r['lo'])/r['s_mid']:+.1f}% across ({r['s1']/r['es1']:+.0f}$\\sigma$)", loc = 'left', fontsize = 10, color = INK)
    ax.set_xlabel('wavelength [nm]', color = INK2); ax.set_ylabel('line sigma [pm] (pfsMerged)', color = INK2)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc = 'upper left', ncol = 3, frameon = False, fontsize = 8.5, bbox_to_anchor = (0.005, 1.0))
fig.tight_layout(rect = (0, 0, 1, 0.9))
os.makedirs(osp.join(HERE, 'figs'), exist_ok = True)
fig.savefig(osp.join(HERE, 'figs', f'lsf_vs_wavelength{ARGS.tag}.png'), dpi = 150)
print('saved figs/lsf_vs_wavelength.png')
