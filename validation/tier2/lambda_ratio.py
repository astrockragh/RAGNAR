'''
Measure the intensity ratio of the two components of resolved OH Lambda-doublets in the PFS sky, against the model (pgopher Einstein-A
coefficients, which make the e and f components nearly equal).

Doublets: clean, robust lines of the _fix3 centroider lists made of exactly the two Lambda components of one upper level, with separations
> MIN_SEP (from ../lambda_doublets.csv). Per stack (visit x spectrograph, per-v' forward-model fits in fits/):
  1. everything except the doublet is subtracted using the forward-model fit (continuum polynomial + all other lines, with the fitted
     amplitudes, LSF and shift)
  2. the doublet is fitted with two profiles at the pgopher positions (separation fixed), with a common shift and width, two free
     amplitudes and a constant
  3. ratio = red/blue observed amplitude, divided by the model's red/blue intensity ratio (incl. transmission) -> 1 if the model is right
The same fit with the ratio forced to the model's is kept for the figures.
Writes lambda_ratio.csv, figs/lambda_doublet_fits.png (the splits) and figs/lambda_ratio_summary.png
'''
import sys, glob, re, os, os.path as osp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(osp.dirname(HERE)))
from sky_model import ArmModel, tapered_profile, R_SCAT
from dev.species_intensities import get_pgopher_linelist

MIN_SEP = 0.060 # nm
A = get_pgopher_linelist()
A_pos = A['Position'].to_numpy()
cand = pd.read_csv(osp.join(osp.dirname(HERE), 'lambda_doublets.csv'))
cand = cand[cand.sep_pm > 1e3*MIN_SEP]
R = pd.read_csv(osp.join(osp.dirname(HERE), 'clipped_robust_lines.csv'))

def components(arm, wav):
    '''the two pgopher lines of a doublet (blue first), from the list's subline wavelengths'''
    r = R[(R.arm == arm) & np.isclose(R.wav, wav)].iloc[0]
    sw = np.sort([float(x) for x in r.subline_wav.strip('[]').split()])
    rows = [A.iloc[np.argmin(np.abs(A_pos - w))] for w in sw]
    return rows

def prof(x, pos, sig):
    return tapered_profile((x - pos)[None, :], np.array([sig]))[0]

results, fits_for_plot = [], {}
for f in sorted(glob.glob(osp.join(HERE, 'fits', 'fit_*_sm?_[rn].npz'))):
    visit, sm, arm = re.match(r'.*fit_(\d+)_sm(\d)_([brn])\.npz', f).groups(); visit, sm = int(visit), int(sm)
    C = cand[cand.arm == arm]
    if not len(C): continue
    fit = np.load(f); st = np.load(osp.join(HERE, f'sky_stack_{visit}_sm{sm}.npz'))
    M = ArmModel(arm, st['wav'], st['sky'], st['err'], float(st['airmass']), pwv = float(fit['pwv']), mask_t = 0.97)
    assert np.allclose(M.wav, fit['wav'])
    cont = fit['continuum']
    Trot, sig0, sig1, shift = float(fit['Trot']), float(fit['sig0']), float(fit['sig1']), float(fit['shift'])
    amp = dict(zip(fit['components'], fit['coef'])); a = np.array([amp[c] for c in M.comp])
    I = M.intensities(Trot)*a # fitted flux of every line (transmission included)
    pos = M.L['Position'].to_numpy() + shift
    sig = sig0*(1 + sig1*(pos - 0.5*(M.lo + M.hi))/(M.hi - M.lo))
    for _, c in C.iterrows():
        comp = components(arm, c.wav)
        idx = [int(np.argmin(np.abs(M.L['Position'].to_numpy() - r.Position))) for r in comp]
        if any(abs(M.L['Position'].to_numpy()[i] - r.Position) > 1e-5 for i, r in zip(idx, comp)): continue
        s0 = np.mean(sig[idx]); centre = np.mean(pos[idx])
        w = np.abs(M.wav - centre) < 5*s0
        x, y, e = M.wav[w], M.sky[w] - cont[w], M.err[w]
        # subtract every other line in the window
        near = np.nonzero(np.abs(pos - centre) < 15*s0 + 1)[0]
        others = sum(I[k]*prof(x, pos[k], sig[k]) for k in near if k not in idx)
        own_model = sum(I[k]*prof(x, pos[k], sig[k]) for k in idx)
        yd = y - others
        p_b, p_r = pos[idx[0]], pos[idx[1]]
        ratio_model = I[idx[1]]/I[idx[0]]
        def model(p, x, forced = None):
            ab, ar, d, s, c0 = p if forced is None else (p[0], p[0]*forced, p[1], p[2], p[3])
            return ab*prof(x, p_b + d, s) + ar*prof(x, p_r + d, s) + c0
        ew = np.sqrt(e**2 + (0.01*np.abs(y))**2)
        if not (np.all(np.isfinite(yd)) and np.all(np.isfinite(ew)) and np.isfinite(s0) and np.isfinite(ratio_model) and ratio_model > 0): continue
        A0 = max(np.sum(yd)*np.median(np.diff(x))/2, 1e-12)
        x0 = np.array([A0, A0*ratio_model, 0.0, s0, 0.0], dtype = float)
        if not np.all(np.isfinite(x0)): print("skip", c.wav, x0); continue
        sol = least_squares(lambda p: (model(p, x) - yd)/ew, x0, x_scale = [A0, A0, 0.005, 0.01, A0/s0/10])
        solf = least_squares(lambda p: (model(p, x, ratio_model) - yd)/ew, [A0, 0.0, s0, 0.0], x_scale = [A0, 0.005, 0.01, A0/s0/10])
        J = sol.jac; cov = np.linalg.pinv(J.T @ J)*max(np.sum(sol.fun**2)/(len(x) - 5), 1)
        ab, ar = sol.x[:2]; eab, ear = np.sqrt(np.diag(cov))[:2]
        rr = ar/ab; err_rr = abs(rr)*np.sqrt((eab/ab)**2 + (ear/ar)**2)
        lab = [f"({int(r[chr(118)+chr(39)])}-{int(r[chr(118)+chr(34)])}){r.Branch}" for r in comp]
        par = [re.search(r'([ef])\(', l).group(1) for l in lab]
        results.append(dict(arm = arm, visit = visit, sm = sm, wav = c.wav, label = lab[0][:-1].rstrip('ef(0123456789.') , blue = lab[0], red = lab[1], blue_parity = par[0], red_parity = par[1],
                            branch = c.branch, J = c.J, sep_pm = 1e3*(p_r - p_b), ratio_obs = rr, ratio_obs_err = err_rr, ratio_model = ratio_model, ratio_corr = rr/ratio_model,
                            contamination = np.sum(np.abs(others))/max(np.sum(own_model), 1e-30), chi2_free = np.sum(sol.fun**2), chi2_forced = np.sum(solf.fun**2), npix = len(x)))
        if visit == 129806 and sm == 2:
            fits_for_plot[c.wav] = dict(arm = arm, x = x, yd = yd, e = e, free = model(sol.x, x), forced = model(solf.x, x, ratio_model),
                                        blue = sol.x[0]*prof(x, p_b + sol.x[2], sol.x[3]) + sol.x[4], red = sol.x[1]*prof(x, p_r + sol.x[2], sol.x[3]) + sol.x[4],
                                        p_b = p_b + sol.x[2], p_r = p_r + sol.x[2], lab = lab, centre = centre, sig = s0)
D = pd.DataFrame(results)
D.to_csv(osp.join(HERE, 'lambda_ratio.csv'), index = False)
S = D[D.contamination < 0.3].groupby(['arm', 'wav']).agg(blue = ('blue', 'first'), red = ('red', 'first'), red_parity = ('red_parity', 'first'), branch = ('branch', 'first'),
                                                        J = ('J', 'first'), sep_pm = ('sep_pm', 'first'), rcorr = ('ratio_corr', 'mean'), rcorr_sd = ('ratio_corr', 'std'),
                                                        n = ('ratio_corr', 'size'), model = ('ratio_model', 'first'), contamination = ('contamination', 'median'),
                                                        dchi2 = ('chi2_forced', 'sum')).reset_index()
S['dchi2'] = S.dchi2 - D[D.contamination < 0.3].groupby(['arm', 'wav']).chi2_free.sum().to_numpy()
S['rcorr_err'] = S.rcorr_sd/np.sqrt(S.n)
# the doublet's split in units of the line width: below ~1.4 sigma, a common shift and the component ratio both just move the centroid,
# so the free-shift fit can't separate them and the ratio is not measured
sig_arm = {'r': 0.108, 'n': 0.098} # pfsMerged line widths from the forward-model fits
S['split_sigma'] = S.sep_pm/1e3/S.arm.map(sig_arm)
S['resolved'] = S.split_sigma >= 1.4
print(S.sort_values(['arm', 'sep_pm']).to_string(index = False, float_format = lambda v: f'{v:.3f}'))
S.to_csv(osp.join(HERE, 'lambda_ratio_summary.csv'), index = False)

#############
### Plots ###
#############
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
INK, INK2, GRID, BLUE, ORANGE, AQUA, GREY = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6', '#eb6834', '#1baf7a', '#8f8e8a'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.8})
os.makedirs(osp.join(HERE, 'figs'), exist_ok = True)

# figure 1: the splits, widest separations first, in one moon-down stack
keys = [k for k in sorted(fits_for_plot, key = lambda k: -S.loc[np.isclose(S.wav, k), 'sep_pm'].max() if np.isclose(S.wav, k).any() else 0)
        if np.isclose(S.wav, k).any() and S.loc[np.isclose(S.wav, k), 'resolved'].iloc[0]][:12]
ncol = 4; nrow = int(np.ceil(len(keys)/ncol))
fig, axs = plt.subplots(nrow, ncol, figsize = (3.3*ncol, 2.7*nrow + 0.6), squeeze = False)
for ax, k in zip(axs.flat, keys):
    p = fits_for_plot[k]; srow = S[np.isclose(S.wav, k)].iloc[0]
    dx = p['x'] - p['centre']
    ax.grid(True, color = GRID, linewidth = 0.8); ax.set_axisbelow(True)
    ax.errorbar(dx, p['yd'], yerr = p['e'], fmt = 'o', ms = 3.5, color = INK, elinewidth = 0.8, capsize = 0, mec = 'white', mew = 0.6, zorder = 4, label = 'PFS sky, other lines subtracted')
    ax.plot(dx, p['forced'], color = GREY, linewidth = 1.4, zorder = 2, label = 'fit with the model ratio')
    ax.plot(dx, p['free'], color = BLUE, linewidth = 2, zorder = 3, label = 'fit with free ratio')
    ax.plot(dx, p['blue'], color = AQUA, linewidth = 1.2, zorder = 2, label = 'blue component')
    ax.plot(dx, p['red'], color = ORANGE, linewidth = 1.2, zorder = 2, label = 'red component')
    ax.set_title(f"{p['lab'][0].split('(')[0][:5]}{p['lab'][0][5:].split('(')[0][:-1]}({p['lab'][0].split('(')[-1]}  {k:.2f} nm", loc = 'left', fontsize = 9, color = INK)
    ax.text(0.03, 0.95, f"split {srow.sep_pm:.0f} pm ({srow.split_sigma:.1f}$\\sigma$)\nred/blue {srow.model*srow.rcorr:.2f} ± {srow.model*srow.rcorr_err:.2f}\nmodel {srow.model:.2f}; {srow.red_parity} is red",
            transform = ax.transAxes, va = 'top', fontsize = 7.8, color = INK2)
    ax.set_xlabel('wavelength - doublet centre [nm]', color = INK2)
for ax in list(axs.flat)[len(keys):]:
    ax.axis('off')
h, l = axs.flat[0].get_legend_handles_labels()
fig.suptitle('Resolved OH Lambda-doublets (split >= 1.4 sigma) in the PFS sky, 129806 SM2: two-component fits (ratio ± error over 15 stacks)', x = 0.01, y = 0.995, ha = 'left', fontsize = 10.5, color = INK)
fig.legend(h, l, loc = 'upper left', ncol = 5, frameon = False, fontsize = 8.5, bbox_to_anchor = (0.005, 0.965))
fig.tight_layout(rect = (0, 0, 1, 0.91))
fig.savefig(osp.join(HERE, 'figs', 'lambda_doublet_fits.png'), dpi = 150)

# figure 2: measured / model ratio against the split, per branch type
fig, ax = plt.subplots(figsize = (7.5, 4.2))
ax.grid(True, color = GRID, linewidth = 0.8); ax.set_axisbelow(True)
ax.axhline(1, color = INK2, linewidth = 1)
ax.axvspan(0, 1.4*np.mean(list(sig_arm.values()))*1e3, color = GRID, alpha = 0.6, lw = 0)
ax.text(5, 0.97, 'split < 1.4 sigma: ratio\ndegenerate with the shift', fontsize = 7.5, color = INK2, va = 'top', transform = ax.get_xaxis_transform())
styles = {'P': (BLUE, 'o', 'P branch'), 'Q': (ORANGE, 's', 'Q branch'), 'R': (AQUA, '^', 'R branch'), 'O': (GREY, 'D', 'O branch')}
for b, (col, mk, name) in styles.items():
    g = S[(S.branch.str[1] == b) & (S.contamination < 0.3)]
    if not len(g): continue
    gr, gu = g[g.resolved], g[~g.resolved]
    ax.errorbar(gr.sep_pm, gr.rcorr, yerr = gr.rcorr_err, fmt = mk, ms = 6, color = col, mec = 'white', mew = 1.0, elinewidth = 1, capsize = 0, label = f'{name} ({len(gr)} resolved)')
    ax.errorbar(gu.sep_pm, gu.rcorr, yerr = gu.rcorr_err, fmt = mk, ms = 5, color = col, mfc = 'white', mew = 1.0, elinewidth = 0.8, capsize = 0, alpha = 0.7)
ax.set_xlabel('Lambda-doublet split [pm]', color = INK2)
ax.set_ylabel('(red/blue) measured / (red/blue) model', color = INK2)
ax.set_ylim(0, 3)
ax.set_title('Lambda-doublet component ratio in the PFS sky, relative to the pgopher model', loc = 'left', fontsize = 10, color = INK)
ax.legend(frameon = False, fontsize = 8.5, loc = 'upper right')
fig.tight_layout()
fig.savefig(osp.join(HERE, 'figs', 'lambda_ratio_summary.png'), dpi = 150)
print('saved figs/lambda_doublet_fits.png and figs/lambda_ratio_summary.png')
