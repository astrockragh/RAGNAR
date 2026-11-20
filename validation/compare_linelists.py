'''
Compare line lists for wavelength calibration: a simple list (strong lines merged under a resolution criterion) against the RAGNAR lists.

Simple list: every line in RAGNAR's input table (pgopher OH, HITRAN O2, atomic lines) at fiducial intensities (Trot 195 K, Tvib 1e4 K),
chain-merged when neighbours are closer than MERGE x sigma of the arm into an intensity-weighted centroid, keeping merged lines brighter
than I_MIN x the brightest. No transmission, variability or blend-structure information, which is the point of the comparison.

Test 1, PFS sky (pfsArm, sky fibres of the all-sky visits): in each sampled fibre, every list line's centroid is measured the same way
(Gaussian-weighted centroid with the arm's sigma, iterated, above a local background). Per fibre, measured - reference wavelength is fitted
with a Legendre polynomial (sigma clipped), as a wavelength-solution correction. Scores per list:
    n_used      lines used per fibre (S/N > SNR_MIN, surviving the clipping)
    rms_fit     robust rms of the residuals about the fibre's solution
    rms_noclip  rms of all S/N > SNR_MIN lines about the solution, without clipping, and frac_clipped: how much the clipping has to hide
    rms_cv      5-fold cross-validated rms: residuals of held-out lines, i.e. how well the solution predicts lines it was not fitted to
    rms_line    rms over lines of each line's median residual across fibres: systematic reference-wavelength errors (blends, positions)
    solution differences between lists, per fibre, over the wavelength range covered by that fibre's used lines
Test 2, simulated spectra (the saved RAGNAR spectra, which span the simulated conditions; the true wavelength scale is exact):
    bias and scatter of the measured centroid against each list's reference wavelength. RAGNAR's lists were derived from these spectra,
    so this test favours them; test 1 does not.

Usage: python compare_linelists.py ARM [--merge 2.355] [--imin 1e-3] [--fibres 40] [--sim-spectra PATH]
Writes linelist_comparison_{arm}.txt and linelist_comparison_{arm}_lines.csv next to this script.
'''
import sys, glob, argparse, os.path as osp
import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.polynomial import legendre

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(HERE))
from dev.species_intensities import get_pgopher_linelist, make_OH_intensities, get_O2_at_T, get_atomic_lines

DATA = '/scratch/gpfs/cj1223/pfs_sky'
VISITS = [129688, 129689, 129806, 129807]
DROP = {(129689, 3)} # n-arm offset of -9x the continuum in that exposure (validation/tier2)
SKY, GOOD = 2, 1     # pfs.datamodel TargetType.SKY, FiberStatus.GOOD
# the b arm has only ~15 lines with S/N > 10 per fibre, nearly all above 520 nm, so its correction is kept to degree 2
ARMS = {'b': dict(range = (420, 645), sigma = 0.088, tag = 'sweep_0.088_{c}_b_fix3_20261007', deg = 2),
        'r': dict(range = (635, 965), sigma = 0.112, tag = 'sweep_0.112_{c}_r_fix3_20261007', deg = 4),
        'n': dict(range = (965, 1260), sigma = 0.105, tag = 'sweep_0.105_{c}_n_fix3_20261007', deg = 4)}
SNR_MIN = 10
LISTS_DIR = osp.join(osp.dirname(HERE), 'lineLists')

###################
### Line lists ###
###################

def simple_list(lo, hi, sigma, merge, imin):
    '''Strong lines merged when closer than merge*sigma: intensity-weighted centroids, intensity cut relative to the brightest'''
    A = get_pgopher_linelist(); A['Intensity'] = make_OH_intensities(A, 195, 1e4)
    L = pd.concat([A[['Position', 'Intensity']], get_O2_at_T(195)[['Position', 'Intensity']], get_atomic_lines()[['Position', 'Intensity']]])
    L = L[(L.Position > lo - 1) & (L.Position < hi + 1)].sort_values('Position')
    L = L[L.Intensity > 1e-3*imin*L.Intensity.max()] # drop the vanishingly faint lines before merging
    pos, I = L.Position.to_numpy(), L.Intensity.to_numpy()
    group = np.concatenate([[0], np.cumsum(np.diff(pos) > merge*sigma)])
    G = pd.DataFrame(dict(pos = pos, I = I, g = group)).groupby('g').apply(lambda d: pd.Series(dict(wav = np.average(d.pos, weights = d.I), intensity = d.I.sum(), n = len(d))), include_groups = False)
    G = G[(G.intensity > imin*G.intensity.max()) & (G.wav > lo) & (G.wav < hi)]
    return G.reset_index(drop = True)

def ragnar_list(arm, cent, robust_only):
    d = pd.read_pickle(osp.join(LISTS_DIR, ARMS[arm]['tag'].format(c = cent) + '.pkl'))
    lo, hi = ARMS[arm]['range']
    d = d[(d.wav > lo) & (d.wav < hi)]
    if robust_only: d = d[d.robust]
    return d[['wav', 'intensity']].reset_index(drop = True)

####################
### Measurement ###
####################

def centroids(w, f, var, ref, sigma, n_iter = 4):
    '''Gaussian-weighted centroids near the reference wavelengths ref, for one spectrum (w ascending). Returns centroid, S/N.
    Background: median of the pixels between 3 and 6 sigma from the line. Every list is measured with exactly this.'''
    dx = np.median(np.diff(w))
    half = int(np.ceil(6*sigma/dx)) + 1
    i0 = np.searchsorted(w, ref)
    idx = np.clip(i0[:, None] + np.arange(-half, half + 1)[None, :], 0, len(w) - 1)
    x, y, v = w[idx], f[idx], var[idx]
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    d = x - ref[:, None]
    bgm = ok & (np.abs(d) > 3*sigma)
    bg = np.array([np.median(yy[m]) if m.sum() >= 3 else np.nan for yy, m in zip(y, bgm)])
    yb = np.where(ok, y - bg[:, None], 0.0)
    c = ref.copy()
    for _ in range(n_iter):
        wt = np.exp(-0.5*((x - c[:, None])/sigma)**2)*ok
        wt = np.where(np.abs(x - c[:, None]) < 2.5*sigma, wt, 0.0)
        s = np.sum(wt*yb, axis = 1)
        c_new = np.sum(wt*yb*x, axis = 1)/np.where(s > 0, s, np.nan)
        c = np.where(np.isfinite(c_new) & (np.abs(c_new - ref) < sigma), c_new, c)
    wt = np.exp(-0.5*((x - c[:, None])/sigma)**2)*ok*(np.abs(x - c[:, None]) < 2.5*sigma)
    snr = np.sum(wt*yb, axis = 1)/np.sqrt(np.sum(wt**2*np.where(ok, v, 0), axis = 1))
    snr[~np.isfinite(bg)] = np.nan
    return c, snr

def fit_solution(lam, off, lo, hi, deg, clip = 3.0, n_iter = 5):
    '''Legendre fit of measured - reference vs wavelength with iterative clipping. Returns coefficients and the used mask'''
    x = 2*(lam - lo)/(hi - lo) - 1
    use = np.isfinite(off)
    for _ in range(n_iter):
        if use.sum() < deg + 3: return None, use
        coef = legendre.legfit(x[use], off[use], deg)
        r = off - legendre.legval(x, coef)
        s = 1.4826*np.median(np.abs(r[use] - np.median(r[use])))
        new = np.isfinite(off) & (np.abs(r) < clip*max(s, 1e-6))
        if np.array_equal(new, use): break
        use = new
    return coef, use

def robust_rms(r):
    r = r[np.isfinite(r)]
    return 1.4826*np.median(np.abs(r - np.median(r))) if len(r) else np.nan

########################
### Test 1: PFS sky ###
########################

def test_sky(arm, lists, n_fibres, seed = 0):
    lo, hi = ARMS[arm]['range']; sigma = ARMS[arm]['sigma']; deg = ARMS[arm]['deg']
    rng = np.random.default_rng(seed)
    per_fibre = {name: [] for name in lists}; per_line = {name: [] for name in lists}; solutions = []
    grid = np.linspace(lo, hi, 200); xg = 2*(grid - lo)/(hi - lo) - 1
    for visit in VISITS:
        cfg = fits.open(glob.glob(f'{DATA}/**/pfsConfig_PFS_{visit}_*.fits', recursive = True)[0])['CONFIG'].data
        good = set(np.array(cfg['fiberId'])[(np.array(cfg['targetType']) == SKY) & (np.array(cfg['fiberStatus']) == GOOD)])
        for sm in [1, 2, 3, 4]:
            if (visit, sm) in DROP: continue
            f = glob.glob(f'{DATA}/**/pfsArm_PFS_{visit}_{arm}{sm}_*.fits', recursive = True)
            if not f: continue
            h = fits.open(f[0])
            W, F, N, M, C = [np.asarray(h[k].data, dtype = float) for k in ['WAVELENGTH', 'FLUX', 'NORM', 'MASK', 'COVAR']]
            fib = h['FIBERID'].data
            use = np.nonzero(np.isin(fib, list(good)))[0]
            for k in rng.choice(use, size = min(n_fibres, len(use)), replace = False):
                w = W[k]; flux = F[k]/N[k]; var = C[k, 0]/N[k]**2
                bad = (M[k] != 0) | ~np.isfinite(flux)
                flux = np.where(bad, np.nan, flux)
                sol_here = {}
                for name, L in lists.items():
                    ref = L.wav.to_numpy()
                    c, snr = centroids(w, flux, var, ref, sigma)
                    off = np.where(snr > SNR_MIN, c - ref, np.nan)
                    coef, used = fit_solution(ref, off, lo, hi, deg)
                    if coef is None: continue
                    x = 2*(ref - lo)/(hi - lo) - 1
                    resid = off - legendre.legval(x, coef)
                    # 5-fold cross validation over the used lines
                    iu = np.nonzero(used)[0]; folds = rng.permutation(len(iu)) % 5; cv = np.full(len(ref), np.nan)
                    for fo in range(5):
                        train = np.zeros(len(ref), bool); train[iu[folds != fo]] = True
                        cf, _ = fit_solution(ref, np.where(train, off, np.nan), lo, hi, deg, clip = 1e9, n_iter = 1)
                        if cf is not None:
                            test = iu[folds == fo]; cv[test] = off[test] - legendre.legval(x[test], cf)
                    meas = np.isfinite(off)
                    per_fibre[name].append(dict(visit = visit, sm = sm, fiber = fib[k], n_used = int(used.sum()), n_measured = int(meas.sum()),
                                                rms_fit = robust_rms(resid[used]), rms_noclip = np.sqrt(np.mean(resid[meas]**2)),
                                                frac_clipped = 1 - used.sum()/max(meas.sum(), 1), rms_cv = robust_rms(cv), wmin = np.percentile(ref[used], 5), wmax = np.percentile(ref[used], 95)))
                    per_line[name].append(pd.DataFrame(dict(line = np.arange(len(ref)), wav = ref, resid = np.where(used, resid, np.nan),
                                                            resid_all = np.where(meas, resid, np.nan), used = used, measured = meas, snr = snr,
                                                            visit = visit, sm = sm, fiber = fib[k])))
                    sol_here[name] = np.where((grid >= np.percentile(ref[used], 5)) & (grid <= np.percentile(ref[used], 95)), legendre.legval(xg, coef), np.nan)
                solutions.append(sol_here)
    return per_fibre, per_line, solutions, grid

##############################
### Test 2: simulated sky ###
##############################

def test_sim(arm, lists, spectra_path, n_spec = 60, seed = 0):
    import pickle
    wav, specs = pickle.load(open(spectra_path, 'rb'))
    sigma = ARMS[arm]['sigma']
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(specs), size = min(n_spec, len(specs)), replace = False)
    var = np.full(len(wav), 1e-12) # noiseless; the variance only sets the S/N, which is not used here
    out = {}
    for name, L in lists.items():
        ref = L.wav.to_numpy()
        off = np.array([centroids(wav, np.asarray(specs[i]), var, ref, sigma)[0] - ref for i in pick])
        out[name] = pd.DataFrame(dict(wav = ref, bias = np.median(off, axis = 0), scatter = np.std(off, axis = 0)))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('arm'); ap.add_argument('--merge', type = float, default = 2.355); ap.add_argument('--imin', type = float, default = 1e-3)
    ap.add_argument('--fibres', type = int, default = 40); ap.add_argument('--sim-spectra', default = None)
    a = ap.parse_args()
    arm = a.arm; lo, hi = ARMS[arm]['range']; sigma = ARMS[arm]['sigma']
    lists = {f'simple (merge {a.merge:g} sigma)': simple_list(lo, hi, sigma, a.merge, a.imin),
             'simple (merge 1 sigma)': simple_list(lo, hi, sigma, 1.0, a.imin),
             'RAGNAR cent=0, all': ragnar_list(arm, 0, False), 'RAGNAR cent=0, robust': ragnar_list(arm, 0, True),
             'RAGNAR cent=1, all': ragnar_list(arm, 1, False), 'RAGNAR cent=1, robust': ragnar_list(arm, 1, True)}
    out = [f'# Line lists for wavelength calibration, {arm} arm ({lo}-{hi} nm, sigma {sigma} nm)',
           f'Simple lists: lines merged closer than the given multiple of sigma, intensity cut {a.imin:g} x brightest. Centroids: Gaussian-weighted, S/N > {SNR_MIN}.',
           f'Per-fibre correction: Legendre degree {ARMS[arm]["deg"]}, 3-sigma clipped. Residuals in pm.', '']
    pf, pl, sols, grid = test_sky(arm, lists, a.fibres)
    rows = []
    for name in lists:
        P = pd.DataFrame(pf[name]); Lr = pd.concat(pl[name])
        line_med = Lr.groupby('line').resid.median(); line_n = Lr.groupby('line').resid.count()
        well = line_n > 0.5*len(P)
        rows.append(dict(list = name, lines_in_list = len(lists[name]), n_used = P.n_used.median(), frac_clipped = P.frac_clipped.median(), rms_fit_pm = 1e3*P.rms_fit.median(),
                         rms_noclip_pm = 1e3*P.rms_noclip.median(), rms_cv_pm = 1e3*P.rms_cv.median(),
                         rms_line_pm = 1e3*np.sqrt(np.mean(line_med[well]**2)), lines_used_in_half_the_fibres = int(well.sum())))
        Lr.assign(list = name).to_csv(osp.join(HERE, f'linelist_comparison_{arm}_lines.csv'), mode = 'w' if name == list(lists)[0] else 'a', header = name == list(lists)[0], index = False)
    T = pd.DataFrame(rows)
    out += ['## Test 1: PFS sky fibres (median over fibres)', T.to_string(index = False, float_format = lambda v: f'{v:.2f}'), '']
    # how much the solutions from different lists disagree, per fibre, over the arm
    names = list(lists); ref_name = 'RAGNAR cent=1, robust'
    out.append(f'## Solution differences against "{ref_name}" (rms over the range both lists cover with used lines, median over fibres, pm)')
    for n in names:
        if n == ref_name: continue
        d = [1e3*np.sqrt(np.nanmean((s[n] - s[ref_name])**2)) for s in sols if n in s and ref_name in s and np.isfinite(s[n] - s[ref_name]).any()]
        out.append(f'  {n}: {np.median(d):.2f} (84th percentile {np.percentile(d, 84):.2f})')
    if a.sim_spectra:
        S = test_sim(arm, lists, a.sim_spectra)
        out += ['', f'## Test 2: simulated spectra ({osp.basename(a.sim_spectra)}; favours RAGNAR, whose lists were derived from them). pm']
        for n, d in S.items():
            out.append(f'  {n}: lines {len(d)}, rms bias {1e3*np.sqrt(np.mean(d.bias**2)):.2f}, median |bias| {1e3*np.median(np.abs(d.bias)):.2f}, '
                       f'90th pct |bias| {1e3*np.percentile(np.abs(d.bias), 90):.2f}, median scatter over conditions {1e3*np.median(d.scatter):.2f}')
    txt = '\n'.join(out); print(txt)
    open(osp.join(HERE, f'linelist_comparison_{arm}.txt'), 'w').write(txt + '\n')
