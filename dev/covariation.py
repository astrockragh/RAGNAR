'''
Lines whose intensities vary together, from the simulated spectra.

Every spectrum of a RAGNAR run is the sky at one point of the grid (Trot, Tvib, hot population, chemistry, PWV, airmass, LSF).
For a pair of lines i, j the scatter of ln(I_i/I_j) over those spectra says how well their ratio is fixed: ~0 for lines from one
upper state with the same transmission and no blending, larger when they come from different levels, are blended with lines that
vary differently, or sit in absorption that changes with PWV. Lines are grouped by complete linkage on that scatter, so every pair
inside a group has a ratio scatter below the threshold, and a group's lines can be used together for relative flux calibration.

Intensities are the line-only spectrum at the line centre, like the `intensity` column (no centroider smoothing, no continuum).
Lines from several runs (e.g. the arms of one instrument) can be grouped together with combine_covariation, as long as the runs
simulated the same skies: same grid and the same seed for the chemistry draws (make_intensity_ratios).
'''
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .utils import nearest_index
from .species_intensities import get_pgopher_linelist, get_O2_at_T, get_atomic_lines, make_OH_intensities, DATA_PATH

ATOMIC_UPPER = {'OI(1S_0-1D_2)': 'O(1S)', 'OI(1D_2-3P_2)': 'O(1D)', 'OI(1D_2-3P_1)': 'O(1D)', 'OI(1D_2-3P_0)': 'O(1D)',
                'NaI(D2)': 'Na(3p 2P3/2)', 'NaI(D1)': 'Na(3p 2P1/2)', 'H_alpha': 'H(n=3)', 'H_beta': 'H(n=4)'}

# what is varied per simulated spectrum, in the order of get_spectra_and_peaks(..., return_params = True)
PARAM_NAMES = ['pwv', 'zd', 'Trot', 'Tvib', 'r_hot_scale', 'lsf_factor', 'chem_alpha', 'X_OH', 'X_O2b', 'X_O2a', 'X_Ogreen', 'X_Ored', 'X_Na']

def line_intensity_matrix(specs, wav_range, wavs):
    '''
    Intensity of every line in every simulated spectrum, read at the line centre (the nearest grid point, as define_lines_and_intensities does)
    Returns an (Nspec, Nline) array
    '''
    idx = nearest_index(wav_range, np.asarray(wavs, dtype = float))
    return np.vstack([np.asarray(s)[idx] for s in specs])

def ratio_scatter(I, floor = 1e-12):
    '''
    S[i, j] = standard deviation over the spectra of ln(I_i/I_j), for an (Nspec, Nline) intensity array.
    0.01 means the ratio of the two lines is fixed to ~1% over everything the run varies.
    '''
    L = np.log(np.maximum(I, floor))
    L = L - L.mean(axis = 0)
    C = L.T @ L/len(L)
    d = np.diag(C)
    S = np.sqrt(np.maximum(d[:, None] + d[None, :] - 2*C, 0))
    S = (S + S.T)/2
    np.fill_diagonal(S, 0)
    return S

def covariation_groups(S, threshold = 0.01):
    '''
    Complete-linkage groups on the ratio scatter: every pair of lines in a group has S below threshold.
    Groups are numbered from 0 in order of their first line (lines sorted by wavelength give groups sorted by wavelength).
    '''
    n = len(S)
    if n < 2:
        return np.zeros(n, dtype = int)
    lab = fcluster(linkage(squareform(S, checks = False), method = 'complete'), t = threshold, criterion = 'distance')
    _, first = np.unique(lab, return_index = True)
    remap = {lab[f]: r for r, f in enumerate(np.sort(first))}
    return np.array([remap[l] for l in lab])

def line_table(min_wav, max_wav, Trot = 195, Tvib = 1e4, data_path = DATA_PATH):
    '''
    All lines in (min_wav, max_wav) with their intrinsic intensity [kR] at (Trot, Tvib), and labels for their upper state (up) and
    upper vibrational level or species (vgroup). OH upper states are (v', J', N'), so the two Lambda-doublet components share one;
    O2 upper states come from the HITRAN upper-state energy.
    '''
    A = get_pgopher_linelist(path = data_path)
    A['Intensity'] = make_OH_intensities(A, Trot, Tvib) # normalized over the full list, so cut afterwards
    L = pd.concat([A, get_O2_at_T(Trot, path = data_path), get_atomic_lines(path = data_path)])
    L = L[(L.Position > min_wav) & (L.Position < max_wav)].sort_values('Position', kind = 'mergesort')
    sp = L.Species.to_numpy()
    up, vg = np.empty(len(L), dtype = object), np.empty(len(L), dtype = object)
    oh = sp == 'OH'
    vu = L["v'"].to_numpy()[oh].astype(int)
    up[oh] = [f"OH v'={v} J'={j} N'={int(n)}" for v, j, n in zip(vu, L["J'"].to_numpy()[oh], L["N'"].to_numpy()[oh])]
    vg[oh] = [f"OH v'={v}" for v in vu]
    o2 = sp == 'O2'
    if o2.any():
        band = np.where(L.Position.to_numpy()[o2] < 1000, 'O2 b', 'O2 a')
        up[o2] = [f"{b} E'={e:.1f}" for b, e in zip(band, L['eupper'].to_numpy()[o2])]
        vg[o2] = band
    at = ~(oh | o2)
    up[at] = [ATOMIC_UPPER.get(l, l) for l in L.Label.to_numpy()[at]]
    vg[at] = sp[at]
    return pd.DataFrame({'pos': L.Position.to_numpy(), 'I': L.Intensity.to_numpy(), 'up': up, 'vgroup': vg})

def upper_states(wavs, sigma, min_wav, max_wav, Trot = 195, Tvib = 1e4, data_path = DATA_PATH, nsig = 8):
    '''
    Line-centre decomposition with a Gaussian LSF of width sigma (scalar or one per line), no absorption:
    the upper state and the upper vibrational level/species that contribute most at each line centre, and their fractions.
    '''
    T = line_table(min(wavs) - nsig*np.max(sigma), max(wavs) + nsig*np.max(sigma), Trot, Tvib, data_path)
    T = T[(T.pos > min_wav - nsig*np.max(sigma)) & (T.pos < max_wav + nsig*np.max(sigma))]
    pos, I = T.pos.to_numpy(), T.I.to_numpy()
    cu, uu = pd.factorize(T.up)
    cv, uv = pd.factorize(T.vgroup)
    wavs = np.asarray(wavs, dtype = float)
    sig = np.broadcast_to(np.atleast_1d(sigma), wavs.shape)
    lo, hi = np.searchsorted(pos, wavs - nsig*sig), np.searchsorted(pos, wavs + nsig*sig)
    out = {'upper_state': [], 'upper_frac': [], 'vgroup': [], 'vgroup_frac': []}
    for w, s, a, b in zip(wavs, sig, lo, hi):
        c = I[a:b]*np.exp(-0.5*((w - pos[a:b])/s)**2)
        if c.sum() <= 0:
            out['upper_state'].append(''); out['upper_frac'].append(np.nan); out['vgroup'].append(''); out['vgroup_frac'].append(np.nan)
            continue
        fu = np.bincount(cu[a:b], weights = c, minlength = len(uu))/c.sum()
        fv = np.bincount(cv[a:b], weights = c, minlength = len(uv))/c.sum()
        out['upper_state'].append(uu[fu.argmax()]); out['upper_frac'].append(round(fu.max(), 4))
        out['vgroup'].append(uv[fv.argmax()]); out['vgroup_frac'].append(round(fv.max(), 4))
    return pd.DataFrame(out)

def _group_columns(S, groups, considered, n):
    ''' Per-line covar_group, covar_n, covar_scatter (largest scatter to the other members) and covar_nearest (smallest scatter to any other line) '''
    g = np.full(n, -1); size = np.zeros(n, dtype = int); worst = np.full(n, np.nan); nearest = np.full(n, np.nan)
    idx = np.flatnonzero(considered)
    g[idx] = groups
    counts = np.bincount(groups)
    size[idx] = counts[groups]
    S_off = S + np.diag(np.full(len(S), np.inf))
    nearest[idx] = S_off.min(axis = 1) if len(S) > 1 else np.nan
    same = groups[:, None] == groups[None, :]
    worst[idx] = np.where(same, S, 0).max(axis = 1)
    return g, size, worst, nearest

def add_covariation(lineDf, specs, wav_range, sigma, threshold = 0.01, robust_only = True, min_intensity = 0.0, Trot = 195, Tvib = 1e4, data_path = DATA_PATH):
    '''
    Find the lines that vary together in a run and add to lineDf
        upper_state, upper_frac, vgroup, vgroup_frac: dominant upper state / v' (or species) at the line centre, and its fraction
        covar_group: group number, -1 for lines not considered (not robust, or fainter than min_intensity)
        covar_n: number of lines in the group
        covar_scatter: the largest scatter of ln(I_i/I_j) to the other lines in the group (0 for a line on its own)
        covar_nearest: the smallest scatter of ln(I_i/I_j) to any other considered line, i.e. how close a single line is to having a partner
        linear_group, linear_n, linear_scatter, linear_nearest: the same for lines whose intensities lie on a straight line with each other
            (intercept allowed, linear_scatter), which includes the fixed-ratio pairs
    Inputs:
        specs, wav_range: the line-only simulated spectra (no continuum) and their wavelength grid
        sigma: the Gaussian LSF width [nm], scalar or one per line, for the upper-state decomposition
        threshold: largest ratio scatter allowed inside a group (0.01 = the ratio is fixed to 1%)
    Returns lineDf and a dict with the intensity array I (Nspec, Nline, all lines), the ratio scatter S of the considered lines and their indices
    '''
    lineDf = lineDf.copy()
    I = line_intensity_matrix(specs, wav_range, lineDf.wav)
    considered = np.ones(len(lineDf), dtype = bool)
    if robust_only:
        considered &= lineDf.robust.to_numpy(dtype = bool)
    considered &= lineDf.intensity.to_numpy() > min_intensity
    S = ratio_scatter(I[:, considered])
    groups = covariation_groups(S, threshold)
    us = upper_states(lineDf.wav.to_numpy(), sigma, wav_range[0], wav_range[-1], Trot, Tvib, data_path)
    for c in us.columns:
        lineDf[c] = us[c].to_numpy()
    lineDf['covar_group'], lineDf['covar_n'], lineDf['covar_scatter'], lineDf['covar_nearest'] = _group_columns(S, groups, considered, len(lineDf))
    L = linear_scatter(I[:, considered]) # the same, for lines that change linearly together (intercept allowed)
    lineDf['linear_group'], lineDf['linear_n'], lineDf['linear_scatter'], lineDf['linear_nearest'] = _group_columns(L, covariation_groups(L, threshold), considered, len(lineDf))
    return lineDf, {'I': I, 'S': S, 'considered': np.flatnonzero(considered), 'threshold': threshold}

def combine_covariation(runs, threshold = 0.01, robust_only = True, min_intensity = 0.0, names = None, metric = 'ratio'):
    '''
    Group lines across several runs that simulated the same skies (e.g. the arms of a spectrograph, run with the same grid and seed).
    runs: list of (lineDf with the add_covariation columns, I (Nspec, Nline), params (Nspec, len(PARAM_NAMES)) or None), or of paths to the
          *_covar.npz files written by ragnar()
    Every run must have the same number of spectra, and the emission parameters (everything but lsf_factor) must agree spectrum by spectrum.
    metric: 'ratio' (fixed ratio, ratio_scatter, columns covar_*_all) or 'linear' (on a straight line, linear_scatter, columns linear_*_all)
    Returns a DataFrame of the considered lines of all runs (run, line_index, wav, intensity, upper_state, vgroup, ..., covar_group_all
    across runs) and their ratio scatter S
    '''
    loaded = [load_covariation(r) if isinstance(r, str) else r for r in runs]
    names = names or [f'run{k}' for k in range(len(loaded))]
    ref = None
    tabs, Is = [], []
    for name, (df, I, P) in zip(names, loaded):
        if P is not None:
            P = np.asarray(P, dtype = float)
            emis = [k for k, p in enumerate(PARAM_NAMES) if p != 'lsf_factor']
            if ref is None:
                ref = P
            elif P.shape != ref.shape or not np.allclose(P[:, emis], ref[:, emis]):
                raise ValueError(f'{name} did not simulate the same skies as the first run (different grid or chemistry seed)')
        keep = np.ones(len(df), dtype = bool)
        if robust_only:
            keep &= df.robust.to_numpy(dtype = bool)
        keep &= df.intensity.to_numpy() > min_intensity
        t = df[keep].copy()
        t.insert(0, 'run', name)
        t.insert(1, 'line_index', np.flatnonzero(keep)) # row of the line in its own run's list
        tabs.append(t)
        Is.append(I[:, keep])
    T = pd.concat(tabs, ignore_index = True)
    I = np.hstack(Is)
    S = ratio_scatter(I) if metric == 'ratio' else linear_scatter(I)
    groups = covariation_groups(S, threshold)
    g, size, worst, nearest = _group_columns(S, groups, np.ones(len(T), dtype = bool), len(T))
    pre = 'covar' if metric == 'ratio' else 'linear'
    T[f'{pre}_group_all'], T[f'{pre}_n_all'], T[f'{pre}_scatter_all'], T[f'{pre}_nearest_all'] = g, size, worst, nearest
    return T, S

def save_covariation(path, lineDf, info, params = None):
    ''' Write what combine_covariation needs (the line list's own columns are in the .csv/.pkl) '''
    np.savez_compressed(path, wav = lineDf.wav.to_numpy(), I = info['I'].astype(np.float32), considered = info['considered'],
                        threshold = info['threshold'], params = np.zeros((0, len(PARAM_NAMES))) if params is None else np.asarray(params, dtype = float),
                        param_names = np.array(PARAM_NAMES))

def load_covariation(path, lineDf = None):
    '''
    Load a *_covar.npz written by ragnar(). The line list is read from the .pkl next to it unless lineDf is given.
    Returns (lineDf, I, params)
    '''
    d = np.load(path)
    if lineDf is None:
        lineDf = pd.read_pickle(path.replace('_covar.npz', '.pkl'))
    if not np.allclose(lineDf.wav.to_numpy(), d['wav']):
        raise ValueError(f'{path} does not belong to the given line list')
    P = d['params'] if len(d['params']) else None
    return lineDf, d['I'].astype(float), P

###########################################################
### Lines and sections that change linearly together ###
###########################################################

def linear_scatter(I):
    '''
    L[i, j] = fractional rms residual of the best straight line (with intercept) between the intensities of lines i and j over the spectra,
    in units of the mean intensity, the larger of the two directions: sqrt(1 - r_ij^2)*max(cv_i, cv_j), with r the correlation and cv the
    coefficient of variation. Lines with a fixed ratio have L ~ S; L can be small when the ratio changes but the two still move on one line.
    '''
    mu, sd = I.mean(axis = 0), I.std(axis = 0)
    Z = (I - mu)/np.where(sd > 0, sd, 1)
    r = np.clip(Z.T @ Z/len(I), -1, 1)
    cv = sd/np.where(mu > 0, mu, np.inf)
    L = np.sqrt(np.maximum(0, 1 - r**2))*np.maximum(cv[:, None], cv[None, :])
    L = (L + L.T)/2
    np.fill_diagonal(L, 0)
    return L

def bin_spectra(wav_range, specs, lo, hi, pix):
    ''' average fine-grid spectra (Nspec rows) into pixels of width pix on lo-hi; returns pixel centres and (Nspec, Npix) '''
    centres = np.arange(lo + pix/2, hi - pix/2, pix)
    e = np.searchsorted(wav_range, np.r_[centres - pix/2, centres[-1] + pix/2])
    S = np.asarray(specs)
    Y = np.add.reduceat(S[:, e[0]:e[-1]], e[:-1] - e[0], axis = 1)/np.diff(e)[None, :]
    return centres, Y

def _rank1_residual(G, norm):
    """ fractional residual of the best rank-1 fit, from the Gram matrix G of the block and the normalisation sum(mean^2)*Nspec """
    ev = np.linalg.eigvalsh(G)
    return np.sqrt(max(0.0, ev.sum() - ev[-1])/norm)

class _Blocks:
    """
    Features of a spectrum (pixels between minima of the mean spectrum, so each holds one peak with its surroundings) and what the
    one-pattern fit of a window of consecutive features needs.
    intercept = False: every spectrum is A(spectrum)*shape(lambda) in the window (the same up to a scalar multiplier)
    intercept = True:  every spectrum is mean(lambda) + a(spectrum)*pattern(lambda)
    """
    def __init__(self, wav, Y, intercept):
        Y = np.asarray(Y, dtype = float)
        self.ns, self.wav = len(Y), np.asarray(wav, dtype = float)
        self.m = Y.mean(axis = 0)
        self.X = Y - self.m if intercept else Y
        self.colsq = np.sum(self.X**2, axis = 0)
        mins = np.flatnonzero((self.m[1:-1] <= self.m[:-2]) & (self.m[1:-1] < self.m[2:])) + 1
        self.b = np.r_[0, mins, len(self.m)] # feature k is pixels b[k]:b[k+1]
        self.K = len(self.b) - 1
        self.fnorm = np.array([self.ns*np.sum(self.m[self.b[k]:self.b[k+1]]**2) for k in range(self.K)])
        self.fsum = np.array([np.sum(self.m[self.b[k]:self.b[k+1]]) for k in range(self.K)]) # mean line flux of each feature
        pix = np.median(np.diff(self.wav)) if len(self.wav) > 1 else 0.0
        self.edges = np.r_[self.wav[self.b[:-1]] - pix/2, self.wav[-1] + pix/2] # feature edges [nm]

    def gram(self, k):
        sl = slice(self.b[k], self.b[k+1])
        return self.X[:, sl] @ self.X[:, sl].T

    def shares(self, k, j):
        return self.fsum[k:j]/max(self.fsum[k:j].sum(), 1e-300)

    def residuals(self, k, j, G, min_frac = 0.0):
        """
        E_worst: for each feature, the rms of the fit residual over its pixels and all spectra, divided by the rms of its mean spectrum;
                 the largest of these among the features holding at least min_frac of the window's line flux (line wings and faint bumps
                 between lines are tiny features whose relative residual says nothing about the window)
        E_window: the same over the whole window (flux weighted, bright features dominate)
        """
        ev, V = np.linalg.eigh(G)
        win = slice(self.b[k], self.b[j])
        res = self.colsq[win] - (V[:, -1] @ self.X[:, win])**2
        rf = np.add.reduceat(res, self.b[k:j] - self.b[k])
        fn = self.fnorm[k:j]
        use = (fn > 0) & (self.shares(k, j) >= min_frac)
        worst = np.sqrt(np.max(np.where(use, np.maximum(rf, 0)/np.where(fn > 0, fn, 1), 0)))
        return float(worst), float(np.sqrt(max(0.0, ev.sum() - ev[-1])/max(fn.sum(), 1e-300)))

def linear_sections(wav, Y, threshold = 0.01, lookahead = 3, max_width = 10.0, local = True, intercept = False, min_feature_frac = 0.02):
    """
    Split a spectrum into contiguous sections in which every feature is the same in all simulated spectra up to one scalar multiplier per
    spectrum (intercept = False, the default): Y_s(lambda) = A_s*shape(lambda) to within `threshold`. With intercept = True the test is the
    looser mean + a_s*pattern. Section edges sit at local minima of the mean spectrum (features = one peak with its surroundings), so lines
    are not cut. The split uses the fewest sections; a feature that fails even on its own comes out as a single section (linear = False).
        local = True: the worst feature counts (E_worst, see _Blocks.residuals), so a bright line cannot carry faint ones that do not follow
        local = False: the whole section counts (E_window)
        min_feature_frac: features with less of the section's line flux than this do not count for E_worst
    Inputs:
        wav, Y: pixel centres [nm] and (Nspec, Npix) line-only spectra on them (the continuum is left out of the test)
        lookahead: features to keep adding after the threshold is first exceeded, in case the next ones bring the residual back down
        max_width: longest section to consider [nm]
    Returns a DataFrame of sections (lo, hi, i0, i1, n_feat, flux, E_worst, E_window, E_affine (whole window with an intercept), mode_frac,
    linear, flux_frac, edges) and the per-spectrum flux of each section (Nspec, Nsec), for comparing sections with each other.
    """
    B = _Blocks(wav, Y, intercept)
    ok = [dict() for _ in range(B.K)]
    for k in range(B.K):
        G = np.zeros((B.ns, B.ns)); bad = 0
        for j in range(k + 1, B.K + 1):
            G += B.gram(j - 1)
            worst, glob = B.residuals(k, j, G, min_feature_frac)
            E = worst if local else glob
            if E <= threshold or j == k + 1:
                ok[k][j] = E
            if E > threshold:
                bad += 1
                if bad > lookahead:
                    break
            else:
                bad = 0
            if B.wav[B.b[j] - 1] - B.wav[B.b[k]] > max_width:
                break
    best = np.full(B.K + 1, np.inf); prev = np.zeros(B.K + 1, dtype = int); best[0] = 0
    for j in range(1, B.K + 1):
        for k in range(j):
            if j in ok[k] and (ok[k][j] <= threshold or j == k + 1) and best[k] + 1 < best[j]:
                best[j], prev[j] = best[k] + 1, k
    cuts, j = [], B.K
    while j > 0:
        cuts.append((prev[j], j)); j = prev[j]
    Yr = np.asarray(Y, dtype = float); Xc = Yr - B.m
    rows, F = [], []
    for k, j in cuts[::-1]:
        i0, i1 = B.b[k], B.b[j]
        G = sum(B.gram(q) for q in range(k, j))
        worst, glob = B.residuals(k, j, G, min_feature_frac)
        Gc = Xc[:, i0:i1] @ Xc[:, i0:i1].T
        ev = np.linalg.eigvalsh(Gc)
        rows.append(dict(lo = B.wav[i0], hi = B.wav[i1 - 1], i0 = i0, i1 = i1, n_feat = j - k, n_feat_counted = int(np.sum(B.shares(k, j) >= min_feature_frac)),
                         flux = float(B.m[i0:i1].sum()),
                         E_worst = worst, E_window = glob, E_affine = _rank1_residual(Gc, B.fnorm[k:j].sum()),
                         mode_frac = float(ev[-1]/ev.sum()) if ev.sum() > 0 else 1.0, edges = np.round(B.edges[k:j+1], 4).tolist()))
        F.append(Yr[:, i0:i1].sum(axis = 1))
    T = pd.DataFrame(rows)
    T['linear'] = (T.E_worst if local else T.E_window) <= threshold
    T['flux_frac'] = T.flux/B.m.sum()
    return T, np.vstack(F).T

def linear_windows(wav, Y, min_width = 5.0, max_width = 10.0, intercept = False, min_feature_frac = 0.02):
    """
    Every contiguous window with edges at minima of the mean spectrum and a width in [min_width, max_width] nm, with how far its features
    are from being the same in all simulated spectra up to one scalar multiplier per spectrum (intercept = False; with True, mean + a*pattern):
        E_worst: the worst feature's rms residual (over its pixels and all spectra) relative to the rms of its own mean spectrum, among the
                 features with at least min_feature_frac of the window's line flux
        E_window: the same over the whole window (flux weighted)
    Returns a DataFrame sorted by E_worst, with the feature edges [nm] and flux shares of each window (to measure the same features in data).
    """
    B = _Blocks(wav, Y, intercept)
    rows = []
    for k in range(B.K):
        G = np.zeros((B.ns, B.ns))
        for j in range(k + 1, B.K + 1):
            G += B.gram(j - 1)
            width = B.wav[B.b[j] - 1] - B.wav[B.b[k]]
            if width > max_width:
                break
            if width < min_width:
                continue
            worst, glob = B.residuals(k, j, G, min_feature_frac)
            win = slice(B.b[k], B.b[j])
            sh = B.shares(k, j)
            rows.append(dict(lo = B.wav[B.b[k]], hi = B.wav[B.b[j] - 1], i0 = B.b[k], i1 = B.b[j], n_feat = j - k, n_feat_counted = int(np.sum(sh >= min_feature_frac)),
                             flux = float(B.m[win].sum()), E_worst = worst, E_window = glob, edges = np.round(B.edges[k:j+1], 4).tolist(),
                             feat_frac = np.round(sh, 4).tolist()))
    T = pd.DataFrame(rows)
    if len(T):
        T['width'] = T.hi - T.lo
        T['flux_frac'] = T.flux/B.m.sum()
        T = T.sort_values('E_worst', ignore_index = True)
    return T
