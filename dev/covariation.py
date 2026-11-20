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
    return lineDf, {'I': I, 'S': S, 'considered': np.flatnonzero(considered), 'threshold': threshold}

def combine_covariation(runs, threshold = 0.01, robust_only = True, min_intensity = 0.0, names = None):
    '''
    Group lines across several runs that simulated the same skies (e.g. the arms of a spectrograph, run with the same grid and seed).
    runs: list of (lineDf with the add_covariation columns, I (Nspec, Nline), params (Nspec, len(PARAM_NAMES)) or None), or of paths to the
          *_covar.npz files written by ragnar()
    Every run must have the same number of spectra, and the emission parameters (everything but lsf_factor) must agree spectrum by spectrum.
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
    S = ratio_scatter(I)
    groups = covariation_groups(S, threshold)
    g, size, worst, nearest = _group_columns(S, groups, np.ones(len(T), dtype = bool), len(T))
    T['covar_group_all'], T['covar_n_all'], T['covar_scatter_all'], T['covar_nearest_all'] = g, size, worst, nearest
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
