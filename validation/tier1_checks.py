'''
Tier 1 validation of the RAGNAR simulated spectra: internal checks that don't need observed data.

The expected values here are computed independently of the RAGNAR code paths they test (analytic profile integrals,
an independent Boltzmann fit, literature wavelengths, MODTRAN read directly), so a pass means more than the code agreeing with itself.

1. Flux bookkeeping: the integral of each simulated spectrum vs the analytic integral of every line profile it should contain
2. Rotational/vibrational temperature round trip: fit Trot and Tvib back out of simulated spectra
3. Wavelength convention: atomic line positions vs NIST air wavelengths converted to vacuum
4. Transmission: OH line peak ratios between PWV = 3 mm and 0 mm spectra vs MODTRAN read directly
5. Line width: fitted LSF width of isolated lines vs the input LSF scatter

Run from RAGNAR_new/:  python validation/tier1_checks.py
'''
import sys, os, pickle, time
import os.path as osp
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erf
from scipy.optimize import curve_fit

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(HERE))

from dev.make_spectra import get_spectra_and_peaks, get_relevant_lines, get_transmission_at_lines, load_transmission, make_intensity_ratios
from dev.species_intensities import get_pgopher_linelist, get_O2_at_T, get_atomic_lines, make_OH_intensities

BASE = '/tigress/cj1223/'

def spec_index(n_specs, pwv, tr, tv, k, a, rh = None):
    '''Index of a spectrum in the list saved by run_RAGNAR.py: loops PWV(2) x Trot(2) x Tvib(2) x r_hot scale(N) x LSF scatter(3) x chemistry(5).
    The r_hot dimension was added on 2026-10-07; N is inferred from the number of spectra (1 for older files). rh defaults to the middle scale.'''
    n_rh = n_specs//(2*2*2*3*5)
    rh = n_rh//2 if rh is None else rh
    return (((((pwv*2 + tr)*2 + tv)*n_rh + rh)*3 + k)*5 + a)
R_SCAT, WF = 0.0125, 3.4 # the line wing parameters get_spectra_and_peaks uses by default
HC_K = 1.4387769 # hc/k [cm K]
SPEC_R = sys.argv[1] if len(sys.argv) > 1 else '/tigress/cj1223/spectra/sweep_0_0.108_r_20261005.pkl' # saved r arm spectra, from run_RAGNAR.py -arm r -ls 3 -seed 0
ARMS = {'b': (400, 650, 0.091), 'r': (600, 1000, 0.108), 'n': (950, 1290, 0.122)}
report = []

def say(s = ''):
    print(s, flush = True)
    report.append(s)

def profile_integral(a, b, pos, sig, r = R_SCAT, gamma = None, half_width = None):
    '''Analytic integral from a to b of the RAGNAR line profile: a unit Gaussian core + exponential wing, normalized to unit flux.
    With half_width, the tapered version used to build spectra (the value at +-half_width subtracted, renormalized within the window), see utils.line_profile.
    Written out independently of the RAGNAR functions.'''
    gamma = WF*sig if gamma is None else gamma
    def cdf(x): # integral from -inf to x of the untapered, unit-flux profile
        d = x - pos
        core = 0.5*(1 + erf(d/(np.sqrt(2)*sig)))
        wing = np.where(d < 0, r*np.exp(d/gamma), 2*r - r*np.exp(-d/gamma))
        return (core + wing)/(1 + 2*r)
    if half_width is None:
        return cdf(b) - cdf(a)
    edge = (np.exp(-0.5*(half_width/sig)**2)/(np.sqrt(2*np.pi)*sig) + r/gamma*np.exp(-half_width/gamma))/(1 + 2*r)
    norm = cdf(pos + half_width) - cdf(pos - half_width) - 2*half_width*edge
    return (cdf(b) - cdf(a) - edge*(b - a))/norm

def all_lines(Tr, Tv, Trot_0, min_wav, max_wav):
    '''The line table exactly as get_spectra_and_peaks builds it for one temperature'''
    A = get_pgopher_linelist()
    A['Intensity'] = make_OH_intensities(A, Tr, Tv)
    A0 = pd.concat([A, get_O2_at_T(Tr), get_atomic_lines()])
    return A0[np.logical_and(A0['Position'] > min_wav, A0['Position'] < max_wav)]

############################
### 1. Flux bookkeeping ###
############################

def check_flux(arm):
    min_wav, max_wav, sig = ARMS[arm]
    wav = np.linspace(min_wav, max_wav, int(200*(max_wav - min_wav)/sig))
    Trots, Tvibs = [180, 210], [8000, 12000]
    Trot_0, Tvib_0 = np.mean(Trots), np.mean(Tvibs)
    dl = 15*sig

    np.random.seed(0)
    _, specs, _, _, _ = get_spectra_and_peaks(wav, Trots = Trots, Tvibs = Tvibs, linewidth_scatters = 1, chem_scatters = 1, LSF_err = 0.04, inclusion_limit = 1e-3,
                                             cont = False, intensity_cut = 0.0, line_spread = sig, resolution = np.inf, dlambda = 15, use_centroider = False,
                                             aerosols = ['background'], pwvs = [0.0], zds = [30], base = BASE, verbose = 0)

    # rebuild what should be in the spectra: the included lines, their transmission, and the species weights
    fid = all_lines(Trot_0, Tvib_0, Trot_0, min_wav, max_wav)
    include = np.array(get_relevant_lines(fid, delta_local = dl, sigma_line = sig, inclusion_limit = 1e-3))
    trans = load_transmission('background', 0.0, 30, basedir = BASE, min_wav = min_wav, max_wav = max_wav)
    ts = get_transmission_at_lines(fid[include], trans['nm'].to_numpy(), trans['combin'].to_numpy(), Trot = Trot_0)

    np.random.seed(0)
    i = 0
    for Tr in Trots:
        for Tv in Tvibs:
            w = make_intensity_ratios(0.5) # same random draw, in the same order, as inside get_spectra_and_peaks
            L = all_lines(Tr, Tv, Trot_0, min_wav, max_wav)[include]
            pos, I, sp = L['Position'].to_numpy(), L['Intensity'].to_numpy()*ts, L['Species'].to_numpy()
            weight = np.select([sp == 'OH', (sp == 'O2') & (pos < 1000), (sp == 'O2') & (pos > 1000), sp == 'OI(1S)', sp == 'OI(1D)', sp == 'NaI', sp == 'H'],
                               [w[0], w[1], w[2], w[3], w[4], w[5], w[4]], 0.0)
            na = np.nonzero(sp == 'NaI')[0] # the first Na line gets Tr/<Tr>, the rest <Tr>/Tr
            if len(na):
                weight[na[0]] *= Tr/Trot_0
                weight[na[1:]] *= Trot_0/Tr
            # the code only adds a line on grid points strictly inside (pos - dl, pos + dl), and the spectrum only covers the grid
            a = np.maximum(pos - dl, wav[0]); b = np.minimum(pos + dl, wav[-1])
            expected = np.sum(I*weight*profile_integral(a, b, pos, sig, half_width = dl))
            nominal = np.sum(I*weight) # every line should carry exactly its intensity, apart from lines cut by the ends of the grid
            got = np.trapezoid(specs[i], wav)
            say(f'  {arm} arm, Trot {Tr} K, Tvib {Tv} K: integral / analytic expectation = {got/expected:.6f}, integral / sum of line intensities = {got/nominal:.5f}')
            i += 1
    return

######################################
### 2. Temperature round trip ###
######################################

def isolated_OH_groups(min_wav, max_wav, sig, Trot_0, Tvib_0, max_Jup = 6.5, contamination_lim = 0.01, half_width = 3):
    '''Lambda-doublet groups of OH lines with less than contamination_lim of the flux in their +-half_width sigma window from other lines'''
    L = all_lines(Trot_0, Tvib_0, Trot_0, min_wav + 2, max_wav - 2)
    include = np.array(get_relevant_lines(L, delta_local = 15*sig, sigma_line = sig, inclusion_limit = 1e-3))
    L = L[include]
    OH = L[L['Species'] == 'OH'].copy()
    OH['key'] = OH["v'"].astype(int).astype(str) + '-' + OH['v"'].astype(int).astype(str) + OH['Branch'].str.replace(r'[ef]\(', '(', regex = True)
    groups = []
    allpos, allI = L['Position'].to_numpy(), L['Intensity'].to_numpy()
    for key, g in OH.groupby('key'):
        if g["J'"].max() > max_Jup or g['Position'].max() - g['Position'].min() > 0.3*sig:
            continue
        p0 = np.average(g['Position'], weights = g['Intensity'])
        lo, hi = p0 - half_width*sig, p0 + half_width*sig
        inside = profile_integral(lo, hi, allpos, sig)*allI
        own = np.isin(L.index.to_numpy(), g.index.to_numpy())
        if inside[~own].sum() < contamination_lim*inside[own].sum():
            groups.append(dict(key = key, pos = p0, ids = g.index.to_numpy(), vup = int(g["v'"].iloc[0]),
                               Eup = np.average(g['Eupper'], weights = g['Intensity']), gA = np.sum(2*(2*g["J'"] + 1)*g['A']), lo = lo, hi = hi))
    return pd.DataFrame(groups)

def boltzmann(groups, flux, Ev):
    '''Per upper vibrational level: fit ln(F/gA) = c_v - E_rot/(k Trot) with a common Trot, E_rot measured from the bottom of each vibrational level (Ev).
    Returns Trot and the intercepts c_v'''
    groups = groups.assign(Erot = groups['Eup'] - Ev[groups['vup']].to_numpy())
    vs = np.sort(groups['vup'].unique())
    X = np.column_stack([groups['Erot']] + [(groups['vup'] == v).astype(float) for v in vs])
    y = np.log(flux/groups['gA'])
    coef = np.linalg.lstsq(X, y, rcond = None)[0]
    return -HC_K/coef[0], dict(zip(vs, coef[1:]))

def model_vib_energies():
    A = get_pgopher_linelist()
    return A.groupby("v'")['Eupper'].min()

def check_temperatures():
    min_wav, max_wav, sig = ARMS['r']
    wav, specs = pickle.load(open(SPEC_R, 'rb'))
    Trots, Tvibs = [180, 210], [8000, 12000]
    Trot_0, Tvib_0 = np.mean(Trots), np.mean(Tvibs)
    G = isolated_OH_groups(min_wav, max_wav, sig, Trot_0, Tvib_0)
    trans = load_transmission('background', 0.0, 30, basedir = BASE, min_wav = min_wav, max_wav = max_wav)
    L = all_lines(Trot_0, Tvib_0, Trot_0, min_wav, max_wav)
    t_line = pd.Series(get_transmission_at_lines(L, trans['nm'].to_numpy(), trans['combin'].to_numpy(), Trot = Trot_0), index = L.index)
    G['t'] = [np.mean(t_line[ids]) for ids in G['ids']]
    G = G[G['t'] > 0.95].reset_index(drop = True) # stay out of absorption bands so this tests the emission model
    say(f'  {len(G)} isolated, unabsorbed low-J OH Lambda-doublets used, upper levels v\' = {sorted(G.vup.unique())}')
    Ev = model_vib_energies()
    frac = profile_integral(G['lo'].to_numpy(), G['hi'].to_numpy(), G['pos'].to_numpy(), sig, half_width = 15*sig)
    for itr, Tr in enumerate(Trots):
        for itv, Tv in enumerate(Tvibs):
            index = spec_index(len(specs), 0, itr, itv, 1, 2) # PWV 0, nominal r_hot, nominal LSF, middle chemistry draw
            s = specs[index]
            F = np.array([np.trapezoid(s[(wav > lo) & (wav < hi)], wav[(wav > lo) & (wav < hi)]) for lo, hi in zip(G['lo'], G['hi'])])/frac/G['t']
            # the same fit on the model's own line intensities, before they go into a spectrum
            A = get_pgopher_linelist(); A['Intensity'] = make_OH_intensities(A, Tr, Tv)
            Fm = np.array([A.loc[ids, 'Intensity'].sum() for ids in G['ids']])
            Tfit, c = boltzmann(G, F, Ev)
            Tfit_m, c_m = boltzmann(G, Fm, Ev)
            # vibrational temperature from the level populations N_v ~ exp(intercept) * Q_rot (Q_rot as an observer would compute it, with the fitted Trot)
            out = []
            for cc, T in [(c, Tfit), (c_m, Tfit_m)]:
                vs = np.array([v for v in cc if v < 8]) # v' = 8, 9 have chemical overpopulation factors in the model
                Qrot = []
                for v in vs:
                    Erot = np.unique(A.loc[A["v'"] == v, 'Eupper'] - Ev[v])
                    Qrot.append(np.sum(np.exp(-Erot*HC_K/T)))
                lnN = np.array([cc[v] for v in vs]) + np.log(Qrot)
                slope = np.polyfit(Ev[vs].to_numpy(), lnN, 1)[0]
                out.append(-HC_K/slope)
            say(f'  input Trot {Tr} K, Tvib {Tv} K | Trot from spectrum {Tfit:.2f} K, from model intensities {Tfit_m:.2f} K'
                f' | Tvib from spectrum {out[0]:.0f} K, from model intensities {out[1]:.0f} K')
    # diagnose the model itself: the same fit on line intensities with the hot population switched off should give back the input Trot exactly
    no_hot = lambda A, Tr, Tv: make_OH_intensities(A, Tr, Tv, r_hot_scale = 0.0)
    A = get_pgopher_linelist()
    say('  Model diagnostic, Trot fitted from line intensities (the hot population raises a single-temperature fit, so "model as is" is expected above the input):')
    for Tr in Trots:
        for Tv in Tvibs:
            row = []
            for name, f in [('model as is', make_OH_intensities), ('hot population off', no_hot)]:
                A['Intensity'] = f(A, Tr, Tv)
                row.append(f"{name} {boltzmann(G, np.array([A.loc[i, 'Intensity'].sum() for i in G['ids']]), Ev)[0]:.2f} K")
            say(f'    input {Tr} K / {Tv} K: ' + ', '.join(row))
    return G

##################################
### 3. Wavelength convention ###
##################################

def air_to_vac(lam_air_A):
    '''Morton (2000) / IAU standard air to vacuum conversion, wavelengths in Angstrom'''
    s2 = (1e4/lam_air_A)**2
    n = 1 + 8.34254e-5 + 2.406147e-2/(130 - s2) + 1.5998e-4/(38.9 - s2)
    return lam_air_A*n

def check_wavelengths():
    nist_air = {'OI(1S_0-1D_2)': 5577.339, 'NaI(D2)': 5889.9510, 'NaI(D1)': 5895.9242, 'OI(1D_2-3P_2)': 6300.304, 'OI(1D_2-3P_1)': 6363.776, 'H_alpha': 6562.79, 'H_beta': 4861.35}
    at = get_atomic_lines().set_index('Label')
    for lab, air in nist_air.items():
        vac = air_to_vac(air)/10
        say(f'  {lab:15s} RAGNAR {at.loc[lab, "Position"]:.4f} nm, NIST air {air/10:.4f} nm -> vacuum {vac:.4f} nm, difference {1e3*(at.loc[lab, "Position"] - vac):+.2f} pm (air would be {1e3*(at.loc[lab, "Position"] - air/10):+.0f} pm)')

#############################
### 4. Transmission ###
#############################

def modtran_direct(pwv, lines_nm, doppler_rel):
    '''Read the MODTRAN curve without RAGNAR and weight it with each line's Doppler Gaussian'''
    path = f'{BASE}/modtran_output_LBL_background/MaunaKea_PWV{pwv}_ZD30_highres.csv'
    d = np.loadtxt(path, skiprows = 7, delimiter = ',', usecols = (0, 1), comments = '}')
    nm, t = (1e7/d[:, 0])[::-1], d[::-1, 1]
    out = []
    for l in lines_nm:
        sl = slice(*np.searchsorted(nm, [l - 0.05, l + 0.05]))
        wgt = np.exp(-0.5*((nm[sl] - l)/(doppler_rel*l))**2)
        out.append(np.sum(wgt*t[sl])/np.sum(wgt))
    return np.array(out)

def check_transmission():
    min_wav, max_wav, sig = ARMS['r']
    wav, specs = pickle.load(open(SPEC_R, 'rb'))
    # every OH line strong enough to measure, isolated so its peak isn't shared
    G = isolated_OH_groups(min_wav, max_wav, sig, 195, 10000, max_Jup = 12.5, contamination_lim = 0.02, half_width = 2)
    idx0 = spec_index(len(specs), 0, 0, 0, 1, 2) # PWV 0
    idx3 = spec_index(len(specs), 1, 0, 0, 1, 2) # PWV 3, same temperatures, hot population, LSF and OH chemistry draw
    peak = lambda s, p: s[np.argmin(np.abs(wav - p))]
    ratio = np.array([peak(specs[idx3], p)/peak(specs[idx0], p) for p in G['pos']])
    doppler = np.sqrt(1.380649e-23*195/(2.9979e8**2*17*1.66054e-27))
    # evaluate MODTRAN at each Lambda-doublet component, not at the doublet's mean position: inside H2O bands the transmission changes between the two
    L = all_lines(195, 10000, 195, min_wav, max_wav)
    comp = np.concatenate(G['ids'].to_numpy())
    t0 = pd.Series(modtran_direct(0.0, L.loc[comp, 'Position'].to_numpy(), doppler), index = comp)
    t3 = pd.Series(modtran_direct(3.0, L.loc[comp, 'Position'].to_numpy(), doppler), index = comp)
    I = L['Intensity']
    exp_ratio = np.array([np.sum(I[ids]*t3[ids])/np.sum(I[ids]*t0[ids]) for ids in G['ids']])
    for name, m in [('clean, 820-890 nm', (G.pos > 820) & (G.pos < 890)), ('H2O band, 920-980 nm', (G.pos > 920) & (G.pos < 980)), ('all r arm', G.pos > 0)]:
        d = ratio[m] - exp_ratio[m]
        say(f'  {name:22s}: {m.sum():3d} lines, median t(3mm)/t(0mm) = {np.median(exp_ratio[m]):.3f} (MODTRAN direct), spectrum peak ratio - MODTRAN: median {np.median(d):+.4f}, max |.| {np.max(np.abs(d)):.4f}')

###########################
### 5. Line width ###
###########################

def check_widths():
    min_wav, max_wav, sig = ARMS['r']
    wav, specs = pickle.load(open(SPEC_R, 'rb'))
    G = isolated_OH_groups(min_wav, max_wav, sig, 195, 10000, max_Jup = 6.5, contamination_lim = 0.002, half_width = 4)
    G = G.sort_values('pos').reset_index(drop = True)
    expected = stats.norm(1, 0.04).ppf(np.linspace(0, 1, 5)[1:-1])
    L = all_lines(195, 10000, 195, min_wav, max_wav)
    def doublet_model(cp, cw):
        # the Lambda-doublet components are ~0.3 sigma apart, fitting them as one line would make the width look ~1% too large
        def model(x, A, dmu, s):
            return A*sum(w*(np.exp(-0.5*((x - c - dmu)/s)**2)/(np.sqrt(2*np.pi)*s) + R_SCAT/(WF*s)*np.exp(-np.abs(x - c - dmu)/(WF*s))) for c, w in zip(cp, cw))/(1 + 2*R_SCAT)
        return model
    for k in range(3):
        s = specs[spec_index(len(specs), 0, 0, 0, k, 2)]
        fits = []
        for p, ids in zip(G['pos'], G['ids']):
            comp = L.loc[ids]
            model = doublet_model(comp['Position'].to_numpy(), (comp['Intensity']/comp['Intensity'].sum()).to_numpy())
            m = (wav > p - 4*sig) & (wav < p + 4*sig)
            popt, _ = curve_fit(model, wav[m], s[m], p0 = [s[m].max()*sig*2.5, 0, sig])
            fits.append(popt[2]/sig)
        say(f'  LSF scatter {k}: input width factor {expected[k]:.4f}, fitted {", ".join(f"{f:.4f}" for f in fits)} ({len(fits)} isolated doublets)')

if __name__ == '__main__':
    t0 = time.time()
    say('# Tier 1 validation of RAGNAR simulated spectra')
    say(f'Run {time.strftime("%Y-%m-%d %H:%M")}')
    say('\n## 1. Flux bookkeeping')
    for arm in ARMS:
        check_flux(arm)
    say(f'\n## 2. Temperature round trip (r arm, saved spectra {osp.basename(SPEC_R)})')
    check_temperatures()
    say('\n## 3. Wavelength convention (atomic lines)')
    check_wavelengths()
    say('\n## 4. Transmission (r arm, PWV 3 mm vs 0 mm)')
    check_transmission()
    say('\n## 5. Line width (r arm)')
    check_widths()
    say(f'\n(total {time.time() - t0:.0f} s)')
    open(osp.join(HERE, 'tier1_raw_output.txt'), 'w').write('\n'.join(report) + '\n')
