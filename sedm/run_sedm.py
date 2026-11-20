'''
RAGNAR simulations for the SEDM (R = lambda/FWHM ~ 100, 340-1000 nm), 2026-11-16.

Questions from the SEDM team: a) can they wavelength calibrate on the sky lines, b) can they do relative flux (colour) calibration on them.
This runs RAGNAR in resolution mode and saves the line list (lineLists/) and the simulated spectra with their parameter table,
which analyze_sedm.py then uses.

Run from RAGNAR_new/: python sedm/run_sedm.py -R 100 -cent 1 -seed 0
'''
import os, sys, argparse, itertools
from datetime import datetime
import numpy as np
import os.path as osp

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(HERE))
import RAGNAR

ap = argparse.ArgumentParser()
ap.add_argument('-R', type = float, default = 100) # resolving power defined as lambda/FWHM, constant with wavelength
ap.add_argument('-cent', type = int, default = 1)
ap.add_argument('-seed', type = int, default = 0)
ap.add_argument('-spec_dir', type = str, default = '/tigress/cj1223/spectra/')
ap.add_argument('-tag', type = str, default = '')
# Mauna Kea curves to use (only Mauna Kea exists). The default '_wst' grid mimics Wendelstein (1.84 km, where SEDM is being reinstalled):
# the surface pressure there is ~1.33x Mauna Kea's, so MK at ZD 40/60 has Wendelstein's slant O2 and Rayleigh columns at ZD ~0/48,
# and MK PWV p corresponds to Wendelstein PWV 1.33p (2.7, 10.6 and 20 mm here)
ap.add_argument('-pwvs', type = float, nargs = '+', default = [2.0, 8.0, 15.0])
ap.add_argument('-zds', type = int, nargs = '+', default = [40, 60])
a = ap.parse_args()
np.random.seed(a.seed)

# ragnar() multiplies resolution by 2.82982 and then uses sigma = lambda/resolution, so pass R*FWHM/sigma/2.82982 to get lambda/FWHM = R
res_in = a.R*2*np.sqrt(2*np.log(2))/2.82982

# the MODTRAN curves start at 333.3 nm, and the range is a little wider than 340-1000 so the edges see their neighbours
min_wav, max_wav = 334, 1015

# the first runs (2026-11-16, no tag) used pwvs = [1, 5, 10] and zds = [0, 50] as a stand-in for Palomar
grid = dict(aerosols = ['background'], pwvs = a.pwvs, zds = a.zds, Trots = [180, 210], Tvibs = [8000, 12000], r_hot_scales = [0.75, 1.0, 1.33],
            linewidth_scatters = 3, chem_scatters = 5)

name = f'sedm_R{a.R:g}_{a.cent}{a.tag}'
lineDf, A0, peak_wavs, specs, continuum_specs, wav_range, transmission_curves, transmission_wavs\
    = RAGNAR.ragnar(min_wav = min_wav, max_wav = max_wav, steps_per_line = 100, resolution = res_in, LSF_err = 0.05, use_centroider = a.cent,
                    r_scat = 0.0, # no PFS-like scattering wing, the SEDM LSF is taken as Gaussian
                    save_name = name, continuum = False, verbose = 0, **grid)

# parameter table in the order the spectra are made in get_spectra_and_peaks: aerosol, pwv, zd, Trot, Tvib, r_hot_scale, LSF scatter, chemistry
params = np.array(list(itertools.product(range(len(grid['aerosols'])), grid['pwvs'], grid['zds'], grid['Trots'], grid['Tvibs'], grid['r_hot_scales'],
                                         range(grid['linewidth_scatters']), range(grid['chem_scatters']))), dtype = float)
assert len(params) == len(specs), (len(params), len(specs))

today = datetime.today().strftime('%Y%m%d')
spec_path = osp.join(a.spec_dir, f'{name}_{today}.npz')
np.savez(spec_path, wav_range = wav_range, specs = np.array(specs, dtype = np.float32), params = params,
         param_names = np.array(['aerosol', 'pwv', 'zd', 'Trot', 'Tvib', 'r_hot_scale', 'lsf_k', 'chem_k']), res_in = res_in, R = a.R)
print(f'{len(lineDf)} lines ({lineDf.robust.sum()} robust), spectra saved at {spec_path}')
