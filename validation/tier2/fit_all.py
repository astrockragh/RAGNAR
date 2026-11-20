'''
Fit the RAGNAR forward model (sky_model.ArmModel) to every per-spectrograph sky stack in every arm.
Usage: python fit_all.py VISIT ARM [--band] [--free-wing]
Pixels dominated by lines with line-specific (H2O/O2) absorption are masked (ArmModel mask_t = 0.97), which makes the fit insensitive to PWV,
so a nominal 2 mm is used; the transmission of the masked lines is tested separately (h2o_check.py). Writes fits/fit_{visit}_sm{sm}_{arm}[_band][_wing].npz with the model,
the per-component spectra and the fitted parameters.
'''
import sys, json, glob, os, os.path as osp, time
import numpy as np
from sky_model import ArmModel
visit, arm = int(sys.argv[1]), sys.argv[2]
mode = 'band' if '--band' in sys.argv else 'vup'
free_wing = '--free-wing' in sys.argv
pwv = 2.0
os.makedirs('fits', exist_ok = True)
for f in sorted(glob.glob(f'sky_stack_{visit}_sm*.npz')):
    sm = int(f.split('_sm')[1][0])
    d = np.load(f)
    t = time.time()
    M = ArmModel(arm, d['wav'], d['sky'], d['err'], float(d['airmass']), pwv = pwv, mode = mode, mask_t = 0.97)
    r = M.fit(free_wing = free_wing)
    tag = ('_band' if mode == 'band' else '') + ('_wing' if free_wing else '')
    np.savez(f'fits/fit_{visit}_sm{sm}_{arm}{tag}.npz', wav = M.wav, sky = M.sky, err = M.err, model = M.model, continuum = M.continuum, B = M.B,
             use = M.use, components = np.array(M.components), coef = M.coef[:len(M.components)], minor = np.array(M.minor_members), pwv = pwv,
             **{k: v for k, v in r.items() if k != 'amplitudes'})
    print(f'{visit} SM{sm} {arm}{tag}: PWV {pwv}, Trot {r["Trot"]:.1f} K, sigma0 {r["sig0"]:.4f}, slope {r["sig1"]:+.3f}, shift {1e3*r["shift"]:+.1f} pm, r_scat {r["r_scat"]:.4f}, chi2/pix {r["chi2"]/r["npix"]:.1f} ({time.time()-t:.0f} s)', flush = True)
