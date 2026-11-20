from datetime import datetime
import os, pickle, argparse
import numpy as np
import os.path as osp

import RAGNAR

parser = argparse.ArgumentParser()

## how many line-width scatters to do
parser.add_argument("-ls", "--ls", type = int, required = False) #defaults to None
## Whether or not to use a centroider
parser.add_argument("-cent", "--cent", type = int, required = False) #defaults to None
## which PFS arm, b/r/n
parser.add_argument("-arm", "--arm", type = str, required = True) #defaults to None
## random seed for the chemistry draws (make_intensity_ratios uses np.random), None = unseeded like before
parser.add_argument("-seed", "--seed", type = int, required = False)
## where to dump the simulated spectra, empty string = don't save them
parser.add_argument("-spec_dir", "--spec_dir", type = str, default = '/tigress/cj1223/spectra/')
## appended to the output names, so a rerun with changed code doesn't overwrite earlier lists from the same day
parser.add_argument("-tag", "--tag", type = str, default = '')
## overrides for experiments; None keeps the arm's settings below
parser.add_argument("-sig", "--sig", type = float, required = False)
parser.add_argument("-rscat", "--rscat", type = float, required = False)
parser.add_argument("-wf", "--wf", type = float, required = False)
parser.add_argument("-wmin", "--wmin", type = float, required = False)
parser.add_argument("-wmax", "--wmax", type = float, required = False)

args = parser.parse_args()
if args.cent == None or args.cent == 0:
    cent = 0
else:
    cent = 1

if args.seed is not None:
    np.random.seed(args.seed)

wf = 3.4 # wing scale in units of sigma, the ragnar() default
lsf_err = 0.05 # LSF_err 0.04 -> 0.05 (2026-10-07): spans the ~6-7% LSF differences between spectrographs

if str(args.arm) == 'b':
    min_wav, max_wav = 400, 650
    sig = 0.088 # was 0.091; PFS all-sky pfsMerged fits give 0.085, ~0.088 on native pixels (validation/tier2, 2026-10)
    r_scat = 0.0125 # consistent with the PFS sky

if str(args.arm) == 'r':
    min_wav, max_wav = 600, 1000
    sig = 0.112 # was 0.108; pfsArm sky lines give 0.1095 intrinsic, 0.112 including the pixel, at r_scat = 0.0069 (validation/tier2, 2026-10)
    r_scat = 0.0125/1.8 # the PFS sky (2026-10 all-sky visits) wants ~0.0047; reduced by 1.8 so the model wing stays a bit wider than observed

if str(args.arm) == 'n':
    min_wav, max_wav = 950, 1290
    sig = 0.105 # was 0.122; pfsArm sky lines give 0.102 intrinsic, 0.1045 including the pixel, at r_scat = 0.0069 (validation/tier2, 2026-10)
    r_scat = 0.0125/1.8 # the PFS sky wants ~0.0030; reduced by 1.8 so the model wing stays a bit wider than observed

if str(args.arm) == 'm':
    # PFS medium resolution, red. pfsArm sky lines (6 windowed visits 150805-150810, 722-770 nm only, pfs_mr/, 2026-11-16): 0.0637 including the pixel
    # (FWHM of the stacked profile and the core fits agree; a Gaussian fitted over +-0.9 nm gives 0.0613, the profile is a bit flat-topped and leans red).
    # Spectrographs 0.0627-0.0645, fibres +-3-5%, m3 grows to 0.068 at high fiberId, so LSF_err 0.05 as for the LR arms.
    # Against the data (pfs_mr/test_linelist.py), 0.0637 beats 0.0613 (per-line centroid RMS 1.28 vs 1.35 pm) and the wing does not change the centroids.
    min_wav, max_wav = 681, 918 # the m arm (701-898 nm) +-10 nm, plus 10 nm so lines near the ends see their neighbours; trim to 691-908 nm
    sig = 0.0637
    r_scat = 0.006 # the stacked wing fit wants r ~ 0.004 with a broad scale (8 sigma); x1.5 so the model wing stays a bit wider than observed
    wf = 8.0

if args.sig is not None: sig = args.sig
if args.rscat is not None: r_scat = args.rscat
if args.wf is not None: wf = args.wf
if args.wmin is not None: min_wav = args.wmin
if args.wmax is not None: max_wav = args.wmax

name = f"sweep_{sig}_{cent}_{args.arm}{args.tag}"

# lineDf, A0, peak_wavs, specs, continuum_specs, wav_range, transmission_curves, transmission_wavs\
#     = RAGNAR.ragnar(min_wav = min_wav, max_wav = max_wav, steps_per_line = 200, resolution = args.R, line_spread = args.sig, LSF_err = 0.05, linewidth_scatters = args.ls,\
#                     aerosols = ['background', 'volcanic'], pwvs = [0.0, 3.0], zds = [0, 50], save_name = name, use_centroider = cent, chem_scatters = 5)

# data_path defaults to the data/ directory next to dev/, the old ~/work/OHspec/make_linelists/data/ no longer exists
lineDf, A0, peak_wavs, specs, continuum_specs, wav_range, transmission_curves, transmission_wavs\
    = RAGNAR.ragnar(min_wav = min_wav, max_wav = max_wav, steps_per_line = 200, resolution = None, line_spread = sig, LSF_err = lsf_err, linewidth_scatters = args.ls,
                    aerosols = ['background'], pwvs = [0.0, 3.0], zds = [30], save_name = name, use_centroider = cent, chem_scatters = 5, Tvibs = [8000, 12000], Trots = [180, 210], r_scat = r_scat, wf = wf,
                    r_hot_scales = [0.75, 1.0, 1.33]) # spans the night-to-night change of the hot fractions in the PFS sky and the PFS-vs-Noll+ 2020 difference

if args.spec_dir:
    # date the file so the existing sweep_{cent}_{sig}_{arm}.pkl spectra are never overwritten
    today = datetime.today().strftime("%Y%m%d")
    spec_path = osp.join(osp.expanduser(args.spec_dir), f'sweep_{cent}_{sig}_{args.arm}{args.tag}_{today}.pkl')
    if osp.exists(spec_path):
        spec_path = spec_path.replace('.pkl', f'_{datetime.today().strftime("%H%M%S")}.pkl')
    with open(spec_path, "wb") as handle:
        pickle.dump([wav_range, specs], handle)
    print(f'Saved the simulated spectra at {spec_path}')
