# Copy of run_RAGNAR.py with the two stale paths fixed (2026-09-20):
#   - data_path: ~/work/OHspec/... no longer exists -> absolute tigress path
#   - spectra dump kept on tigress
# Identical RAGNAR.ragnar call otherwise, so cent=1 is directly comparable to the
# cent=0 lists from 20240723.
import os, pickle, argparse
import numpy as np
import os.path as osp

import RAGNAR

parser = argparse.ArgumentParser()
parser.add_argument("-ls", "--ls", type = int, required = False)
parser.add_argument("-cent", "--cent", type = int, required = False)
parser.add_argument("-arm", "--arm", type = str, required = True)
args = parser.parse_args()

cent = 0 if (args.cent is None or args.cent == 0) else 1

if str(args.arm) == 'b':
    min_wav, max_wav = 400, 650
    sig = 0.091
if str(args.arm) == 'r':
    min_wav, max_wav = 600, 1000
    sig = 0.108
if str(args.arm) == 'n':
    min_wav, max_wav = 950, 1290
    sig = 0.122

name = f"sweep_{sig}_{cent}_{args.arm}"

DATA = '/tigress/cj1223/tiger2_home/work/OHspec/make_linelists/data/'

# make_spectra.py line 166 calls get_O2_at_T(Tr) without passing data_path, so it
# falls back to the stale default ~/work/OHspec/make_linelists/data/ which no
# longer exists. Patch the default here rather than editing the tracked package.
import functools
import dev.make_spectra as _ms
_ms.get_O2_at_T = functools.partial(_ms.get_O2_at_T, path = DATA)

lineDf, A0, peak_wavs, specs, continuum_specs, wav_range, transmission_curves, transmission_wavs\
    = RAGNAR.ragnar(min_wav = min_wav, max_wav = max_wav, steps_per_line = 200, resolution = None, line_spread = sig, LSF_err = 0.04, linewidth_scatters = args.ls,\
                    aerosols = ['background'], pwvs = [0.0, 3.0], zds = [30], save_name = name, use_centroider = cent, chem_scatters = 5, Tvibs = [8000, 12000], Trots = [180, 210], data_path = DATA)

with open(f'/tigress/cj1223/spectra/sweep_{cent}_{sig}_{args.arm}.pkl', "wb") as handle:
    pickle.dump([wav_range, specs], handle)
