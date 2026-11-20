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
## Whether or not to use a centroider
parser.add_argument("-arm", "--arm", type = str, required = True) #defaults to None

args = parser.parse_args()
if args.cent == None or args.cent == 0:
    cent = 0
else:
    cent = 1
    
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

# lineDf, A0, peak_wavs, specs, continuum_specs, wav_range, transmission_curves, transmission_wavs\
#     = RAGNAR.ragnar(min_wav = min_wav, max_wav = max_wav, steps_per_line = 200, resolution = args.R, line_spread = args.sig, LSF_err = 0.05, linewidth_scatters = args.ls,\
#                     aerosols = ['background', 'volcanic'], pwvs = [0.0, 3.0], zds = [0, 50], save_name = name, use_centroider = cent, chem_scatters = 5)

lineDf, A0, peak_wavs, specs, continuum_specs, wav_range, transmission_curves, transmission_wavs\
    = RAGNAR.ragnar(min_wav = min_wav, max_wav = max_wav, steps_per_line = 200, resolution = None, line_spread = sig, LSF_err = 0.04, linewidth_scatters = args.ls,\
                    aerosols = ['background'], pwvs = [0.0, 3.0], zds = [30], save_name = name, use_centroider = cent, chem_scatters = 5, Tvibs = [8000, 12000], Trots = [180, 210], data_path = osp.expanduser('~/work/OHspec/make_linelists/data/') )


with open(osp.expanduser(f'~/../../tigress/cj1223/spectra/sweep_{cent}_{sig}_{args.arm}.pkl'), "wb") as handle:
    pickle.dump([wav_range, specs], handle)