from datetime import datetime
import os, pickle, argparse
import numpy as np
import os.path as osp

from RAGNAR import ragnar

parser = argparse.ArgumentParser()

## resolution as R = \frac{\lambda}{\sigma}
parser.add_argument("-R", "--R", type = int, required = False) #defaults to None
## Gaussian line spread 
parser.add_argument("-sig", "--sig", type = float, required = False) #defaults to None
## how many line-width scatters to do
parser.add_argument("-ls", "--ls", type = int, required = False) #defaults to None
## Whether or not to use a centroider
parser.add_argument("-cent", "--cent", type = int, required = False) #defaults to None
## Save path
parser.add_argument("-save", "--save", type = str, required = False) #defaults to None


args = parser.parse_args()

assert args.R != args.sig
assert np.logical_or(args.R != None, args.sig != None)

if args.cent == None or args.cent == 0:
    cent = 0
else:
    cent = 1

if args.R == None:
    name = f"sweep_{args.sig}_{cent}"
else:
    name = f"sweep_{args.R}_{cent}"

lineDf, A0, peak_wavs, specs, continuum_specs, wav_range, transmission_curves, transmission_wavs\
    = ragnar(min_wav = 400, max_wav = 1000, steps_per_line = 200, resolution = args.R, line_spread = args.sig, LSF_err = 0.02, linewidth_scatters = args.ls,\
                    aerosols = ['background', 'volcanic'], pwvs = [0.0, 3.0], zds = [0, 50], save_name = name, use_centroider = cent, chem_scatters = 5)

if args.save:
    with open(osp.expanduser(str(args.save)), "wb") as handle:
        pickle.dump([wav_range, specs], handle)