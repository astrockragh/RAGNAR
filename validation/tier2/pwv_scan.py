'''Pick the MODTRAN PWV for each visit from the r-arm fit of the all-spectrograph stack (the H2O band at 920-960 nm constrains it)'''
import sys, json, numpy as np
from sky_model import ArmModel
visit = int(sys.argv[1])
d = np.load(f'sky_stack_{visit}.npz')
out = {}
for pwv in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
    M = ArmModel('r', d['wav'], d['sky'], d['err'], float(d['airmass']), pwv = pwv)
    r = M.fit()
    s = (M.wav > 920) & (M.wav < 950)
    out[pwv] = dict(chi2 = float(r['chi2']), chi2_h2o = float(np.sum(((M.sky[s] - M.model[s])/np.sqrt(M.err[s]**2 + (0.01*M.sky[s])**2))**2)), Trot = float(r['Trot']))
    print(visit, pwv, out[pwv], flush = True)
json.dump(out, open(f'pwv_scan_{visit}.json', 'w'), indent = 1)
