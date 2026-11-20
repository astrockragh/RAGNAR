'''
Median sky spectrum per visit from the PFS all-sky exposures, for comparison with RAGNAR.

Input: pfsMerged, where FLUX + SKY is the observed sky (the sky model is estimated from the fibres) and NORM is a smooth response,
so (FLUX + SKY)/NORM is the response-corrected sky in arbitrary units (a single scalar puts it in kR later).
Only fibres with targetType SKY and fiberStatus GOOD in the pfsConfig are used, and pixels with bad mask bits are dropped.

Run with the pfs-datamodel environment:  /scratch/gpfs/cj1223/pfs_env/bin/python stack_sky.py
Writes sky_stack_{visit}.npz (all spectrographs) and sky_stack_{visit}_sm{1-4}.npz (one spectrograph each) next to this script.
The n arm has persistence from earlier arcs and sky exposures that differs between spectrographs and visits, so the per-spectrograph stacks
are what the model is fitted to, and agreement between spectrographs is the check on persistence. DROP lists stacks to leave out.
'''
import glob, os.path as osp
import numpy as np
from astropy.io import fits
from pfs.datamodel import TargetType, FiberStatus

DATA = '/scratch/gpfs/cj1223/pfs_sky'
OUT = osp.dirname(osp.abspath(__file__))
VISITS = [129688, 129689, 129806, 129807]
DROP = {(129689, 3): 'n-arm offset of -9x the continuum with a strong wavelength trend, likely a bad persistence/dark correction'}
BAD_BITS = ['MP_BAD', 'MP_SAT', 'MP_CR', 'MP_EDGE', 'MP_NO_DATA', 'MP_SUSPECT', 'MP_BAD_FLAT', 'MP_BAD_FIBERTRACE', 'MP_BAD_SKY', 'MP_BAD_FIBERNORMS', 'MP_UNMASKEDNAN', 'MP_IPC']

for visit in VISITS:
    merged = fits.open(glob.glob(f'{DATA}/**/pfsMerged_PFS_{visit}_*.fits', recursive = True)[0])
    config = fits.open(glob.glob(f'{DATA}/**/pfsConfig_PFS_{visit}_*.fits', recursive = True)[0])
    cfg = config['CONFIG'].data
    good_ids = set(np.array(cfg['fiberId'])[(np.array(cfg['targetType']) == TargetType.SKY) & (np.array(cfg['fiberStatus']) == FiberStatus.GOOD)])

    wav = np.asarray(merged['WAVELENGTH'].data, dtype = float)
    fiber = merged['FIBERID'].data
    use = np.isin(fiber, list(good_ids))
    hdr = merged[0].header
    badmask = np.bitwise_or.reduce([1 << hdr[b] for b in BAD_BITS if b in hdr])

    sm_of = {} # spectrograph of each fibre, from the pfsArm files
    for sm in [1, 2, 3, 4]:
        for f in glob.glob(f'{DATA}/**/pfsArm_PFS_{visit}_[brn]{sm}_*.fits', recursive = True)[:1]:
            sm_of.update({int(x): sm for x in fits.open(f)['FIBERID'].data})
    fib_sm = np.array([sm_of.get(int(f), 0) for f in fiber[use]])

    sky = (np.asarray(merged['FLUX'].data[use], dtype = float) + np.asarray(merged['SKY'].data[use], dtype = float))/np.asarray(merged['NORM'].data[use], dtype = float)
    bad = (merged['MASK'].data[use] & badmask) != 0
    sky[bad | ~np.isfinite(sky)] = np.nan

    def stack(rows, name, min_fibres):
        n = np.sum(np.isfinite(sky[rows]), axis = 0)
        med = np.nanmedian(sky[rows], axis = 0)
        # robust scatter between fibres, and the error on the median from it
        q16, q84 = np.nanpercentile(sky[rows], [16, 84], axis = 0)
        err = 1.2533*(q84 - q16)/2/np.sqrt(np.maximum(n, 1))
        med[n < min_fibres] = np.nan
        np.savez(osp.join(OUT, f'{name}.npz'), wav = wav, sky = med, err = err, n = n, fiber_scatter = (q84 - q16)/2,
                 airmass = hdr.get('AIRMASS'), moon_el = hdr.get('MOON-EL'), moon_ill = hdr.get('MOON-ILL'), mjd = hdr.get('MJD'))
        ok = np.isfinite(med)
        print(f'{name}: {rows.sum()} good sky fibres, {ok.sum()} of {len(wav)} pixels with >= {min_fibres} fibres, median fibres per pixel {int(np.median(n[ok]))}')

    stack(np.ones(len(sky), dtype = bool), f'sky_stack_{visit}', 100)
    for sm in [1, 2, 3, 4]:
        if (visit, sm) in DROP:
            print(f'  dropping {visit} SM{sm}: {DROP[(visit, sm)]}')
            continue
        stack(fib_sm == sm, f'sky_stack_{visit}_sm{sm}', 50)
