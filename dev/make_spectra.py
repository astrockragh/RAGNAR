import time
import numpy as np
import os.path as osp
import pandas as pd

from .utils import gauss, gauss_conv, get_blocks
from .species_intensities import get_pgopher_linelist, get_O2_at_T, get_atomic_lines, make_OH_intensities

from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter


def is_notebook():
    ''' Check if we're in a notebook or not to get the right tqdm version '''
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def get_spectra_and_peaks(wav_range, Trots = [180, 215], Tvibs = [8500, 13000], linewidth_scatters = 3, LSF_err = 0.05, inclusion_limit = 1e-3, chem_scatters = 3, use_centroider = True,\
                          intensity_cut = 5e-6, line_spread = 0.1, resolution = np.inf, dlambda = 15, aerosols = ['background', 'volcanic'],\
                          pwvs = [0.0, 2.0], zds = [0,30], cont = True, base = '~/../../tigress/cj1223/', data_path = 'data/', obs = 'MaunaKea', verbose = 1):
    
    '''
    Function to simulate spectra, and find the wavelength peaks of those spectra
    
    Inputs:
    
    wav_range: The wavelength grid on which to simulate spectrum [nm]
    
    Trots, Tvibs: arrays or lists of the rotational/vibrational temperatures to simulate [K]
    
    inclusion_limit: The limit for how little a line can contribute at its own line center for us to include it in calculations
    intensity_cut: Consider only peaks that are stronger than this as relevant lines
    line_spread: The Gaussian line width of lines, as measured IN THE INSTRUMENT [nm]
    resolution: R defined as sigma/lambda, where sigma is the line-width of the lines, as measured IN THE INSTRUMENT [nm]
    
    dlambda: How far to either side of the center of a given line to go to look for other lines that may impact that line
    chem_scatters: The number of samplings to take across the chemical dependencies 
    
    aerosols: list of aerosol models to loop over. 'background', 'volcanic' are possible
    pwvs: list precipitable water vapour to loop over [mm]. From 0.0, to 20.0, in increments of 1 are possible
    zds: list of zenith declination of telescope to loop over [degrees]. 0 to 80 in increments of 10 are possible
    base: The base path of the MODTRAN outputs. 
    data_path: The base path of the atomic/molecular parameter inputs
    obs: name of the observing site for which the transmission is loaded, see the google drive for possible sites

    
    verbose: Print Level, 0,1,2 available
    --------------------------
    Returns: list of lists of wavelength peaks, list of simulated spectra, list of transmission curves, list of transmission wavelength grids
    '''
    
    if resolution==np.inf: # if resolution wasn't set
        dlambda = dlambda*line_spread        
    else:
        dlambda = 2*dlambda*np.mean(wav_range)/resolution
    
    # find minimum and maximum for the spectrum
    
    min_wav = np.min(wav_range) 
    max_wav = np.max(wav_range)
        
    transmission_curves = []
    transmission_wavs = []
    specs = []
    cont_specs = []

    A = get_pgopher_linelist(path = data_path) # get the linelist for the OH
    if cont:
        if verbose == 2:
            print('Making continuum')
        continuum = make_continuum(wav_range, line_spread = line_spread, resolution = resolution) 
        
    Trot_0 = np.mean(Trots)
    Tvib_0 = np.mean(Tvibs)
    
    # loop over atmospheric transmission
    for aerosol in aerosols:
        for pwv in pwvs:
            for zd in zds:
            
                # Intensities should always be calculated before cutting any lines that could possibly exist in our system
                A['Intensity'] = make_OH_intensities(A, Trot_0, Tvib_0, ) #placeholder intensity, mainly for finding lines we may care about at all
                
                dfO2 = get_O2_at_T(Trot_0, path = data_path) #make fiducial O2 intensities
                dfAtomic = get_atomic_lines(path = data_path) #make fiducial OI, Na intensities                
                
                A0 = pd.concat([A, dfO2, dfAtomic]) # merge dataframes
                
                if verbose == 2:
                    print('All lines loaded and combined')
                    
                # load transmission curve
                if verbose != 0:
                    print(f'Loading transmission curve with {aerosol} aerosol model, {pwv:.2f} mm PWV, and {zd:.0f} degree telescope zenith declination angle')
                    start = time.time()
                    
                trans = load_transmission(aerosol, pwv, zd, obs = obs, basedir = base, min_wav = min_wav, max_wav = max_wav)
                
                if verbose != 0:
                    stop = time.time()
                    print(f'Transmission curve loaded in {stop-start:.1f} s.')
                
                # limit the lines to the wavelength range we care about
                A0 = A0[np.logical_and(A0['Position']>min_wav, A0['Position']<max_wav)] 
                
                if verbose == 2:
                    print(f'A total of {len(A0)} lines are being considered.')
                    
                if verbose != 0: 
                    print('Finding relevant lines to include.')
                    
                # find the includes that we may care about and save mask for later
                include = get_relevant_lines(A0, delta_local = dlambda, sigma_line = line_spread, resolution = resolution, inclusion_limit = inclusion_limit)
                
                if verbose == 2:
                    print(f'A total of {sum(include)} lines are relevant for the spectrum.')
                    
                if verbose!=0: 
                    print('Convolving transmission curve at line position')
                
                # Get transmission at line positions. Along with loading, this is the bottle_neck, mainly due to just having convolutions over insanely long arrays
                # I mask the array so that the convolution is fast but the masking of that array takes time
                ts = get_transmission_at_lines(A0[include], trans['nm'].to_numpy(), trans['combin'].to_numpy(), Trot = Trot_0)
                
                # save transmission for later
                transmission_curves.append(trans['combin'].to_numpy())
                transmission_wavs.append(trans['nm'].to_numpy())

                # loop over temperatures
                progress_bar = tqdm(total=len(Trots)*len(Tvibs)*linewidth_scatters, desc = 'Simulating Spectra')
                if verbose>0:
                    print('Simulating spectra')
                for Tr in Trots:
                    for Tv in Tvibs:
                        for k in range(linewidth_scatters):
                            if linewidth_scatters>1:
                                alpha_lines = np.linspace(0,1, linewidth_scatters+2)[1:-1] # this is kinda silly but also very fast so it doesn't matter
                                l = stats.norm(1, LSF_err).ppf(alpha_lines)[k] #better than drawing a random number, sample evenly
                                l = np.abs(l) # just to make sure
                            else:
                                l = 1
                            if verbose != 0:
                                if resolution == np.inf:
                                    print(f'Finding lines with rotational temperature {Tr:.1f} K, and vibrational temperature {Tv:.1f} K and linewidth {line_spread*l:.4f}')
                                else:
                                    print(f'Finding lines with rotational temperature {Tr:.1f} K, and vibrational temperature {Tv:.1f} K and resolution {resolution*l/2.82982:.1f}')

                            A['Intensity'] = make_OH_intensities(A, Tr, Tv) # intensity calculations needs to include all lines, so this needs to be done on the full linelist
                            dfO2 = get_O2_at_T(Tr) #make intensities for the O2 at Trot             
                
                            A0 = pd.concat([A, dfO2, dfAtomic]) # merge dataframes
                            
                            A0 = A0[np.logical_and(A0['Position']>min_wav, A0['Position']<max_wav)] # cut to only the lines in the wavelength range we care about
                            joint = A0

                            A0 = A0[include] # mask relevant lines

                            A0['Intensity'] =  A0['Intensity']*ts # get the actual observed intensity, so factor in transmission

                            spec_OH = np.zeros_like(wav_range) # initalize spectrum for OH
                            spec_O2_B = np.copy(spec_OH) # initalize spectrum for O2
                            spec_O2_a1 = np.copy(spec_OH) # initalize spectrum for O2
                            
                            spec_O_green = np.copy(spec_OH) # initalize spectrum for green O line
                            spec_O_red = np.copy(spec_OH) # initalize spectrum for red O system
                            spec_Na = np.copy(spec_OH) # initalize spectrum for Na doublet, need to vary line ratio and intensity
                            spec_H = np.copy(spec_OH) # initalize spectrum for H, need to vary line ratio and intensity

                            
                            k = 0
                            # loop over lines and add to spectrum
                            if verbose>1:
                                print('Species included')
                                print(np.unique(A0['Species']))
                            for species in np.unique(A0['Species']):
                                lines = A0[A0['Species']==species]
                                
                                for I, pos in zip(np.array(lines['Intensity']), np.array(lines['Position'])): 
                                    mask = np.logical_and(wav_range>pos - dlambda, wav_range<pos + dlambda)
                                    if np.sum(mask)>0:
                                        if resolution == np.inf:
                                            ls = line_spread*l
                                        else:
                                            ls = pos/resolution*l
                                        if species == 'OH':
                                            spec_OH[mask] += I*gauss(wav_range[mask], pos, ls)
                                        if species == 'O2' and pos < 1000:
                                            spec_O2_B[mask] += I*gauss(wav_range[mask], pos, ls)
                                        if species == 'O2' and pos > 1000:
                                            spec_O2_a1[mask] += I*gauss(wav_range[mask], pos, ls)
                                        if species == 'OI(1S)':
                                            spec_O_green[mask] += I*gauss(wav_range[mask], pos, ls)
                                        if species == 'OI(1D)':
                                            spec_O_red[mask] += I*gauss(wav_range[mask], pos, ls)
                                        if species == 'NaI':
                                            line_ratio = (Tr/np.mean(Trots)) # do line ratio variation, correlates with Trot
                                            if k == 0:
                                                Ir = line_ratio
                                                k+=1
                                            else:
                                                Ir = 1/line_ratio
                                            spec_Na[mask] += I*Ir*gauss(wav_range[mask], pos, ls)
                                        if species == 'H':
                                            spec_H[mask] += I*gauss(wav_range[mask], pos, ls)

                                            
                            
                            # vary the relative intensities of the species based on chemistry
                            alpha_species = np.linspace(0,1, chem_scatters+2)[1:-1] # this needs to be tuneable parameter
                            for alpha in alpha_species:
                                I_ratio = make_intensity_ratios(alpha)
                                spec_all = spec_OH*I_ratio[0]+spec_O2_B*I_ratio[1]+spec_O2_a1*I_ratio[2]+spec_O_green*I_ratio[3]+spec_O_red*I_ratio[4]+spec_Na*I_ratio[5]+spec_H*I_ratio[4]
                                if cont:
                                    spec_cont = spec_all + continuum
                                    cont_specs.append(spec_cont)
                                specs.append(spec_all) # save simulated spectrum for later
                            progress_bar.update(1)
    if resolution != None:
        line_s = (max_wav+min_wav)/2/resolution
    else:
        line_s = line_spread

    peak_wavs = get_peaks(specs, wav_range, line_spread=line_s, intensity_cut = intensity_cut, use_centroider = use_centroider, verbose = verbose)
    if verbose == 2:
        print("Spectra simulated. Returning a list of list of peak wavelengths, a list of simulated spectra, a list of simulated spectra with continuum, a list of transmission curves and a list of corresponding wavelength curves")
    return peak_wavs, specs, cont_specs, transmission_curves, transmission_wavs

def make_intensity_ratios(alpha):
    '''
    A function to make intensity ratios so that lines follow the variations described in the paper
    
    Inputs:
    alpha: [0,1], where to sample from in the cdf
    -------------------
    Returns: The relative intensities of the different species
    '''
    
    X_O = stats.norm(1, 0.1).ppf(alpha) # draw the relative atomic oxygen concentration
    X_O2 = np.abs(stats.norm(1, 0.35).ppf(alpha)) # draw the relative molecular oxygen concentration
    X_O3 = stats.norm(1, 0.15).ppf(1-alpha) # draw the relative ozone concentration 
    # draw the relative intensity of the red atomic oxygen system, this also impacts the a1 Delta O2 system, since this is essentially a measure of if there is a lot of scattered sunlight
    X_red = stats.lognorm(1, loc = 0.2, scale = 0.9).ppf((1-alpha)/2+np.random.uniform()/2)
                    #OH,     O_2_B,       O_2_a1,   green OI, red OI,    NaD
    return np.array([X_O3, X_O**2*X_O2, X_O**2*X_red, X_O**3, X_red, X_O*X_O3])

def get_relevant_lines(lineDf, delta_local = None, sigma_line = 0.1, resolution = np.inf, inclusion_limit = 3e-4):
    '''
    Given a DataFrame of lines, find which lines contribute significantly.
    We consider only lines which, given a Gaussian line spread (sigma_line), contributes more than a given fraction (1/inclusion limit) of the intensity at its own line center
    delta_local defines the range within which to find other lines that could contribute significantly to the intensity
    
    Inputs:
    
    lineDf: DataFrame with lines to check
    delta_local: the wavelength delta to check for lines within on either side of the line position [nm]
    sigma_line: the Gaussian width of each line, in the instrument
    inclusion_limit: If a line contributes a fraction of less than this limit at its own line center, get rid of it, it won't matter anyway
    
    --------------------------
    Returns: Boolean mask for which lines to consider
    '''
    
    include = []
    
    if delta_local == None: # if undefined
        if resolution == np.inf: # if the resolution is not set
            delta_local = 20*sigma_line # check within 20 sigma on both sides
        else: # if the resolution is set
            delta_local = 30*np.mean(lineDf['Position'])*resolution
            
    #make arrays to facilitate i-indexing
    wav_array = np.array(lineDf['Position']) # line positions
    I_array = np.array(lineDf['Intensity']) # line intensities
    
    for i in range(len(lineDf['Position'])):
        
        w0 = wav_array[i] # wavelength of the line being considered
        I0 = I_array[i] # intensity of the line being considered
 
        mask = np.abs(lineDf['Position']-w0) < delta_local # get the lines that are close enough to maybe influence things
        local_lines = lineDf[mask] 
        
        individual_lines = [] # list of line intensities at the line 
        for pos, I in zip(local_lines['Position'], local_lines['Intensity']):
            if resolution == np.inf:
                line = I*gauss(w0, pos, sigma_line) # get contribution from local lines at w0 which is the wavelength in question
            else:
                line = I*gauss(w0, pos, w0*resolution) # get contribution from local lines at w0 which is the wavelength in question
            individual_lines.append(line)
        
        if resolution == np.inf: # if the resolution is not set
            self = I0/(sigma_line*np.sqrt(2*np.pi)) # line center intensity for line, recomputed so that I don't have to do any weird indexing
        else:
            self = I0/(resolution*w0*np.sqrt(2*np.pi)) # line center intensity for line, recomputed so that I don't have to do any weird indexing
            
        if self < np.sum(individual_lines)*inclusion_limit: # if a given line contributes less than the inclusion limit at its own line center, discard it
            include.append(False)
        else:
            include.append(True)
            
    return include

###############################
### Continuum related stuff ###
###############################

def make_continuum(wav_range, line_spread = 0.1, resolution = np.inf, I0 = 80):
    '''
    Given a wavelength range, make a I0 R/nm continuum, with a slight positive slope (calibrated to fit PFS/SuNSS) and a gaussian peak centered at 595 nm, with width 10 nm
    
    Inputs:
    wav_range: wavelength grid to make continuum on
    line_spread: the line spread of the Gaussian LSF
    I0: The intensity of the continuum, defined such that it's about 8 R/nm at a line spread of 0.1 nm
    --------------------------
    Returns the continuum strength in kR
    '''
    if resolution == np.inf:
        
        I0 = I0*line_spread
        continuum = np.ones_like(wav_range)*I0*(wav_range/500)**(0.2)
        continuum += gauss(wav_range, 595, 10)*I0*(np.sqrt(2*np.pi)*10)
        
    else:
        continuum = np.ones_like(wav_range)*I0*(wav_range/resolution)*(wav_range/500)**(0.2)
        continuum += gauss(wav_range, 595, 10)*I0*(595/resolution)*(np.sqrt(2*np.pi)*10)
    
    return continuum/1000 # cast it into kR

def make_continuum_at_lines(wav_range, line_spread, line_wavelengths, resolution = np.inf):
    '''
    Given a wavelength range, and some line wavelengths, get the continuum strength at the lines
    
    Inputs:
    wav_range: wavelength grid to make continuum on
    line_spread: the line spread of the Gaussian LSF
    line_wavelengths: the wavelengths of the lines
    --------------------------
    Returns the continuum strength in kR at the line centers
    '''
        
    continuum = make_continuum(wav_range, line_spread = line_spread, resolution = resolution)    
    cont_at_lines = []
    
    if resolution == np.inf:
        dlambda = wav_range[1]-wav_range[0]
        for l in line_wavelengths:
            cont_line = continuum[np.isclose(l, wav_range, atol = dlambda, rtol = 0)]
            cont_at_lines.append(np.mean(cont_line))
    else:
        for l in line_wavelengths:
            cont_line = continuum[np.isclose(l, wav_range, atol = 0, rtol = 1/resolution)]
            cont_at_lines.append(np.mean(cont_line))
    return cont_at_lines

####################
### Peak finders ###
####################

def get_peaks(specs, wav_range, line_spread = 0.1, intensity_cut = 1e-5, use_centroider = True, verbose = 1):
    
    '''Given a list of spectra (specs) simulated over a wavelength range (wav_range), find the peaks. 
    The intensity_cut is an inclusion limit, so everything below this is not included in the further calculations. Set this lower than you think appropriate and then refilter later
    --------------------------
    Returns: A list of lists of wavelength peaks
    '''
    if verbose > 1:
        print('Finding peaks')
    peak_wavs = []
    prominences = []
    
    for spec in specs:
        if use_centroider:
            spec_conv = gaussian_filter(spec, line_spread/(wav_range[1]-wav_range[0])) #centroiding
        else:
            spec_conv = spec
            
        peak_indices, pro_dict = find_peaks(spec_conv, prominence = 0) # find the indices of the peaks of the spectrum
        intensities = spec[peak_indices] # get the intensities
#         prominences = pro_dict['prominence'] # get the prominences of the peaks

        #  This isn't super reliable, so set the intensity_cut lower than you think might be good and then refilter the linelist later
        # It does make things faster though
        Imask = intensities >= intensity_cut
        peak_indices = peak_indices[Imask] # do a cut on the intensities.

        peak_wav = wav_range[peak_indices] # get the wavelengths

        peak_wavs.append(peak_wav)
        if verbose == 1:
            print(f'Found {len(peak_wav)} peaks that fulfill the criteria')
    return peak_wavs

def get_common_lines(wavs, line_spread = 0.1, resolution = np.inf, verbose = 1):
    '''
    Function for finding the lines that seem to be close enough to overlapping that we can robustly identify them as the same line in all spectra
    The absolute tolerance should be set to be roughly equal to (or a bit below) the used Gaussian line spread if a single line spread is used.
    Otherwise, set the relative tolerance to be equal to a little less than 1/resolution used if resolution is used. This makes everything wavelength dependent

    Inputs:
    wavs: list of lists of possible lines
    atol: absolute tolerance of a given line being the same [nm]

    resolution: NOT IMPLEMENTED. Should be resolution of spectrograph

    verbose: Print the number of lines found to overlapping with the other line lists and how many lines are found to be overlapping
    ------------------------
    Returns a list of lists of the wavelengths of the stable lines
    '''
    
    rtol = 1/(resolution*np.sqrt(2)) # set the relative tolerance, this will be wavelength dependent
    
    if rtol>0:
        line_spread = 0
    
    l = len(wavs)
    indices = np.arange(l)
    
    if verbose != 0:
        print('Finding the identical, unvarying lines between runs')
    all_masked = []
    uncertain_wavs = []
    for i in range(l):
        masks = []
        for index in np.delete(indices, i):
            mask = np.isin(wavs[i], wavs[index])
            masks.append(mask)
        m = np.all(masks, axis = 0)
        all_masked.append(wavs[i][m])
        uncertain_wavs.append(wavs[i][~m])

    if verbose == 2:
        print('Number of shared CERTAIN lines found')
        print(np.array([len(a) for a in all_masked]))
        
    maskss = []
    arg = np.argmin(np.array([len(l) for l in uncertain_wavs])) #find the shortest uncertain wavelengths
    
    if verbose != 0:
        print('Finding which uncertain lines are close to each other')    
    
    for u_wavs in uncertain_wavs:
        masks = []
        for wav in uncertain_wavs[arg]: # only compare against the shortest
            mask = np.isclose(u_wavs, wav, atol = line_spread/2, rtol = rtol) # look if there are lines that are close and could be the same peak
            masks.append(np.any(mask))
        maskss.append(masks)

    if verbose == 2:
        print('Number of shared UNCERTAIN lines found')
        print(np.array([sum(m) for m in maskss]))
        
    shared_uncertain = uncertain_wavs[arg][np.all(maskss, axis = 0)]
    
    good_wavs = []
    
    numbers = [] # list of the number of overlaps that a given line can be found with
    nlines = []
    
    if verbose != 0:
        print('Finding the variable lines that are still identifiable across all runs')  
        
    for u_wavs in uncertain_wavs:
        good_wav = []
        masks = []
        for wav in shared_uncertain:
            mask = np.isclose(u_wavs, wav, atol = line_spread/2, rtol = rtol) #if the line position if away by more than line_spread it's not the same!
            masks.append(mask) 
            good_wav.append(u_wavs[mask])

        good_wavs.append(np.hstack(good_wav))
        ns, nl = np.unique([sum(m) for m in masks], return_counts = True)
        numbers.append(ns)
        nlines.append(nl)
        
    ls = np.array([len(n) for n in numbers]) # should be length 1 everytime
    if verbose == 2:
        print('Checking if lines are degenerate')
        print(ls>1)
        print("Checking how degenerate the lines are. If any number here is not 1, it means that there isn't a 1 to 1 mapping between runs")
        print(numbers)
        print(nlines)
    if np.any(ls>1):
        print('WARNING: The LSF uncertainty (LSF_err) is so high that only the very most stable lines can be robustly identified. Reducing the search area for overlaps and trying again.')
        return (False, all_masked, shared_uncertain, uncertain_wavs)
    else:
        all_good_wavs = [np.sort(np.hstack([g, u])) for g, u in zip(good_wavs, all_masked)]
        if verbose != 0:
            print(f'{len(all_good_wavs[0])} common lines found')
        return (True, all_good_wavs, shared_uncertain, uncertain_wavs)
    
def get_uncertain_lines(uncertain_wavs, shared_uncertain, line_spread = 0.1, resolution = np.inf, verbose = 1):
    '''A routine to take the lines that aren't matched across all runs, and match them
        Takes the uncertain wavelengths, and the shared uncertain lines from the other peak matching algorithm as inputs as well as the LSF parameters

        returns the mean and sigma of the peak wavelengths and the number of simulations where the peak is identified
    '''
   
    rtol = 1/(resolution*np.sqrt(2))
    if verbose > 0:
        print('Finding the rejected lines')
    
    bad_ind = [] ## here we get the individual bad lines that are not shared between the lines
    for uwavs in tqdm(uncertain_wavs, desc = 'Treating Uncertain Peaks'):
        mask_u = []
        for w in shared_uncertain:
            mask = np.where(np.isclose(w, uwavs, atol = line_spread/2, rtol = rtol), True, False) # which lines are close

            mask_u.append(mask)
        bad_ind.append(uwavs[~np.sum(np.vstack(mask_u), axis = 0).astype(bool)])
        
    u_bad, counts_bad = np.unique(np.hstack(bad_ind), return_counts = True) # get the unique bad lines and how often we see them
    
    masks = [] # a mask to see which ones of the unique peaks that are close
    for b0 in u_bad:
        mask = []
        for b1 in u_bad:
            mask.append(np.isclose(b0, b1, atol = line_spread/2, rtol = rtol))
        masks.append(mask)
    blocks = get_blocks(masks) # link the lines that probably correspond to a single peak (a block)

    mu_wav_unstable = []
    sig_wav_unstable = []
    counts = []

    for i in range(len(blocks)-1): #now we analyze everything block by block
        lines = []
        counts.append(np.sum(counts_bad[blocks[i]:blocks[i+1]])) # the total number of runs where we find the line 
        for j in range(blocks[i], blocks[i+1]):
            lines.append([u_bad[j]]*counts_bad[j])
        lines = np.hstack(lines)
        mu = np.mean(lines)
        std = np.std(lines)
        mu_wav_unstable.append(mu)
        sig_wav_unstable.append(std)
    mu_wav_unstable = np.array(mu_wav_unstable)
    sig_wav_unstable = np.array(sig_wav_unstable)
    counts = np.array(counts)
    
    return mu_wav_unstable, sig_wav_unstable, counts

def define_lines_and_intensities(peak_wavs, specs, wav_range, line_spread = 0.1, resolution = np.inf, verbose = 1):
    
    '''Given a list of lists of found peak wavelengths, list of simulated spectra and the wavelength for the spectrum grid, find the mean and variance of the peak wavelengths and intensities.
        The line_spread should be set as to define the tolerance within which lines should be found.
        The only lines that are returned are the ones that can be identified in all of the spectra.
        If one wants to set a resolution, that could be done as well but it is currently not implemented, so do not change it from inf
        
        Inputs:
        peak_wavs: list of lists of peak wavelengths
        specs: list of simulated spectra
        wav_range: wavelength grid for spectra
        line_spread: the Gaussian sigma of the lines, as measured IN THE INSTRUMENT
        resolution: Resolution of spectrograph
        verbose: says itself
        ------------------------
        Returns: means and sigmas of of the location and intensities of each found peak
    '''
    
    check = False
    k = 0
    while check == False and k < 15:
        check, good_wavs, shared_uncertain, uncertain_wavs = get_common_lines(peak_wavs, line_spread = line_spread*(1-k*0.05), resolution = resolution*(1+k*0.05), verbose = verbose) # do a search and if it fails reduce the search are by 5 %
        k += 1
    if check == False and verbose != 0:
        print('Limit for reducing search area reached (at 75% of original line spread), only the most stable and isolated lines are carried forward.')
        print('Please reduce the LSF_err')
        print(f'{len(good_wavs[0])} stable lines found')
    if len(shared_uncertain)-max([len(uw) for uw in uncertain_wavs])!=0:
        mu_unstable_wav, var_unstable_wav, unstable_counts = get_uncertain_lines(uncertain_wavs, shared_uncertain, line_spread*(1-(k-1)*0.05), resolution = resolution*(1+(k-1)*0.05), verbose = verbose )
    # get the mean and sigma of the wavelength of the good lines
    mu_stable_wav = np.mean(np.vstack(good_wavs), axis = 0)
    var_stable_wav = np.std(np.vstack(good_wavs), axis = 0)
    
    dwav = (wav_range[1] - wav_range[0])/2 # resolution of peak finder
    var_stable_wav = np.sqrt(var_stable_wav**2+dwav**2)
    if len(shared_uncertain)-max([len(uw) for uw in uncertain_wavs])!=0:
        var_unstable_wav = np.sqrt(var_unstable_wav**2+dwav**2)

    # get the mean and sigma of the spectra
    stack_spec = np.vstack(specs)
    mean_spec = np.mean(stack_spec, axis=0)
    std_spec = np.std(stack_spec, axis=0)

    # get the mean and sigma of the spectra at the line centers of the stable lines
    I_mu = []
    I_std = []
    for wav in mu_stable_wav:
        index = np.argmin(np.abs(wav_range-wav)) # line center
        # mean and sigma of the intensities
        I_mu.append(mean_spec[index])
        I_std.append(std_spec[index])
    mu_stable_I = np.array(I_mu)
    var_stable_I = np.array(I_std)
    
    # get the mean and sigma of the spectra at the line centers of the unstable lines
    if len(shared_uncertain)-max([len(uw) for uw in uncertain_wavs])!=0:
        I_mu = []
        I_std = []
        for wav in mu_unstable_wav:
            index = np.argmin(np.abs(wav_range-wav)) # line center
            # mean and sigma of the intensities
            I_mu.append(mean_spec[index])
            I_std.append(std_spec[index])
        mu_unstable_I = np.array(I_mu)
        var_unstable_I = np.array(I_std)
    
    stable_counts = np.ones_like(mu_stable_wav)*len(peak_wavs)
    if len(shared_uncertain)-max([len(uw) for uw in uncertain_wavs])!=0:
        return mu_stable_wav, var_stable_wav, mu_stable_I, var_stable_I, stable_counts, mu_unstable_wav, var_unstable_wav, mu_unstable_I, var_unstable_I, unstable_counts
    else:
        return mu_stable_wav, var_stable_wav, mu_stable_I, var_stable_I, stable_counts, False, False, False, False, False
        
##################################
### Transmission related stuff ###
##################################

def load_transmission(aerosol, pwv, zd, basedir, obs = 'MaunaKea', min_wav = 360, max_wav = 1290):
    '''Function to load a Line-By-Line (LBL) MODTRAN6 transmission curve DataFrame with a given aerosol model, PWV (in mm) and ZD. 
    aerosol model for Mauna Kea should be either 'volcanic' or 'background'
    Only returns transmission curves in |min_wav, max_wav| interval 

    Inputs:
    aerosol: Aerosol model (str)
    pwv: Precipitable water vapour [mm], (float)
    zd: zenith declination of telescope, 0 is straight up (int)
    basedir: The base path of the MODTRAN outputs. Can be tigress/c1223/ or scratch/gpfs/cj1223/

    min_wav, max_wav = minimum wand maximum wavelength to consider
    --------------------------
    Returns: DataFrame with transmission curve
    '''
    
    path = osp.expanduser(basedir+f'{obs}_aerosol_{aerosol}_PWV{pwv}_ZD{zd}_highres.csv' ) #this is a 200 MB file, if Tigress is really slow, I also have them on scratch/gpfs

    trans = pd.read_csv(path, skiprows = 7, skipfooter = 1, engine='python') # read the csv
    trans.columns = ['Freq', 'combin', 'total', 'path', 'surface'] # define column names without weird spaces. Could consider dropping the last three columns since they are not used

    trans['nm'] = 1e7/trans['Freq'] # from cm^-1 to nm

    trans = trans[np.logical_and(trans['nm']>min_wav, trans['nm']<max_wav)] # limit the wavelength range
    
    return trans

def make_Doppler_line_widths(Trot = 195):
    '''
    Calculating the (thermal) Gaussian line broadening of each line. $\sigma = \sqrt{\frac{k*T}{m*c^2}}$
    Should be around 1e-3 nm at 200 K for OH. This is NOT the line width as measured by your instrument.
    
    Input: Temperature at the OH layer in K. 
    --------------------------
    Returns: A dictionary of $\sigma_{\lambda}/\lambda$ where the keys are the species, so multiply with the wavelength to get the actual linewidth later
    '''
    
    c = 2.9979e8       #m/s, speed of light
    kB = 1.380649e-23  #J/K, Boltzmann constant
    weight_conversion = 1.66054e-27 # conversion from amu to kg
                               #OH, O2, OS, OD, Na, H
    species_weights = np.array([17, 32, 16, 16, 23, 1]) # approximate weights, since it's in the square-root the accuracy doesn't need to be too great
    species_Ts = np.array([Trot, 160+35*(Trot/190)**4, Trot, 1050, Trot, 1e4])
    fr = np.sqrt(kB*species_Ts/(c**2*species_weights*weight_conversion))
    sigmas = {'OH': fr[0],'O2': fr[1],'OI(1S)': fr[2],'OI(1D)': fr[3],'NaI': np.sqrt(fr[4]**2+4.3e-6**2), 'H': fr[5]}
    return sigmas

def get_transmission_at_lines(lineDf, transmission_wavelength, transmission, Trot = 195):

    '''
    Given a DataFrame with line position, and a transmission curve (along with corresponding wavelengths), calculates transmission at line
    Assumes a Gaussian line profile, with sigma = intrinsic_width (optional argument) and convolves the transmission with the line profile
    
    Inputs:
    
    lineDf: DataFrame with lines.
    transmission_wavelength: The wavelength grid of the transmission curve.
    transmission: The actual transmission curve.
    Trot: For calculating the (thermal) Gaussian line broadening of each line.
    --------------------------
    Returns: Transmission at the line positions (NOT absorption)
    '''
    
    assert max(lineDf['Position'])<max(transmission_wavelength) and min(lineDf['Position'])>min(transmission_wavelength), \
    f'The wavelength range of the transmission curve is too narrow for the linelist. \n It is {min(lineDf["Position"]):.2f}-{max(lineDf["Position"]):.2f} for the linelist and {min(transmission_wavelength):.2f}-{max(transmission_wavelength):.2f} for the transmission curve'
    ts = []
    species_relative_width = make_Doppler_line_widths(Trot) # make the thermal Doppler widths at the relevant temperature
    for pos, species in tqdm(zip(lineDf['Position'], lineDf['Species']), total = len(lineDf), desc="Finding Line Transmission"):
        intrinsic_width = species_relative_width[species]*pos
        t = gauss_conv(pos, transmission_wavelength, transmission, sig = intrinsic_width) # do the gaussian convolution of the line with the transmission
        ts.append(t)

    ts = np.array(ts) # make into array so that I can do elementwise comparison

    return ts








