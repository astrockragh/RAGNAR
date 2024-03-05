from datetime import datetime
import os, pickle
import numpy as np
import os.path as osp

from dev import get_spectra_and_peaks, define_lines_and_intensities
from dev import make_dataframe, add_absorption_origin

def ragnar(Trots = [175, 190, 205], Tvibs = [8000, 10500, 13000], save_name = 'lineList_test', importance_limit_group = 1e-2, chem_scatters = 3,\
                  min_wav = 360, max_wav = 1300, steps_per_line = 100, line_spread = 0.1, resolution = np.inf, dlambda = 15, intensity_cut = 0.0, inclusion_limit = 1e-3,\
                  linewidth_scatters = 1, LSF_err = 0.03, continuum = True, aerosols = ['background', 'volcanic'], pwvs = [1.0], zds = [0, 40], base = '~/../../tigress/cj1223/', \
                  data_path = 'data/', use_centroider = False, total_lim = 0.95, contribution_lim = 0.99, originDf_file = 'grid_mk50',\
                          important_species = ['combin', 'aercld', 'molec', 'H2O', 'O3', 'O2', 'CO2', 'CH4'], verbose = 1):
    
    '''Function make a linelist for use for wavelength calibration
    
    Inputs:
    
    Trots, Tvibs: arrays or lists of the rotational/vibrational temperatures to simulate [K]
    
    min_wav, max_wav: wavelength range to consider [nm]. Make this range a little wider than the actual spectral range you would like.
    steps_per_line: number of steps per line spread interval for doing grid calculations. Turning this up will lead to a slower but more precise calculation.
    line_spread: The Gaussian line width of lines, as measured IN THE INSTRUMENT [nm]
    resolution: Resolution of spectrograph. If this is set, overrides

    save_name: if None, do not save: else, save in lineLists directory
    
    inclusion_limit: The limit for how little a line can contribute at its own line center for us to include it in calculations
    intensity_cut: Consider only peaks that are stronger than this as relevant lines
    
    importance_limit_group: If the line contributes less than this, do not include in label.
                            All lines that contribute down to importance_limit_group^3 are included in originIds column however, so we are able to find very weak lines again
    
    dlambda: How far to either side of the center of a given line to go to look for other lines that may impact that line, times the line_spread
    
    linewidth_scatters: If 1, take zero uncertainty in the LSF, otherwise do linewidth_scatters samples of the LSF distribution
    LSF_err: Define the width of the Gaussian from which to sample LSF perturbations
    continuum: Whether or not to add in the continuum
    
    aerosols: list of aerosol models to loop over. 'background', 'volcanic' are possible
    pwvs: list precipitable water vapour to loop over [mm]. From 0.0, to 20.0, in increments of 1 are possible
    zds: list of zenith declination of telescope to loop over [degrees]. 0 to 80 in increments of 10 are possible
    base: The base path of the MODTRAN outputs. Can be tigress/c1223/ or scratch/gpfs/cj1223/
    
    total_lim: if a line has transmission below this, consider it significantly attenuated
    contribution_lim: if a species has transmission below this for a given line, it contributes significantly
    originDf_file: which modtran low res curve to use to estimate possible absorption origin
    important_species: what absorption origins to consider
    
    verbose: 0 = print almost nothing, 1 = print things to follow the progress, 2 = print enough to do some approximative debugging
    
    --------------------------
    Returns: lineList, pgopher DataFrame with possible relevant lines, list of lists of peak wavelengths, list of simulated spectra, list of transmission curves, list of transmission wavelength grids
    '''
    
    if resolution == None:
        resolution = np.inf
    else:
        resolution*=2.82982 #convert from Rayleigh Criterion to Gaussian resolution definition
    
    if LSF_err > 0.1 and linewidth_scatters>1:
        print('The LSF_err is potentially too high. It should ideally be below 0.10, and is typically below 0.05')
        if verbose != 0:
            inp = input('Are you sure that you want to continue? Press 1 to continue, and any other button to break')
            if int(inp)!=1:
                print('Please lower the LSF_err and try again')
                return _, _, _, _, _, _, _, _
            
    ##################################
    ### Define the wavelength grid ###
    ##################################
    
    if resolution != None and resolution != np.inf: # changed 22112023
        line_spread0 = (max_wav+min_wav)/2/resolution
        line_spread = 0
    else:
        line_spread0 = line_spread

    steps = steps_per_line*(max_wav-min_wav)/line_spread0 # number of total steps across the entire wavelength range
    wav_range = np.linspace(min_wav, max_wav, int(steps)) # wavelength grid to do calculations on [nm]
    
    ##########################################
    ### Make the spectra and get the peaks ###
    ##########################################

    peak_wavs, specs, continuum_specs, transmission_curves, transmission_wavs = get_spectra_and_peaks(wav_range, Trots = Trots, Tvibs = Tvibs, chem_scatters = chem_scatters, linewidth_scatters = linewidth_scatters,\
                          LSF_err = LSF_err, inclusion_limit = inclusion_limit, cont = continuum, intensity_cut = intensity_cut, line_spread = line_spread, resolution = resolution, dlambda = dlambda, use_centroider = use_centroider,\
                        aerosols = aerosols, pwvs = pwvs, zds = zds, base = base, data_path = data_path, verbose = verbose)
    if verbose!=0:
        print('Finding the common lines')
        
    ###################################################
    ### Find the common peaks and their variability ###
    ###################################################
    
    mu_stable_wav, var_stable_wav, mu_stable_I, var_stable_I, stable_counts, mu_unstable_wav, var_unstable_wav, mu_unstable_I, var_unstable_I, unstable_counts =\
                    define_lines_and_intensities(peak_wavs, specs, wav_range, line_spread = line_spread, resolution = resolution, verbose = verbose)
    
    #########################
    ### Make the linelist ###
    #########################
    
    if verbose !=0:
        print('Stable lines found, making DataFrame')
    
    if np.any(mu_unstable_wav):
        counts = np.hstack([stable_counts, unstable_counts]) 
        mu_wav, var_wav = np.hstack([mu_stable_wav, mu_unstable_wav]), np.hstack([var_stable_wav, var_unstable_wav])
        mu_I, var_I = np.hstack([mu_stable_I, mu_unstable_I]), np.hstack([var_stable_I, var_unstable_I])
    else:
        counts = stable_counts
        mu_wav, var_wav = mu_stable_wav, var_stable_wav
        mu_I, var_I = mu_stable_I, var_stable_I
    a = np.array([mu_wav, var_wav, mu_I, var_I, counts]).T
    mu_wav, var_wav, mu_I, var_I, counts = a[a[:, 0].argsort()].T

    number_of_spectra = len(specs)
        
    lineDf, A0 = make_dataframe(mu_wav, var_wav, mu_I, var_I, counts, transmission_curves, transmission_wavs, wav_range, Nspec = number_of_spectra, Trot_0 = np.mean(Trots), Tvib_0 = np.mean(Tvibs),\
                                importance_limit_group = importance_limit_group, dlambda = dlambda, line_spread = line_spread, resolution = resolution, inclusion_limit = inclusion_limit, verbose = verbose)

    ########################################
    ### Add the origin of the absorption ###
    ########################################
    
    if verbose > 0:
        print('Adding likely origin of the absorption')
    
    lineDf = add_absorption_origin(lineDf, line_spread = line_spread, resolution = resolution, total_lim = total_lim, contribution_lim = contribution_lim, originDf_file = originDf_file, important_species = important_species)
    
    if verbose>0:
        print(f'Linelist with {len(lineDf)} stable lines in has been made')
    if save_name:
        if not osp.isdir('lineLists'):
            os.mkdir('lineLists')
            
        today = datetime.today().strftime("%d%m%Y")
        path = f'lineLists/{save_name}_{today}'
        lineDf.to_csv(path+'.csv', index = False)
        
        with open(path+'.pkl', "wb") as handle:
            pickle.dump(lineDf, handle)
        
        if verbose > 0:
            print(f'Saving the linelist at {f"lineLists/{save_name}_{today}.csv"}')
            
    return lineDf, A0, peak_wavs, specs, continuum_specs, wav_range, transmission_curves, transmission_wavs