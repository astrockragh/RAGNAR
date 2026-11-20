import numpy as np
import os.path as osp
import pandas as pd

###############################
### Import my own functions ###
###############################

from .utils import gauss, open_window
from .species_intensities import get_atomic_lines, get_O2_at_T, get_pgopher_linelist, make_OH_intensities, DATA_PATH
from .make_spectra import get_relevant_lines, get_transmission_at_lines, make_continuum_at_lines

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


def make_dataframe(mu_stable_wav, var_stable_wav, mu_stable_I, var_stable_I, counts, transmission_curves, transmission_wavs, wav_range, Nspec, Trot_0 = 195, Tvib_0 = 1e4,\
                   importance_limit_group = 1e-2, dlambda = 15, line_spread = 0.1, resolution = np.inf, inclusion_limit = 1e-3, verbose = 1, data_path = DATA_PATH, stable = None, r_scat = 0.0125, wf = 3.4):
    
    ''' Function to make a final DataFrame with good lines
    
    Inputs:
    
    mu/var_stable_wav: mean and sigma (not variance) of the wavelength of the lines
    mu/var_stable_I: mean and sigma (not variance) of the intensity of the lines
    transmission_curves/wavs: list of the used transmission curves and corresponding wavelength grids
    wav_range: the wavelength grid to do calculations on
    Trot/vib_0: The rotational/vibrational temperature to do fiducial calculations with
    
    importance_limit_group: If the line contributes less than this, do not include in label.
    All lines that contribute down to importance_limit_group^3 are included in originIds column however, so we are able to find very weak lines again

    dlambda: How far to either side of the center of a given line to go to look for other lines that may impact that line
    line_spread: The Gaussian line width of lines, as measured IN THE INSTRUMENT [nm]
    inclusion_limit: The limit for how little a line can contribute at its own line center for us to include it in calculations
    
    verbose: Print small checkpoints
    
    ------------------------
   
   Returns: DataFrame of good lines, and a pgopher DataFrame with lines used for the calculation
    
    '''
    
    # limit the wavelength range to the range we actually find stable lines in
    min_found_wav, max_found_wav = np.min(mu_stable_wav), np.max(mu_stable_wav)
    min_wav, max_wav = np.min(wav_range), np.max(wav_range)
    
    if resolution == np.inf:
        dlambda = dlambda*line_spread
    else:
        dlambda = dlambda*(max_wav+min_wav)/2/resolution
    
    # get linelist
    A = get_pgopher_linelist(path =  data_path)

    A['Intensity'] = make_OH_intensities(A, Trot_0, Tvib_0) #placeholder intensity, for finding relative importances
    dfO2 = get_O2_at_T(Trot_0, path = data_path) #make fiducial O2 intensities
    dfAtomic = get_atomic_lines(path = data_path) #make fiducial OI, Na intensities                

    A0 = pd.concat([A, dfO2, dfAtomic]) # merge dataframes
    
    # take the mean of the transmission curves
    transmission_curve = np.mean(np.vstack(transmission_curves), axis = 0)
    transmission_wav = np.mean(np.vstack(transmission_wavs), axis = 0) #this might not be needed
    
    # limit the transmission to the important range for speed-up
    trans_wav_mask = np.logical_and(transmission_wav>min_wav, transmission_wav<max_wav)
    transmission_curve = transmission_curve[trans_wav_mask]
    transmission_wav = transmission_wav[trans_wav_mask]
 
    # limit pgopher line list to the relevant wavelength range
    A0 = A0[np.logical_and(A0['Position']>min_wav, A0['Position']<max_wav)]

    if verbose > 0: 
        print('Finding lines to include.')
    
    # limit the lines to only lines that contribute significantly
    include = get_relevant_lines(A0, delta_local = dlambda, sigma_line = line_spread, resolution = resolution, r_scat = r_scat, wf = wf, inclusion_limit = inclusion_limit)

    if verbose > 0: 
        print('Convolving transmission curve at line position')
    
    # get the transmission at the line positions
    ts = get_transmission_at_lines(A0[include], transmission_wav, transmission_curve, Trot = Trot_0)

    #limit the line list
    A0 = A0[include]
    
    # save transmission
    A0['transmission'] = ts

    # get the actual observed intensity
    A0['Intensity'] =  A0['Intensity']*A0['transmission'] 
    
    # initialize the lists of things I want to save later
    important_labels = []
    importances = []
    original_ids = []
    subline_wavs = []
    percentage_err = []
    Nlines = []
    clean = []
    transmission = []
    E_As = []
    
    # pull out the columns once, filtering the full DataFrame for every line is slow
    A0_pos = A0['Position'].to_numpy(dtype = float)
    A0_I = A0['Intensity'].to_numpy(dtype = float)
    A0_ids = A0.index.to_numpy()
    A0_EA = A0['A'].fillna(0.).to_numpy() # Einstein A's, O2 has them in 'a'
    A0_Ea = A0['a'].fillna(0.0).to_numpy()
    A0_trans = A0['transmission'].to_numpy(dtype = float)
    A0_species = A0['Species'].to_numpy()
    A0_vup, A0_vlow, A0_Jup, A0_Nup = A0["v'"].to_numpy(), A0['v"'].to_numpy(), A0["J'"].to_numpy(), A0["N'"].to_numpy()
    A0_branch, A0_label = A0['Branch'].to_numpy(), A0['Label'].to_numpy()
    
    # go through the mean of the peak wavelengths
    for wav in tqdm(mu_stable_wav, desc="Getting Line Info"):
        
        #select all the lines that could possibly contribute and calculate how much intensity they contrubite at the line center
        narrow = np.nonzero(np.logical_and(A0_pos >  wav - dlambda, A0_pos < wav + dlambda))[0]
        if resolution == np.inf:
            lines = A0_I[narrow]*gauss(wav, A0_pos[narrow], line_spread) # the intensity at that line position, this maybe should be a convolution but the error is small and this is much faster
        else:
            lines = A0_I[narrow]*gauss(wav, A0_pos[narrow], wav/resolution) # the intensity at that line position, this maybe should be a convolution but the error is small and this is much faster

        Itot = np.sum(lines) #total intensity of all lines at the line position       

        line_ids = A0_ids[narrow][lines>Itot*importance_limit_group**3] # the indices of all the lines, even very weak ones, for potential lookup

        
        # select lines that contribute to the sum of the line strength by more than the importance limit for the group
        linemask = lines>Itot*importance_limit_group # mask for lines that actually contribute
        if np.sum(linemask) == 0:
            print(wav, A0.iloc[narrow])
        imp = narrow[linemask] # lines that actually contribute
        Is = lines[linemask]
        importance = np.round(Is/np.sum(Is), 4) # the relative importance of each line in the group
         
        E_A_effective = np.sum(importance*A0_EA[imp]+importance*A0_Ea[imp]) #effective Einstein A's of the transition
        E_As.append(E_A_effective)
        fraction_missing = (Itot-np.sum(lines[linemask]))/Itot # get the fraction of intensity that is left out by only considering the important lines
        percentage_err.append(fraction_missing)

        # get the total transmission of the important lines
        trans_important = np.average(A0_trans[imp], weights = lines[linemask]/A0_trans[imp]) #get the intensity weighted average transmission 
        transmission.append(trans_important)
        
        #######################
        ### Make the labels ###
        #######################
        
        if verbose == 2:
            print('Making the labels')
            
        individual_labels = [] # save vibrational state, branch and J-state and species for check if they are all the same, if they are, the line is a 'clean' line
        label_str = '' # initialize label string
        for i in imp:
            if A0_species[i] == 'OH': # for details of the OH transition which unfortunately is the only thing we have a lot of details for
                vup = A0_vup[i] # upper vibrational state
                Jup = A0_Jup[i]
                Nup = int(A0_Nup[i])
                vlow = A0_vlow[i] # lower vibrational state
                branch = A0_branch[i] # rotational transition details, also considering satellite transition
                label = f"({int(vup)}-{int(vlow)}){branch}"
                label_str +=  label
                label_str_i = f'v{vup}J{Jup}N{Nup}' #only consider upper state
                
                individual_labels.append(label_str_i)

            else: # for everything else
                label = A0_label[i]
                label_str +=  label
                individual_labels.append(label)
                
            if i != imp[-1]:
                label_str += '|'

        if len(np.unique(individual_labels))==1: # if they are the same, it's a clean transition!
            clean.append(True)
        else:
            clean.append(False)

        Nlines.append(len(Is)) # number of important lines
        subline_wavs.append(A0_pos[imp]) # save wavelength of each subline
        important_labels.append(label_str) # save label
        importances.append(importance) 
        original_ids.append(line_ids) 

    # calculcate the distance to the closest other line
    separation = []
    for i in range(len(mu_stable_wav)):
        if i==0:
            down = np.inf
            up = mu_stable_wav[i+1]-mu_stable_wav[i]
        elif i==len(mu_stable_wav)-1:
            down = mu_stable_wav[i]-mu_stable_wav[i-1]
            up = np.inf
        else:
            down = mu_stable_wav[i]-mu_stable_wav[i-1]
            up = mu_stable_wav[i+1]-mu_stable_wav[i]
        sep = np.min([down, up]) # see if going up or down in wavelength space is closest
        separation.append(sep)

    # robust = matched one to one across every spectrum. Without that information, fall back to "found in every spectrum"
    robust = np.asarray(stable, dtype = bool) if stable is not None else counts >= Nspec
    
    # make all the columns
    lineDf = pd.DataFrame(data = mu_stable_wav, columns = ['wav']) # the mean wavelength that we find the peak of the line at
    lineDf['sigma_wav'] = var_stable_wav # the variance of the line wavelength
    lineDf['separation'] = separation # the distance to the nearest other peak

    lineDf['intensity'] = mu_stable_I # the mean intensity of the line
    lineDf['sigma_intensity'] = var_stable_I # the variance of the line intensity
    lineDf['E_A_eff'] = E_As # the effective Einstein A of the line
    
    lineDf['continuum'] = make_continuum_at_lines(wav_range, line_spread, lineDf['wav'], resolution) # the continuum strength at the line position
    lineDf['label'] = important_labels # the label of the lines, including only the lines that contribute significantly
    lineDf['number_of_lines'] = Nlines # the number of lines that contribute significantly to the peak
    lineDf['clean'] = clean # if the line consists of things that are supposed to covary
    lineDf['robust'] = robust # if the line is identifiable across different runs
    lineDf['detection_prob'] = np.round(np.array(counts.astype(int))/Nspec, 4) #probability of detecting line, the fraction of simulated spectra it is found in
    
    lineDf['subline_wav'] = subline_wavs # the wavelengths of the sublines that make up the lin
    lineDf['subline_I'] = importances # their fractional contributions to the total intensity
    lineDf['err_important'] = percentage_err # the possible error in intensity from leaving out the "unimportant" lines
    lineDf['origin_Ids'] = original_ids # the line indexes in the A0 dataframe
    lineDf['transmission'] = transmission # the transmission at the lines
    
    return lineDf, A0


#############################################################
############## Needs to change base path here ###############
#############################################################

def load_species_transmission(file = 'grid_mk50', base = '~/../../tigress/cj1223/', species_include = ['combin', 'aercld', 'molec', 'H2O', 'O3', 'O2', 'CO2', 'CH4'], verbose = 1):
    '''Function which loads a low resolution MODTRAN6 curve, since the low res runs are the only ones that output species-based absorption
        Inputs:
        file: which modtran low res curve to use
        species_include: what absorption origins to consider
        ------------------
        Returns: DataFrame with transmission for different species
    '''
    
    if verbose == 2:
        print('Loading low-resolution species dependent transmission curve')
    
    trans = pd.read_csv(osp.expanduser(base+f'modtran_species_output/{file}.csv'), skiprows = 3, nrows = 1) # load file to get column names
    columns = trans.columns
    trans = pd.read_csv(osp.expanduser(base+f'modtran_species_output/{file}.csv'), skiprows = 4, comment = '}') # load data, the file ends with a '}' line
    trans.columns = columns
    
    trans.columns = [c.strip(' ') for c in trans.columns] 
    trans['nm'] = 1e7/trans['Freq'] # convert inverse cm to nm
    
    trans = trans[['nm']+species_include] ## these are the only species that we care about (as decided by yours truly, CKJ)
    return trans

def add_absorption_origin(lineDf, line_spread = 0.1, resolution = np.inf, total_lim = 0.95, contribution_lim = 0.99, base = '~/../../tigress/cj1223/modtran_species_output', originDf_file = 'grid_mk50', \
                          important_species = ['combin', 'aercld', 'molec', 'H2O', 'O3', 'O2', 'CO2', 'CH4'], verbose = 1):
    '''Given a DataFrame with lines, WITH TRANSMISSION ALREADY CALCULATED FROM LBL, add likely origin of the absorption
    
        Inputs:
        lineDf: DataFrame with OH lines
        line_spread: the Gaussian sigma of the lines, as measured IN THE INSTRUMENT
        total_lim: if a line has transmission below this, consider it significantly attenuated
        contribution_lim: if a species has transmission below this for a given line, it contributes significantly
        
        originDf_file: which modtran low res curve to use
        important_species: what absorption origins to consider
        ------------------
        Returns: DataFrame with transmission for different species
    '''
    
    if verbose == 2:
        print(f'Considering {important_species} for absorption origins. The bitmask in the linelist will be in this order.')
    
    trans_species = load_species_transmission(file = originDf_file, base = base, species_include = important_species)
        
    masks = [] # the masks for a given species to be important
    
    if resolution == np.inf:
        delta = 2*line_spread # check for absorption within 2 times the line spread to check for a likely origin, this is mainly used because of the low resolution of the species-specific curves
    else:
        delta = 2*np.mean(lineDf['wav'])/resolution # same here but for resolution
    
    # the curve is on an ascending frequency grid, flip it to ascending wavelength so the window around each line is a slice
    nm = trans_species['nm'].to_numpy()[::-1]
    species_trans = trans_species[important_species].to_numpy()[::-1]
    
    for wav, t in zip(lineDf['wav'], lineDf['transmission']):
        lo, hi = open_window(nm, wav-delta, wav+delta)
        #there is never enough 'CO, H2Ocnt', 'N2O', 'NO2', 'SO2' for them to have below 99% transmission, even at ZD = 50
        if hi > lo:
            mask = np.min(species_trans[lo:hi], axis = 0)<contribution_lim #only species which actually contribute! This essentially says that if there is any significant absorption from a given species, include it.
        else:
            mask = np.zeros(len(important_species), dtype = bool) # no points in the window, nothing to blame
        mask[0] = t <= total_lim # the total ('combin') bit is set if the line is significantly attenuated overall. The else branch used to set the aerosol bit instead
        masks.append(mask)
        
    ### making a string based bitmask
    ss = []
    for mask in masks:
        s = ''
        for m in mask.astype(np.int8):
            s+=str(m)
        ss.append(s)
    lineDf['transmission_origin'] = ss
    
    return lineDf