import re
import numpy as np
import os.path as osp
import pandas as pd

def get_O2_at_T(Trot, path = 'data/', verbose = False):
    
    '''
    Calculates the intensities of the two strong b^1 Sigma_g^+ (0 - 1) / a^1 \Delta_g (0 - 0) bands at a given temperature
    This takes in the intensities from the HITRAN database and adjusts them from the default temperature of 296 K

    Inputs:
    Trot: Temperature at the OH layer
    path: Path to the O2 lines
    verbose: whether or not to print a lot of stuff
    --------------------------
    Returns: A dataframe with O2 lines, positions, intensities and labels
    '''
    path = path + '/O2_lines_HITRAN.txt'
    T0 = 296 # the temperature that HITRAN calculates intensities with, needed for adjusting later
    
    if not verbose:
        pd.options.mode.chained_assignment = None
    
    # O2 temperatures, should technically be a distribution since the O2 layers are quite broad
    Trots = [160+50*(Trot/180)**4, 160+20*(Trot/190)**4] # the temperatures at the O2 emission layers adjusted from the OH temperature due to the difference in emission heights between the layers
    dfO2 = pd.read_csv(osp.expanduser(path), engine='python')
    dfO2['Position'] = 1e7/dfO2['nu'] # go from frequency in Hertz to wavelength in nm
    
    # define physical constants
    h = 6.62607015e-34 #J*s, Plancks constant
    c = 2.9979e8       #m/s, speed of light
    kB = 1.380649e-23  #J/K, Boltzmann constant

    ### get upper state energy, HITRAN only has lower state energy, but that and the wavelength of the photon (in vacuum, which is what HITRAN has) is all we need
    EJ = dfO2['elower']*1e2*h*c #energy from cm^-1 to Joules
    eupper = (EJ+1e9*h*c/dfO2['Position'])/kB # in K, for ease of calculation later 
    dfO2['eupper'] = eupper # set in dataframe
    
    O2_ranges = [[850, 880], [1210, 1330]] # the b^1 Sigma_g^+ / a^1 \Delta_g wavelength ranges, HITRAN doesn't label transitions so I have to do these ranges
    base_intensities = [0.4, 30] # in kR, Khomich has a^1_Delta_g as 100 kR, but it seems like even 50 kR is a little high for the Rousselot observations, but it seems to fit with PFS
    names = ['b^1_Sigma^+_g', 'a^1_Delta_g'] # unfortunately HITRAN doesn't include transitions labels, so the state will have to do
    
    dfs = []
    for i in range(len(names)): # loop over the groups
        r, name, Trot = O2_ranges[i], names[i], Trots[i]
        mask = np.logical_and(dfO2['Position']>r[0],dfO2['Position']<r[1]) # get the transitions in the group
        df = dfO2[mask] # mask it
        ### Now adjust temperatures ###
        Q0 = np.sum(np.exp(-(df["eupper"]-np.min(df["eupper"]))/Trot)) # boltzmann sum for the new temperature
        Q1 = np.sum(np.exp(-(df["eupper"]-np.min(df["eupper"]))/T0)) # boltzmann sum for the old temperature
        df['Intensity'] = df['sw']*np.exp(-(df["eupper"]-np.min(df["eupper"]))/Trot+(df["eupper"]-np.min(df["eupper"]))/T0)*(Q0/Q1) # boltzmann distribution, adjusted from the 296 K default from HITRAN
        df['Label'] = ['O2_'+name]*len(df['Intensity']) # make the labels
        
        if i == 0: # correct weird mismeasurement from HITRAN
            df['Intensity'][df['Position']>864.5] = df['Intensity'][df['Position']>864.5]*1.17 #the P and pQ branch needs to be higher by about %17 compared to what is in HITRAN
        df['Intensity'] = base_intensities[i] / np.sum(df['Intensity'])*df['Intensity'] # adjust to the measured intensities
        dfs.append(df)
        
    dfO2 = pd.concat(dfs) # put the two O2 groups together
    dfO2['Species'] = ['O2']*len(dfO2) # make the species label
    dfO2.index += 100000 # reindex so that we can easily find these later
    return dfO2

def get_atomic_lines(path = 'data/', verbose = False):
    ''' Get the atomic lines '''
    path = path+'/atomic_lines.txt'
    if not verbose:
        pd.options.mode.chained_assignment = None
        
    dfAtomic = pd.read_csv(osp.expanduser(path), engine='python')
    dfAtomic.index += 200000 # reindex so that we can easily find these later
    return dfAtomic

#####################
### OH below here ###
#####################

def get_pgopher_linelist(path = 'data/'):
    '''
        Function to load pgopher line list.
        
        Inputs:
        path: path to the .csv file from pgopher
        --------------------------
       Returns DataFrame with relevant transitions from pgopher.
    '''
    path = path + '/pgopher_OH_lines.csv' 
    A = pd.read_csv(osp.expanduser(path), skiprows=1, skipfooter = 1, engine = 'python') #load dataframa
    
    A = A.drop(labels = ["Upper Manifold", "Lower Manifold"], axis = 1) # drop things where everything is the same
    A['Species'] = A['Molecule']
    A = A.drop(labels = ["Molecule"], axis = 1) # drop things where everything is the same

    uvs, lvs = [], [] #upper and lower v's
    uNs, lNs = [], [] #upper and lower electronic state
    uBranch, lBranch = [], [] #upper and lower branch
    for n in range(len(A)):
        up, low = A['Label'][n].split("-") # split the string to get upper and lower states
        uv, lv = int(re.findall(r'\d+', up)[1]), int(re.findall(r'\d+', low)[1]) # split the transition description string to get vibrational states. re.findall(d+ finds integeter)
        uvs.append(uv) # upper v
        lvs.append(lv) # lower v

        uN, lN = int(re.findall(r'\d+', up)[-1]), int(re.findall(r'\d+', low)[-1]) # split the transition description string to get electronic states
        uNs.append(uN) # upper electronic state
        lNs.append(lN) # lower electronic state

        b = A['Branch'][n] # this is a string with the upper and lower
        uBranch.append(b[0].upper()) #this isn't actually the upper branch, but if the upper and lower letters are not the same, it is because there is a cross-electronic transition. .upper() capitalizes a string
        lBranch.append(b[1]) #this is the actual branch

    ## vibrational levels
    A["v'"] = uvs # upper
    A['v"'] = lvs # lower
    A['dv'] = A["v'"] - A['v"']

    ## these are electronic substates (pi 3/2 = 1, pi 1/2 =2)
    A["N'"]=uNs # upper electronic state
    A['N"']=lNs # lower electronic state

    # branches
    A["B'"]=uBranch #this isn't actually the upper or lower branch, but if the upper and lower letters are not the same, it is because there is a cross-electronic transition
    A['B"']=lBranch #this is the actual branch
    
    A = A[A["v'"]<10] #select only things that could actually be made in the atmosphere, which is v' lower or equal to 9
    
    return A 

def make_OH_intensities(A, Tr = 190, Tv = 1e4):
    '''Function, which given a DataFrame of pgopher lines, will calculate the intensities of each line 
        The intensities are calculated for a given rotational and vibration temperature and column density
        The rotational temperature should be between 180 and 210 K, and the vibrational temperature between 8,000-12,000 K
        The column density is just a placeholder in case I come up with something better later
        
        Inputs:
        A: Pgopher DataFrame with relevant transitions
        Tr: Rotational Temperature [K]
        Tv: Vibrational Temperature [K]
        --------------------------
        Returns: Array of intensities sorted by by wavelength
        '''
    #select all the different upper states to get zero point energies
    upper = np.unique(A["v'"])

    # define physical constants
    h = 6.62607015e-34 #J*s, Plancks constant
    c = 2.9979e8       #m/s, speed of light
    kB = 1.380649e-23  #J/K, Boltzmann constant

    wavs = [] # save wavelengths so that I can sort things correctly later
    Is = [] # save intensities
    v = A['Eupper']*1e2*h*c/(kB*Tv) #vibrational occupation
    vibrational_partition_function = np.sum(np.exp(np.unique(-v))) #these sums should be only over unique states
    
    ## lists of parameters for vibrational groups from 0 - > 9
    r_hot = np.array([0.3, 0.3, 0.49, 0.63, 0.81, 1.05, 1.17, 1.23, 4.70, 4.91])/100 # from Noll+ 2020, constant extrapolation
    
    T_hot = np.array([9500, 8800, 8100, 7400, 6636, 5952, 4544, 2181, 1741, 1019])*(Tv/1e4) #[K], from Noll+ 2020, linearly extrapolated, and scaled to T_vibrational

    for v0 in upper: # split by vibrational upper state to calculate rotational occupation
        mask = A["v'"] == v0 
        Ag = A[mask] # lines in vibrational upper state group only
        
        # Eupper is given in cm^-1
        E = Ag['Eupper']-np.min(Ag['Eupper']) #rotational energy, corrected to be at a zero-point for that vibrational upper state
        J = E*1e2*h*c/(kB*Tr) # rotational occupation
        
        v_gr = v[mask]
        # I'm a little bit in doubt about the factor of two in front, but it's in Noll+15, he could have done it from doubling the e/f state
        # No it is not from e/f doubling, it is from the spin, the total multiplicity is (2*S+1)(2*J+1), but S=1/2 always.
        g = 2*(2*Ag["J'"]+1)
        
        rotational_partition_function = np.sum(np.exp(-np.unique(J))) #these sums should be only over unique states
        
        J_hot = E*1e2*h*c/(kB*T_hot[v0]) # rotational occupation
        
        hot_rotational_partition_function = np.sum(np.exp(-np.unique(J_hot))) #these sums should be only over unique states
        
        
        I = Ag['A']*g*((1-r_hot[v0])*(np.exp(-J)/rotational_partition_function)+r_hot[v0]*(np.exp(-J_hot)/hot_rotational_partition_function))*(np.exp(-v_gr)/vibrational_partition_function) # double Boltzmann distribution
        
        
        ## vibrational state 8/9 are overpopulated due to chemistry
        if v0 == 8: 
            f = 1.60 # ~ 60% overpopulation
        if v0 == 9:
            f = 1.40 # ~ 40% overpopulation
        else:
            f = 1
        Is.append(I*f)
        wavs.append(Ag['Position'])
      
    Is = np.vstack([np.hstack(wavs), np.hstack(Is)]).T # make joint array for sorting 

    Is = Is[Is[:, 0].argsort()[::-1]] ## sorted by wavelength
    I = Is[:,1] # select only the intensities
    
    I = 1000 / np.sum(I)*I # on average, 1000 kR across the whole range, from Khomich+ 2008
    return I