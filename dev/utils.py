import numpy as np

##################################################################
### A module with a few generally applicable utility functions ###
##################################################################

def gauss(x, mu, sig):
    '''
    Normalized Gaussian on a grid, np.vectorize makes this super slow for some reason
    
    Inputs:
    x: the grid on which to find the values
    mu: mean of the Gaussian
    sig: width of the Gaussian
    --------------------------
    Returns: the Gaussian on the grid
    '''
    return 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)

def gauss_conv(wav, xs, ys, sig=0.001): #this sigma should technically be wavelength dependent and be dependent on the thermal broadening of the line, but it doesn't really matter
    '''
    Gaussian convolution for Line Spread Function
    
    Inputs:
    
    wav: is the position where we want to evaluate it
    xs: is the grid on which we want to evaluate everything, in our case, the wavelength grid of MODTRAN
    ys: is the quantity we want convolved, in my case the transmission
    sig: is the gaussian sigma of the convolution filter. 
    --------------------------
    Returns: Gaussian Convolution of ys at wav
    '''
    mask = np.isclose(xs, wav, atol = 1) #only convolve things that are close
    xs = xs.astype(float)[mask]
    ys = ys[mask]
    f = np.exp(-(float(wav)-xs)**2/(2*sig**2))
    y = np.sum(f * ys/np.sum(f))
    return y
    
def get_blocks(masks):
    ''' A method to find overlaps between lines, effectively making a given matrix block diagonal and returning the blocks
    -------------------------
    Returns the blocks
    '''
    A = np.vstack(masks)
    A_mirrored = A+A.T
    N  = A.shape[0]

    blocks = []
    start = 0
    blocksize = 0
    starts = []

    while start < N:
        blocksize += 1

        if np.all(A_mirrored[start:start + blocksize, start + blocksize:N] == 0):
            block = A[start:start + blocksize, start:start + blocksize]
            blocks.append(block)
            start += blocksize
            blocksize = 0
        starts.append(start)

    return np.unique(starts)
