import numpy as np
from scipy.special import erf

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

def gauss_exp_scatter(x, mu, sig, r_exp, gamma_exp):
    '''
    Normalized Gaussian on a grid with an exponential scattering wing. This is the profile we have in PFS
    with a Gaussian core and an exponential scattering wing.
    
    Inputs:
    x: the grid on which to find the values
    mu: mean of the distribution, both the gaussian and the exponential
    sig: width of the Gaussian
    
    --------------------------
    Returns: the Gaussian on the grid along with the scattering wing
    '''
    dx = (x - mu)
    gauss_core = 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power(dx / sig, 2.0) / 2)
    exp_wing = r_exp/gamma_exp*np.exp(-abs(dx)/gamma_exp)
    
    return (gauss_core+exp_wing)/(1+2*r_exp) # the wing integrates to 2*r_exp, so normalize the total to unit flux

def line_profile(x, mu, sig, half_width, r_exp = 0.0, gamma_exp = 1.0):
    '''
    The line profile used for building spectra: gauss_exp_scatter (or a plain Gaussian if r_exp = 0), only evaluated within |x - mu| < half_width.
    The profile's value at the cutoff is subtracted, so it goes smoothly to zero there, and it is renormalized to unit flux within the window.
    A hard cutoff leaves a small step at mu +- half_width, which creates fake local maxima in the gaps between lines.
    
    Inputs:
    x: the grid on which to find the values, should lie within mu +- half_width
    mu: mean of the distribution
    sig: width of the Gaussian
    half_width: half width of the window the line is evaluated in
    r_exp, gamma_exp: fraction of flux and scale length of the exponential wing
    --------------------------
    Returns: the tapered, normalized profile on the grid
    '''
    edge = gauss_exp_scatter(mu + half_width, mu, sig, r_exp, gamma_exp)
    inside = (erf(half_width/(np.sqrt(2)*sig)) + 2*r_exp*(1 - np.exp(-half_width/gamma_exp)))/(1+2*r_exp) # flux of the untapered profile within the window
    return (gauss_exp_scatter(x, mu, sig, r_exp, gamma_exp) - edge)/(inside - 2*half_width*edge)

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
    starts = [0] # the first block always starts at 0. Without this a first block of size 1 never got its start recorded, and was dropped

    while start < N:
        blocksize += 1

        if np.all(A_mirrored[start:start + blocksize, start + blocksize:N] == 0):
            block = A[start:start + blocksize, start:start + blocksize]
            blocks.append(block)
            start += blocksize
            blocksize = 0
        starts.append(start)

    return np.unique(starts)


####################################################################################
### Sorted-array window helpers. These replace full-array masks/np.isclose calls ###
### inside loops and give exactly the same selections, only much faster.        ###
####################################################################################

def open_window(x, lo_val, hi_val):
    '''
    For an ascending array x, return (lo, hi) such that x[lo:hi] are exactly the elements with lo_val < x < hi_val,
    i.e. the same elements as np.logical_and(x > lo_val, x < hi_val)
    '''
    return np.searchsorted(x, lo_val, side = 'right'), np.searchsorted(x, hi_val, side = 'left')

def close_window(x, b, atol, rtol = 0.0):
    '''
    For an ascending array x with well separated (>> 1 ulp) values, and query value(s) b, return (lo, hi) such that
    x[lo:hi] are exactly the elements where np.isclose(x, b, atol = atol, rtol = rtol) is True, i.e. |x - b| <= atol + rtol*|b|.
    Works for scalar or array b.
    '''
    b = np.asarray(b, dtype = float)
    tol = atol + rtol * np.abs(b)
    n = len(x)
    lo = np.searchsorted(x, b - tol, side = 'left')
    hi = np.searchsorted(x, b + tol, side = 'right')
    # b -/+ tol is rounded, so the edges can be off by one element. Fix them with the exact test.
    def ok(i):
        ic = np.clip(i, 0, max(n-1, 0))
        return (i >= 0) & (i < n) & (np.abs(x[ic] - b) <= tol) if n else np.zeros_like(i, dtype = bool)
    lo = np.where(ok(lo - 1), lo - 1, lo)
    lo = np.where((lo < hi) & ~ok(lo), lo + 1, lo)
    hi = np.where(ok(hi), hi + 1, hi)
    hi = np.where((hi > lo) & ~ok(hi - 1), hi - 1, hi)
    return lo, hi

def any_close(x, b, atol, rtol = 0.0):
    '''
    For an ascending array x and array of queries b, return a boolean array which for each b is
    np.any(np.isclose(x, b, atol, rtol)), i.e. is there an x with |x - b| <= atol + rtol*|b|
    Since the distance grows monotonically away from b, only the nearest neighbours on either side need checking.
    '''
    b = np.asarray(b, dtype = float)
    n = len(x)
    if n == 0:
        return np.zeros(b.shape, dtype = bool)
    i = np.searchsorted(x, b)
    out = np.zeros(b.shape, dtype = bool)
    for j in (i - 1, i):
        valid = (j >= 0) & (j < n)
        xj = x[np.clip(j, 0, n-1)]
        out |= valid & (np.abs(xj - b) <= atol + rtol * np.abs(b))
    return out

def nearest_index(x, b):
    '''
    For an ascending array x, return np.argmin(np.abs(x - b)) for each b (ties go to the lower index, like argmin)
    '''
    b = np.asarray(b, dtype = float)
    n = len(x)
    i = np.clip(np.searchsorted(x, b), 1, n-1)
    left = np.abs(x[i-1] - b)
    right = np.abs(x[i] - b)
    return np.where(right < left, i, i-1)
