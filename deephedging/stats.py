"""
NumPy statistics helpers from base.py for sample-weighted results.
"""
import math
import numpy as np


def np_unique_tol( x : np.ndarray, tol : float = 1E-8, is_sorted : bool = False ) -> np.ndarray:
    """ Returns the sorted elements of 'x' which are at least 'tol' distant from each other """
    x        = np.sort(x) if not is_sorted else np.asarray(x)
    dx       = x[ 1:] - x[:-1]
    ixs      = np.full((len(x),), True, dtype=np.bool_)
    ixs[:-1] = dx > tol
    x        = x[ixs]
    assert len(x) > 0, "Operation failed: all indices removed?"
    if len(x) > 1: assert np.min( x[1:] - x[:-1] )>=tol/2., ( "Operation failed. Please debug. ", x )
    return x

# -------------------------------------------------
# Basic arithmetics for non-uniform distributions
# -------------------------------------------------

def mean( P : np.ndarray, w : np.ndarray, axis : int = None ) -> np.ndarray: 
    """ Compute P-weighted mean """
    assert P.shape == w.shape, "Bad shapes: %s != %s" % (P.shape, w.shape)
    P = np.asarray(P)
    m = np.sum( P * w, axis=axis )
    assert np.sum(np.isnan(m)) == 0, "Internal error: %g\n P = %s\n w = %s" % (m, P, w)
    return m
        
def var( P : np.ndarray, w : np.ndarray, axis : int = None ) -> np.ndarray: 
    """ Compute P-weighted variance """
    m = mean(P,w,axis)
    v = np.sum( P * (( w - m ) ** 2) , axis=axis )
    assert v >= 0., "Internal error: %g vs %g" % (np.sum( P * ( w** 2 ), axis=axis ), m**2)
    return v
        
def std( P : np.ndarray, w : np.ndarray, axis : int = None ) -> np.ndarray: 
    """ Compute P-weighted std deviation """
    s = math.sqrt( var(P,w,axis)  )
    assert np.sum(np.isnan(s)) == 0, "Internal error: %g" % s
    return s
       
def err( P : np.ndarray, w : np.ndarray, axis : int = None ) -> np.ndarray: 
    """ Compute P-weighted std error """
    assert P.shape[0] > 0, "No P?"
    e = std(P,w,axis=axis) / math.sqrt( float(P.shape[0]) )
    assert np.sum(np.isnan(e)) == 0, "Internal error: %g" % e
    return e

def mean_bins( x : np.ndarray, bins : int, weights = None, return_std = False ) -> np.ndarray:
    """
    Return a vector of 'bins' means of x.
    Bins the vector 'x' into 'bins' bins, then computes the mean of each bin, and returns the resulting vector of length 'bins'.
    
    Typical use case is computing the mean over percentiles, e.g.
    
        x = np.sort(x)
        b = mean_bins(x, 9)
        
    The resulting 'b' essentially represents E[X|ai<X<ai+1] with ai = ith/10 percentile
    
    Parameters
    ----------
        x : vector
        bins : int
            Number of bins
        weights : vector
            Sample weights or zero for unit weights
        return_std : bool
            If true, function returns a tuple of means and std devs
    Returns
    -------
        Numpy array of bins, or tuple of means and std devs of return_std is True
    """
    def w_mean_err(x,p,i1,i2):
        p_ = p[i1:i2]
        x_ = x[i1:i2]
        pp = np.sum( p_ )
        if pp<=1E-12:
            return 0.,0.
        mean = np.sum( p_*x_ ) / pp
        var  = np.sum( p_*((x_-mean)**2) ) / pp
        err  = math.sqrt( var )
        return [ mean, err ]
    
    x        = np.asarray(x)
    l        = len(x)
    assert len(x.shape) == 1, "Only plaoin vectors are supported. Need to extend to more axes"
    weights  = weights if not weights is None else np.full( x.shape, 1./float(len(x)) )
    if l <= bins:
        return x if not return_std else (x, np.zeros_like(x))
    assert bins > 0, "'bins' must be positive"

    ixs     = np.linspace(0,l,bins+1, endpoint=True, dtype=np.int32)
    results = np.array( [ w_mean_err(x,weights,ixs[i],ixs[i+1]) for i in range(bins) ] )
    assert results.shape == (bins,2), "Found shape %s" % str(results.shape)
    if not return_std:
        return results[:,0]
    return results[:,0], results[:,1]
    
def mean_cum_bins( x : np.ndarray, bins : int, weights = None ) -> np.ndarray:
    """
    Return a vector of 'bins' cummulative means of x.
    Bins the vector 'x' into 'bins' bins, then iteratively computed the mean of the first j bins, and returns the result as a vector of length 'bins'.
    
    Typical usecase is computing conditional means:
    
        x = np.sort(x)
        b = mean_bins(x, 9)
        
    The resulting 'b' essentially represents E[X|X<ai+1] with ai = ith/10 percentile

    Parameters
    ----------
        x : vector
        bins : int
            Number of bins
        weights : vector
            Sample weights or zero for unit weights
    Returns
    -------
        Nunpy array of bins.
    """
    def w_mean(x,p,i):
        return np.mean( x[0:i] ) if p is None else np.sum( (x*p)[0:i] ) / np.sum( p[0:i] )

    x    = np.asarray(x)
    l    = len(x)
    assert len(x.shape) == 1, "Only plaoin vectors are supported. Need to extend to more axes"
    if l <= bins:
        return x
    assert bins > 0, "'bins' must be positive"
    if bins == 1:
        return w_mean(x,weights,l)

    ixs  = np.linspace(0,l,bins+1, endpoint=True, dtype=np.int32)
    bins = np.array( [ w_mean(x,weights,ixs[i+1]) for i in range(bins) ] )
    return bins

def perct_exp( x : np.ndarray, lo : float, hi : float, weights : np.ndarray = None ) -> np.ndarray:
    """
    Compute the expectation over a percentile i.e. it will sort x and then compute np.mean( x[:len*lo] ) and np.mean( x[hi*len:] ).
    If a matrix instead of vector is given it will assume that the first dim is the sample dimension.
        
    Parameters
    ----------
        x  : array
        lo : lower percentile [0,1]
        hi : higher percentile [0,1]
        weights: sample weights or 1/n
        
    Returns
    -------
        If x is a vector, the function returns a 2-dimensional vector.
        If x is a matrix of second dimension n2, then the function returns a matrix of dimension [2,n2].
    """    
    lo   = float(lo)
    hi   = float(hi)
    assert lo >= 0. and lo <= 1., "Percentiles must be betwee 0 and 1, not %g" % lo
    assert hi >= 0. and hi <= 1., "Percentiles must be betwee 0 and 1, not %g" % hi
    
    if len(x.shape) == 2:
        return np.array( [ perct_exp( x[:,i], lo=lo, hi=hi, weights=weights ) for i in range(x.shape[1]) ] )
    assert len(x.shape) == 1, "Can only handle matrices or vectors"
    
    ixLo    = min( math.ceil(  x.shape[0] * lo ), x.shape[0]-1 )
    ixHi    = max( math.floor( x.shape[0] * hi ), 0 )
    ixs     = np.argsort(x)
    x       = x[ixs]
    weights = weights[ixs] if not weights is None else None
        
    return  np.array( [ np.sum( (weights*x)[:ixLo] ) / np.sum( weights[:ixLo] ) if not weights is None else np.mean( x[:ixLo] ), \
                        np.sum( (weights*x)[ixHi:] ) / np.sum( weights[ixHi:] ) if not weights is None else np.mean( x[ixHi:] ) ] )
