import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def window_segment(x, T, S):
    x = np.asarray(x) # make x an array
    n = x.shape[0] # determine number of samples in x
    if n < T: # ensure number of samples is greater than window size T
        print(f"Number of samples {n} is smaller than the window size of {T}")
        return np.empty((0, T), dtype=x.dtype)
        
    return sliding_window_view(x, window_shape=T)[::S] # return a view, i.e. a reference to an already existing array, instead of a copy which copies all elements from and already existing array
    
