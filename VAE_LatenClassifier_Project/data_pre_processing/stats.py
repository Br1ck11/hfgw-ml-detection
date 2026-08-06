import numpy as np

__all__ = [
    "stream_welford_stats",
    "stream_min_max",
    "blockwise_normalize",
    "blockwise_normalize_to_path",
]


def stream_welford_stats(mm: np.memmap, block_rows: int = 262144):
    """
    Compute global mean and standard deviation over all elements of a 2D memmap/ndarray
    using Welford's numerically stable online algorithm, iterating in row blocks.

    Parameters
    ----------
    mm : np.memmap or np.ndarray
        Array-like of shape (N, T). Only row-slices are read into RAM.
    block_rows : int
        Number of rows per streaming block (how many rows are loaded at once). Tune for your I/O.

    Returns
    -------
    mean : float
    std  : float
    """
    # Initialization of variables
    count = 0
    mean = 0.0
    M2 = 0.0
    rows = mm.shape[0] # determine shape of whole array
    for i in range(0, rows, block_rows): # iterate over whole array in block_rows steps
        block = mm[i:i + block_rows]
        b = block.ravel() # .ravel() shapes 2D input of shape (block_rows, T) into 1D output of shape (block_rows * T,) if data is contiguous in RAM, i.e. not gaps by strides etc.
        n = b.size # get the total number of samples in b, i.e. block_rows * T
        if n == 0:
            continue
        b_mean = float(b.mean()) # compute loaded chunk's mean
        # Sum of squared deviations within block
        b_M2 = float(((b - b_mean) ** 2).sum()) # compute loaded chunk's squared deviations
        
        # Combine with existing aggregates
        delta = b_mean - mean
        total = count + n # total data observed so far
        mean += delta * (n / total) # update global mean and weigh by chunk size n
        M2 += b_M2 + (delta * delta) * count * n / total
        count = total # update count to account for already processed number of data
    std = (M2 / (count - 1)) ** 0.5 if count > 1 else 0.0 # correct by 1/(N - 1) for unbiased estimate
    return mean, std


def stream_min_max(mm: np.memmap, block_rows: int = 262144):
    """
    Compute global min and max over all elements of a 2D memmap/ndarray, iterating in row blocks.

    Parameters
    ----------
    mm : np.memmap or np.ndarray
        Array-like of shape (N, T).
    block_rows : int
        Number of rows per streaming block.

    Returns
    -------
    gmin : float
    gmax : float
    """
    gmin = np.inf # initilaize s.t. first min is definetly smaller than initial gmin
    gmax = -np.inf # initilaize s.t. first max is definetly bigger than initial gmax
    rows = mm.shape[0] # determine shape of whole array
    for i in range(0, rows, block_rows): # iterate over whole array in block_rows steps
        block = mm[i:i + block_rows]
        if block.size == 0:
            continue
        # .min() and .max() compute min and max over all elements of 2D array, i.e. no flattening needed here
        bmin = float(block.min())
        bmax = float(block.max())
        if bmin < gmin:
            gmin = bmin
        if bmax > gmax:
            gmax = bmax
    return gmin, gmax


def blockwise_normalize(src_mm: np.memmap,
                        dst_mm: np.memmap,
                        mode: str,
                        params: dict,
                        eps: float = 1e-8,
                        block_rows: int = 262144):
    """
    Normalize a 2D memmap/ndarray into another memmap/ndarray in row blocks.

    Parameters
    ----------
    src_mm : memmap/ndarray, shape (N, T)
        Source data.
    dst_mm : memmap/ndarray, shape (N, T)
        Destination buffer (will be overwritten).
    mode : {'zscore', 'min_max'}
        Normalization scheme.
    params : dict
        For 'zscore', expects {'mean_value': float, 'std_dev_value': float}.
        For 'min_max', expects {'min_value': float, 'max_value': float}.
    eps : float
        Small constant to avoid division by zero.
    block_rows : int
        Number of rows per processing block.
    """
    rows = src_mm.shape[0] # determine shape of whole array
    if mode == 'zscore':
        mu = params['mean_value']
        sd = params['std_dev_value']
        denom = sd + eps
        for i in range(0, rows, block_rows): # iterate over whole array in block_rows steps
            block = src_mm[i:i + block_rows]
            dst_mm[i:i + block_rows] = (block - mu) / denom
    elif mode == 'min_max':
        mn = params['min_value']
        mx = params['max_value']
        denom = (mx - mn) + eps
        for i in range(0, rows, block_rows):
            block = src_mm[i:i + block_rows]
            dst_mm[i:i + block_rows] = (block - mn) / denom
    else:
        raise ValueError("Unsupported normalization type in blockwise_normalize")
    # Ensure results are persisted for memmaps, i.e. dst_mm is a memmap and not a normal array which does not have a flush operation
    if hasattr(dst_mm, 'flush'):
        dst_mm.flush()


def blockwise_normalize_to_path(src_mm: np.memmap,
                                dst_path: str,
                                mode: str,
                                params: dict,
                                dtype=None,
                                shape=None,
                                eps: float = 1e-8,
                                block_rows: int = 262144):
    """
    Convenience wrapper: allocate a memmap at `dst_path`, normalize `src_mm` into it
    in blocks, and return the path. Keeps RAM bounded and leaves a persistent file
    for later use.
    """
    if shape is None:
        shape = src_mm.shape
    if dtype is None:
        dtype = src_mm.dtype
    dst_mm = np.memmap(dst_path, mode='w+', dtype=dtype, shape=shape)
    blockwise_normalize(src_mm, dst_mm, mode, params, eps=eps, block_rows=block_rows)
    if hasattr(dst_mm, 'flush'):
        dst_mm.flush()
    return dst_path
