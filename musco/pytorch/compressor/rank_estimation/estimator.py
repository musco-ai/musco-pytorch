import numpy as np

import tensorly as tl
tl.set_backend('pytorch')

from . import vbmf

def weaken_rank(rank, vbmf_rank, vbmf_weakenen_factor, min_rank = 8):
    """ Weakens (i.e. increases) a given (E)VBMF rank using a pre-defined scalar.
    
    >>> weakened_rank = max(rank - vbmf_weakenen_factor * (rank - vbmf_rank), min_rank)
    
    Notes
    --------
    >>> vbmf_weakenen_factor = 0. ->  weakened_rank = rank
    >>> vbmf_weakenen_factor = 1. -> weakened_rank = vbmf_rank < rank
    >>> 0. < vbmf_weakenen_factor < 1. -> vmbf_rank < weakened_rank < rank
    
    
    Parameters
    ----------
    rank : int
        Initial tensor rank.
    vbmf_rank : int
        Estimated rank to use for low-rank approximation at the compression step (preliminary value).
    vbmf_weakenen_factor : float
        Factor that is used to weaken (i.e. increase) a `vbmf_rank`. 
        Takes values from ``[0., 1.]``.
    Returns
    -------
    int
        Weakened (E)VBMF rank. 
        Estimated rank to use for low-rank approximation at the compression step (final value)
    """
    min_rank = int(min_rank)
    
    if rank <= min_rank:
        return rank
    
    if vbmf_rank == 0:
        weakened_rank = rank
    else:
        weakened_rank = int(rank - vbmf_weakenen_factor * (rank - vbmf_rank))
    weakened_rank = max(weakened_rank, min_rank)

    return weakened_rank

def estimate_vbmf_ranks(weights, vbmf_weakenen_factor = 1, min_rank = 8):
    """Estimates EVBMF ranks for unfoldings of a given tensor.
    
    Firstly, unfolds the tensor along modes, to which the decomposition will 
    be applied. Secondly, estimates ranks of the unfoldings (matrices) using EVBMF. 
    Thirdly, weakens estimated ranks.
    
    Parameters
    ----------
    weights : Tensor
        Tensor for which EVBMF ranks are estimated.
    vbmf_weakenen_factor : float
        Factor that is used to weaken (i.e. increase) a EVBMF rank.
    min_rank : int
        Minimal possible value for each coordinate of the estimated rank.
    
    Returns
    -------
    list or int
        Weakened estimated EVBMF ranks.
        if ``len(weights.shape) > 2``, has *(list)* type and corresponds to the Tucker2 rank.
        if ``len(weights.shape) == 2``, has *(int)* type and corresponds to the SVD rank.
        
    See Also
    --------
    weaken_rank
    
    """
    if len(weights.shape) > 2:
        unfold_0 = tl.base.unfold(weights, 0) 
        unfold_1 = tl.base.unfold(weights, 1)
        if unfold_0.is_cuda:
            unfold_0 = unfold_0.cpu()
            unfold_1 = unfold_1.cpu()

        _, diag_0, _, _ = vbmf.EVBMF(unfold_0)
        _, diag_1, _, _ = vbmf.EVBMF(unfold_1)
        ranks = [diag_0.shape[0], diag_1.shape[1]]

        ranks_weaken = [weaken_rank(unfold_0.shape[0], ranks[0], vbmf_weakenen_factor, min_rank),
                        weaken_rank(unfold_1.shape[0], ranks[1], vbmf_weakenen_factor, min_rank)]
        
    else:
        unfold = weights.data
        if unfold.is_cuda:
            unfold = unfold.cpu()
        unfold = unfold.numpy()
        try:
            _, diag, _,_ = vbmf.EVBMF(unfold)
        except:
            _, diag, _,_ = vbmf.EVBMF(unfold.T)
        rank = diag.shape[0]
        ranks_weaken = weaken_rank(min(unfold.shape), rank, vbmf_weakenen_factor, min_rank)
        
    return ranks_weaken


def _count_svd_parameters(tensor_shape,
                         rank = 8):
    fout, fin = tensor_shape[:2]
    count = rank * (fin + fout)
    return count

def _count_cp4_parameters(tensor_shape,
                         rank = 8):
    cout, cin, kh, kw = tensor_shape
    count = rank * (cin + kh + kw + cout)
    return count

def _count_cp3_parameters(tensor_shape,
                         rank = 8):
    cout, cin, kh, kw = tensor_shape
    count = rank * (cin + kh*kw + cout)
    return count

def _count_tucker2_parameters(tensor_shape,
                             rank = [8,8]):
    cout, cin, kh, kw = tensor_shape
    
    if not isinstance(rank, (list, tuple, np.ndarray)):
        rank = [rank, rank]
    count = rank[1]*cin + np.prod(rank[:2])*kh*kw + rank[0]*cout
    return np.array(count)
    

def count_parameters(tensor_shape,
                     rank = None,
                     tensor_format = 'cp3'):
    """Computes number of parameters in a tensor of a given tensor format.
    
    Parameters
    ----------
    tensor_shape : tuple
        Sizes of  tensor dimensions.
    tensor_format : {'tucker2', 'cp3', 'cp4', 'svd'}
        Tensor format in which a tensor is represented.
    rank : int or iterable
        Rank of the tensor represented in `tensor_format` format.
        Rank is *(int)* for CP or truncated SVD format.
        Rank is *(int or iterable)* for Tucker format.
        
    Returns
    -------
    int
        Number of parameters in the `tensor_format` tensor with 
        dimension sizes equal to `tensor_shape` and rank equals to `rank`.
        
    Notes
    -----
    Let `tensor_format` = 'cp4', then
    
    >>> cout, cin, kh, kw = tensor_shape
    >>> params_count = rank*(cout + cin + kh + kw)
    
    Let `tensor_format` = 'cp3', then
    
    >>> cout, cin, kh, kw = tensor_shape
    >>> params_count = rank*(cout + cin + kh*kw)
    
    Let `tensor_format` = 'tucker2', then
   
    >>> cout, cin, kh, kw = tensor_shape
    >>> rout, rin = rank
    >>> params_count = rout*cout + rin*cin + rout*rin*kh*kw
    
    Let `tensor_format` = 'svd' (in case of nn.Linear or nn.Conv2d with 1x1 kernel), then
    
    >>> fout, fin  = tensor_shape[:2]
    >>> params_count = rank*(fout + fin)
    """
    
    if tensor_format in ['cp4', 'qcp4']:
        params_count = _count_cp4_parameters(tensor_shape, rank=rank)
    elif tensor_format in ['cp3', 'qcp3']:
        params_count = _count_cp3_parameters(tensor_shape, rank=rank)
    elif tensor_format == 'tucker2':
        params_count = _count_tucker2_parameters(tensor_shape, rank=rank)    
    elif tensor_format == 'svd':
        params_count = _count_svd_parameters(tensor_shape, rank=rank)
    
    return params_count


def estimate_rank_for_compression_rate(tensor_shape,
                                  rate = 2.,
                                  tensor_format = 'tucker2',
                                  prev_rank = None,
                                  min_rank = 2):
    '''Finds a maximal rank, such that the number of (decomposed)layer's parameters decreases by a factor of `rate` after the compression step.
    
    Parameters
    ----------
    tensor_shape : tuple
        Sizes of tensor dimensions.
    tensor_format : {'tucker2', 'cp3', 'cp4', 'svd'}
        Tensor format in which a weight tensor is represented after the compression step.
    prev_rank : int or iterable or None
        None if the layer's weight hasn't been compressed yet.
        Otherwise, `prev_rank` is a rank of the already decomposed layer's weight  before the next compression step.
        `prev_rank` is *(int)*  for  cp3/cp4/svd `tensor_format`.
        `prev_rank` is *(int or iterable)*  for tucker2 `tensor_format`.
    rate : float
        Parameter reduction rate by which the number of (decomposed)layer's parameters decreases after the compression step.
        For example, `rate` = 2 means that the number of parameters halved after the compression step.
    min_rank : int
        Minimal possible value for the estimated rank.
        If rank is *(iterable)*, `min_rank` defines a minimal possible value for each coordinate of the estimated rank.
        
    Returns
    -------
    list or int
        Estimated rank. *(list)* if `tensor_format` = tucker2 and *(int)* otherwise.
        
    Notes
    -----
    Let `tensor_shape` = 'cp3' and `prev_rank` = None, then
    
    >>> cout, cin, kh, kw = tensor_shape
    >>> initial_params_count = cout*cin*kh*kw
    >>> estimated_rank = initial_params_count / (cout + cin + kh*kw) / rate
    
    Let `tensor_shape` = 'cp3' and `prev_rank` != None, then
    
    >>> cout, cin, kh, kw = tensor_shape
    >>> initial_params_count = prev_rank*(cout + cin + kh*kw)
    >>> estimated_rank = initial_params_count / (cout + cin + kh*kw) / rate  
    
    That is
    
    >>> estimated_rank = prev_rank / rate
        
        
    See Also
    --------
    count_parameters
    
    '''
    
    min_rank = int(min_rank)
    
    if prev_rank is not None:
        initial_count =  count_parameters(tensor_shape, rank = prev_rank, tensor_format = tensor_format)
    else:
        initial_count = np.prod(tensor_shape)
        
    if tensor_format in ['cp4', 'cp3', 'svd', 'qcp3', 'qcp4']:
        max_rank = (initial_count /
                    count_parameters(tensor_shape, rank = 1, tensor_format = tensor_format) / 
                    rate)
        max_rank = max(int(max_rank), min_rank) 
    
    elif tensor_format == 'tucker2':  
        cout, cin, kh, kw = tensor_shape

        beta = max(0.8*(cout/cin), 1.)

        a = (beta*kh*kw)
        b = (cin + beta*cout)
        c = -initial_count/rate

        discr = b**2 - 4*a*c
        max_rank = int((-b + np.sqrt(discr))/2/a)
        # [R4, R3]

        max_rank = max(max_rank, min_rank)
        max_rank = (int(beta*max_rank), max_rank)
        
#     print('Inside estimate, tensor shape: {}, max_rank: {}'.format(tensor_shape, max_rank))
    return max_rank
 