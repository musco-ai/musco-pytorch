import numpy as np

import tensorly as tl
tl.set_backend('pytorch')

from musco.pytorch.compressor.rank_selection import vbmf

def weaken_rank(rank, vbmf_rank, vbmf_weakenen_factor, min_rank = 21):
    min_rank = int(min_rank)
    
    if rank <= min_rank:
        return rank
    
    if vbmf_rank == 0:
        weaken_rank = rank
    else:
        weaken_rank = int(rank - vbmf_weakenen_factor * (rank - vbmf_rank))
    weaken_rank = max(weaken_rank, min_rank)

    return weaken_rank

def estimate_vbmf_ranks(weights, vbmf_weakenen_factor = 1, min_rank = 21):
    """ Unfold the 2 modes of the Tensor the decomposition will 
    be performed on, and estimates the ranks of the matrices using VBMF 
    """
    if len(weights.shape) > 2:
        unfold_0 = tl.base.unfold(weights, 0) 
        unfold_1 = tl.base.unfold(weights, 1)
        if unfold_0.is_cuda:
            unfold_0 = unfold_0.cpu()
            unfold_1 = unfold_1.cpu()
        # if 'cuda' in unfold_0.device: 
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


def count_svd_parameters(tensor_shape,
                         rank = 8):
    fout, fin = tensor_shape[:2]
    count = rank * (fin + fout)
    return count

def count_cp4_parameters(tensor_shape,
                         rank = 8):
    cout, cin, kh, kw = tensor_shape
    count = rank * (cin + kh + kw + cout)
    return count

def count_cp3_parameters(tensor_shape,
                         rank = 8):
    cout, cin, kh, kw = tensor_shape
    count = rank * (cin + kh*kw + cout)
    return count

def count_tucker2_parameters(tensor_shape,
                             ranks = [8,8]):
    cout, cin, kh, kw = tensor_shape
    
    if not isinstance(ranks, (list, tuple, np.ndarray)):
        ranks = [ranks, ranks]
    count = ranks[0]*cin + np.prod(ranks[:2])*kh*kw + ranks[1]*cout
    return np.array(count)
    

def count_parameters(tensor_shape,
                     rank = None,
                     key = 'cp3'):
    if key == 'cp4':
        params_count = count_cp4_parameters(tensor_shape, rank=rank)
    elif key == 'cp3':
        params_count = count_cp3_parameters(tensor_shape, rank=rank)
    elif key == 'tucker2':
        params_count = count_tucker2_parameters(tensor_shape, ranks=rank)    
    elif key == 'svd':
        params_count = count_svd_parameters(tensor_shape, rank=rank)
    
    return params_count


def estimate_rank_for_compression_rate(tensor_shape,
                                  rate = 2.,
                                  key = 'tucker2',
                                  prev_rank = None,
                                  min_rank = 2):
    '''
        Find max rank for which inequality (initial_count / decomposition_count > rate) holds true
    '''
    min_rank = int(min_rank)
    
    if prev_rank is not None:
        initial_count =  count_parameters(tensor_shape, rank = prev_rank, key = key)
    else:
        initial_count = np.prod(tensor_shape)
        
    if key in ['cp4', 'cp3', 'svd']:
        max_rank = (initial_count /
                    count_parameters(tensor_shape, rank = 1, key = key) / 
                    rate)
        max_rank = max(int(max_rank), min_rank) 
    
    elif key == 'tucker2':  
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
 