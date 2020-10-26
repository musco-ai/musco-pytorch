from ..estimator import (
    _count_svd_parameters,
    _count_cp4_parameters,
    _count_cp3_parameters,
    _count_tucker2_parameters,
    count_parameters,
    estimate_rank_for_compression_rate,
    weaken_rank,
    estimate_vbmf_ranks
)


def test_count_svd_parameters():
    # params_count = rank*(fout + fin)
    
    tensor_shape = (8, 4, 1, 1)
    rank = 10
    assert _count_svd_parameters(tensor_shape, rank) == 120
    
    tensor_shape = (16, 8)
    rank = 10
    assert _count_svd_parameters(tensor_shape, rank) == 240
    

def test_count_cp4_parameters():
    # params_count = rank*(cout + cin + kh + kw)
    
    cout, cin, kh, kw = 8, 4, 3, 3
    tensor_shape = (cout, cin, kh, kw)
    rank = 10
    assert _count_cp4_parameters(tensor_shape, rank) == 180
    

def test_count_cp3_parameters():
    # params_count = rank*(cout + cin + kh*kw)
    
    cout, cin, kh, kw = 8, 4, 3, 3
    tensor_shape = (cout, cin, kh, kw)
    rank = 10
    assert _count_cp3_parameters(tensor_shape, rank) == 210
    

def test_count_tucker2_parameters():
    # if rank is (iterable): rout, rin = rank
    # if rank is (int): rout, rin = (rank, rank)
    # params_count = rout*cout + rin*cin + rout*rin*kh*kw
    
    cout, cin, kh, kw = 8, 4, 3, 3
    tensor_shape = (cout, cin, kh, kw)
    
    rank = (5, 2)
    assert _count_tucker2_parameters(tensor_shape, rank) == 138
    
    rank = 2
    assert _count_tucker2_parameters(tensor_shape, rank) == 60
    

def test_count_parameters():
    cout, cin, kh, kw = 8, 4, 3, 3
    r = 10
    
    tensor_shape = (cout, cin, 1, 1)
    rank = r
    tensor_format = 'svd'
    assert count_parameters(tensor_shape, rank, tensor_format) == 120
    
    
    tensor_shape = (cout, cin, kh, kw)
    rank = r
    tensor_format = 'cp4'
    assert count_parameters(tensor_shape, rank, tensor_format) == 180
    
    tensor_shape = (cout, cin, kh, kw)
    rank = r
    tensor_format = 'cp3'
    
    assert count_parameters(tensor_shape, rank, tensor_format) == 210
    
    tensor_shape = (cout, cin, kh, kw)
    rank = 2
    tensor_format = 'tucker2'
    assert count_parameters(tensor_shape, rank, tensor_format) == 60
    

def test_estimate_rank_for_compression_rate():
    
    # Test rank estimation for CP4 tensor format
    cout, cin, kh, kw = 20, 10, 5, 5
    tensor_shape = (cout, cin, kh, kw)
    tensor_format = 'cp4'
    
    prev_rank = None
    rate = 5.
    # initial_params_count = cout*cin*kh*kw
    # rank1_params_count = cout+cin+kh+kw
    # rank = initial_params_count / rank1_params_count / rate

    min_rank = 2
    assert estimate_rank_for_compression_rate(tensor_shape, rate, tensor_format, prev_rank, min_rank) == 25
    
    min_rank = 32
    assert estimate_rank_for_compression_rate(tensor_shape, rate, tensor_format, prev_rank, min_rank) == 32
    
    
    prev_rank = 50
    rate = 2
    # rank = prev_rank / rate

    min_rank = 2
    assert estimate_rank_for_compression_rate(tensor_shape, rate, tensor_format, prev_rank, min_rank) == 25
    
    min_rank = 32
    assert estimate_rank_for_compression_rate(tensor_shape, rate, tensor_format, prev_rank, min_rank) == 32
    
    
    # Test rank estimation for Tucker2 tensor format
    cout, cin, kh, kw = 10, 10, 2, 2
    tensor_shape = (cout, cin, kh, kw)
    tensor_format = 'tucker2'
    
    prev_rank = None
    rate = 2.
    # initial_params_count = cout*cin*kh*kw
    # if rank is (iterable): rout, rin = rank
    # if rank is (int): rout, rin = (rank, rank)
    # params_count = rout*cout + rin*cin + rout*rin*kh*kw

    min_rank = 2
    assert estimate_rank_for_compression_rate(tensor_shape, rate, tensor_format, prev_rank, min_rank) == (5, 5)
    
    min_rank = 8
    assert estimate_rank_for_compression_rate(tensor_shape, rate, tensor_format, prev_rank, min_rank) == (8, 8)
    
    
    prev_rank = (10, 10)
    rate = 3

    min_rank = 2
    assert estimate_rank_for_compression_rate(tensor_shape, rate, tensor_format, prev_rank, min_rank) == (5, 5)
    
    min_rank = 8
    assert estimate_rank_for_compression_rate(tensor_shape, rate, tensor_format, prev_rank, min_rank) == (8, 8)
    


def test_weaken_rank():
    # weakened_rank = max(rank - vbmf_weakenen_factor * (rank - vbmf_rank), min_rank)
    
    rank = 16
    vbmf_rank = 8
    min_rank = 2 
    
    vbmf_weakenen_factor = 0
    assert weaken_rank(rank, vbmf_rank, vbmf_weakenen_factor, min_rank) == 16
    
    vbmf_weakenen_factor = 1 
    assert weaken_rank(rank, vbmf_rank, vbmf_weakenen_factor, min_rank) == 8
    
    vbmf_weakenen_factor = 0.5 
    assert weaken_rank(rank, vbmf_rank, vbmf_weakenen_factor, min_rank) == 12
    
    min_rank = 10
    vbmf_weakenen_factor = 1 
    assert weaken_rank(rank, vbmf_rank, vbmf_weakenen_factor, min_rank) == 10
    


def test_estimate_vbmf_ranks():
    pass
