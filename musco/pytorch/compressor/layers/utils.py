from collections import defaultdict
from argparse import Namespace

def get_all_algo_kwargs():
    """Gets parameters for approximation algorithms.
    
    Returns
    -------
    all_algo_kwargs :  collections.defaultdict(dict)
            A dictionary ``{decomposition : algo_kwargs}``, where `decomposition` states for the approximation type and `algo_kwargs` is a dictionary containing parameters for the approximation algorithm. For the available list of algorithm  parameters,
                - see ``tensorly.decomposition.parafac()`` arguments, if `decomposition` takes values from {'cp3', 'cp4'};
                - see ``sktensor.tucker.hooi()`` arguments, if `decomposition` is 'tucker2';
                - see ``np.linalg.svd()`` arguments, if `decomposition` is 'svd'.
    """
    
    all_algo_kwargs = defaultdict(dict)
    
    all_algo_kwargs['cp3'] = vars(Namespace(n_iter_max=5000,
                                            init='random',
                                            tol=1e-8,
                                            svd = None,
                                            cvg_criterion = 'rec_error',
                                            normalize_factors = True))
    
    all_algo_kwargs['cp4'] = vars(Namespace(n_iter_max=5000,
                                        init='random',
                                        tol=1e-8,
                                        svd = None,
                                        cvg_criterion = 'rec_error',
                                        normalize_factors = True))
    
    all_algo_kwargs['tucker2'] = vars(Namespace(init='nvecs'))
    
    all_algo_kwargs['svd'] = vars(Namespace(full_matrices=False))
                                      
    return all_algo_kwargs