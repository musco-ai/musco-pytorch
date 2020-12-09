from collections import defaultdict
from argparse import Namespace
import torch

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

    all_algo_kwargs['cp3_conv1d'] = vars(Namespace(n_iter_max=5000,
                                            init='random',
                                            tol=1e-8,
                                            svd = None,
                                            cvg_criterion = 'rec_error',
                                            normalize_factors = True))
    
    all_algo_kwargs['cp3'] = vars(Namespace(n_iter_max=5000,
                                            init='random',
                                            tol=1e-8,
                                            svd = None,
                                            stop_criterion = 'rec_error_deviation',
                                            normalize_factors = True))
    
    all_algo_kwargs['cp4'] = vars(Namespace(n_iter_max=5000,
                                            init='random',
                                            tol=1e-8,
                                            svd = None,
                                            stop_criterion = 'rec_error_deviation',
                                            normalize_factors = True))

    all_algo_kwargs['tucker2'] = vars(Namespace(init='nvecs'))
    
    all_algo_kwargs['svd'] = vars(Namespace(full_matrices=False))
    
    QSCHEME = torch.per_channel_symmetric
    DIM = 1

    all_algo_kwargs['qcp3_conv1d'] = dict(
        **vars(Namespace(n_iter_max=500,
                         init='random',
                         tol=1e-8,
                         svd=None,
                         normalize_factors=True,
                        )),
        **vars(Namespace(dtype=torch.qint8,
                         qscheme=QSCHEME,
                         dim=DIM,
                        )),
        **vars(Namespace(qmodes=[0, 1, 2],
                         return_scale_zeropoint=False,
                         stop_criterion='rec_error_deviation',
                         return_qerrors=False,
                        ))
    )
    
    all_algo_kwargs['qcp3'] = dict(
        **vars(Namespace(n_iter_max=500,
                         init='random',
                         tol=1e-8,
                         svd=None,
                         normalize_factors=True,
                        )),
        **vars(Namespace(dtype=torch.qint8,
                         qscheme=QSCHEME,
                         dim=DIM,
                        )),
        **vars(Namespace(qmodes=[0, 1, 2],
                         return_scale_zeropoint=False,
                         stop_criterion='rec_error_deviation',
                         return_qerrors=False,
                        ))
    )
    
    all_algo_kwargs['qcp4'] = dict(
        **vars(Namespace(n_iter_max=500,
                         init='random',
                         tol=1e-8,
                         svd=None,
                         normalize_factors=True,
                        )),
        **vars(Namespace(dtype=torch.qint8,
                         qscheme=QSCHEME,
                         dim=DIM,
                        )),
        **vars(Namespace(qmodes=[0, 1, 2, 3],
                         return_scale_zeropoint=False,
                         stop_criterion='rec_error_deviation',
                         return_qerrors=False,
                        ))
    )
                                      
    return all_algo_kwargs