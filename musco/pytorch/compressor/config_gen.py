import torch.nn as nn
from argparse import Namespace
from collections import defaultdict

def generate_layer_compr_kwargs(linfo, config_type = 'none'):
    """Defines how to compress a model's layer.

    Parameters
    ----------
    linfo : defaultdict
        Information about the layer, such as 'type', 'kernel_size', 'groups' (last two fields are defined only for nn.Conv2d)
    config_type : {'none', 'vbmf', 'param_reduction', 'template'}
        - If `config_type` is 'none', none of the layers is compressed by default.
        - If `config_type` is 'vbmf':
            - nn.Conv2d layers with nxn (n > 1) spacial kernels are compressed using **Tucker2 low-rank approximation** with **EVBMF rank selection**.
            - nn.Conv2d layers with 1x1 spacial kernels and nn.Linear layers are compressed using **SVD low-rank approximation** with **EVBMF rank selection**.
            - By default all nn.Conv2d and nn.Linear layers are compressed. Default `vbmf_wekenen_factor` is 0.8
        - If `config_type` is 'param_reduciton':
            - nn.Conv2d layers with nxn (n > 1) spacial kernels are compressed using CP3/CP4/Tucker2 low-rank approximation with **rank selection based on layers' parameter reduction rate**.
            - nn.Conv2d layers with 1x1 spacial kernels and nn.Linear layers are compressed using **SVD low-rank approximation** with **rank selection based on layers' parameter reduction rate**.
            - By default all nn.Conv2d and nn.Linear layers are compressed. Default `param_reduction_rate` is 2. Default `decomposition` for nn.Conv2d layers with nxn (n > 1) spacial kernels is Tucker2.
        - If `config_type` is 'template', a dictionary-placeholder `{lname : layer_compr_kwargs}` is generated, where
        
        | `layer_compr_kwargs` = {
        |    `decomposition` : None,
        |    `rank_selection` : None,
        |    `manual_rank` : None,
        |    `parameter_reduction_rate` : None,
        |    `vbmf_weakenen_factor` : None,
        |    `curr_compr_iter` : 0,
        | }.
            

    Returns
    -------
    defaultdict or None
        | If None, then the layer is not compressed.
        | Else, returns a dictionary `{lname : layer_compr_kwargs}` with keyword arguments `layer_compr_kwargs` defining a layer compression schedule,
        
        | `layer_compr_kwargs` = {
        |    `decomposition` : *(str)*,
        |    `rank_selection` : *(str)*,
        |    `manual_rank` : *(list of (int or iterable) or None)*,
        |    `parameter_reduction_rate` : *(int or None)*,
        |    `vbmf_weakenen_factor` : *(float or None)*,
        |    `curr_compr_iter` : *(int)*,
        | },

    See Also
    --------
    musco.pytorch.compressor.compressor.Compressor.model_compr_kwargs
    validate_model_compr_kwargs
    """
    
    if config_type == 'vbmf':
        
        if linfo['type'] == nn.Conv2d:
            if linfo['groups'] != 1:
                return None

            if linfo['kernel_size'] != (1, 1):
                decomposition = 'tucker2'
            else:
                decomposition = 'svd'

        elif linfo['type'] == nn.Linear:
            decomposition = 'svd'

        layer_compr_kwargs = defaultdict(None, {'decomposition': decomposition,
                                                'rank_selection': 'vbmf',
                                                'manual_rank': None,
                                                'param_reduction_rate': None,
                                                'vbmf_weakenen_factor': 0.8,
                                                'curr_compr_iter': 0})
    elif config_type == 'param_reduction':
        
        if linfo['type'] == nn.Conv2d:
            if linfo['groups'] != 1:
                return None

            if linfo['kernel_size'] != (1, 1):
                decomposition = 'tucker2'
            else:
                decomposition = 'svd'

        elif linfo['type'] == nn.Linear:
            decomposition = 'svd'

        layer_compr_kwargs = defaultdict(None, {'decomposition': decomposition,
                                                'rank_selection': 'param_reduction',
                                                'manual_rank': None,
                                                'param_reduction_rate': 1.5,
                                                'vbmf_weakenen_factor': None,
                                                'curr_compr_iter': 0})
    elif config_type == 'none':
        layer_compr_kwargs = None
    
    elif config_type == 'template':
        layer_compr_kwargs = defaultdict(None, {'decomposition': None,
                                        'rank_selection': None,
                                        'manual_rank': None,
                                        'param_reduction_rate': None,
                                        'vbmf_weakenen_factor': None,
                                        'curr_compr_iter': 0})
    return layer_compr_kwargs


def generate_model_compr_kwargs(model_stats, config_type = 'none'):
    """Defines how to compress each layer in the model.

    Generates default initialization for the model compression schedule.

    Parameters
    ----------
    model_stats : FlopCo
        Model statistics (FLOPs, params, input/output layer shapes, layers' types) collected using FlopCo package.
    config_type : {'none', 'vbmf', 'param_reduction', 'template'}

    Returns
    -------
    deafaultdict(defaultdict)
        A dictionary ``{lname : layer_compr_kwargs}`` that maps each layer in the initial model to a dictionary of parameters, which define a compression schedule for the layer.

    See Also
    --------
    generate_layer_compr_kwargs
    musco.pytorch.compressor.compressor.Compressor.model_compr_kwargs
    validate_model_compr_kwargs
    """
    model_compr_kwargs = defaultdict(defaultdict)

    for lname, linfo in model_stats.ltypes.items():
        model_compr_kwargs[lname] = generate_layer_compr_kwargs(linfo, config_type)

    return model_compr_kwargs


def validate_model_compr_kwargs(model_compr_kwargs):
    for lname, compr_kwargs in model_compr_kwargs.items():
        
        if compr_kwargs is not None:
            compr_kwargs = Namespace(**compr_kwargs)
            
            assert compr_kwargs.rank_selection is not None

            # Check that for a given rank_selection necessary parameters are defined 
            if compr_kwargs.rank_selection == 'manual':
                assert compr_kwargs.manual_rank is not None

            elif compr_kwargs.rank_selection == 'vbmf':
                assert compr_kwargs.vbmf_weakenen_factor is not None

            elif compr_kwargs.rank_selection == 'param_reduction':
                assert compr_kwargs.param_reduction_rate is not None

            if compr_kwargs.decomposition in ['cp3', 'cp4']:
                assert compr_kwargs.rank_selection != 'vbmf'
                
            # Check that appropriate decompositions are used for each layer type