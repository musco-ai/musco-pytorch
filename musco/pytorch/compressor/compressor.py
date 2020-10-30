import torch.nn as nn
from collections import defaultdict
import copy

from musco.pytorch.compressor.utils import get_compressed_model
from musco.pytorch.compressor.layers.utils import get_all_algo_kwargs


class Compressor():
    """Multi-stage compression of a neural network using low-rank approximations.

        - nn.Conv2d layers with 1x1 spacial kernels and nn.Linear layers are compressed by approximating weights with truncated SVD.
        - nn.Conv2d layers with kxk (k>1) spacial kernels are compressed by approximating weights with Tucker2, CP3 or CP4 decomposition.
        - Compression is performed iteratively by alternating compression and fine-tuning steps. During the compression step several layers can be compressed. The fine-tuning step restores an accuracy after the compression step.

    **By default none of the layers is compressed**.


    Attributes
    ----------
    compressed_model : torch.nn.Module
    model_stats : FlopCo
        Model statistics (FLOPs, params, input/output layer shapes, layers' types) collected using FlopCo package. Particularly, `model_stats.ltypes` is a dictionary ``{lname : linfo}`` that for each layer from the initial model contains its name `lname` and  corresponding information, such as 'type', 'kernel_size', 'groups' (last two fields are defined only for nn.Conv2d).
    model_compr_kwargs : collections.defaultdict(dict)
        | A dictionary ``{lname : layer_compr_kwargs}`` that maps each layer in the initial model to a dictionary of parameters, which define a compression schedule for the layer.
        
        - If the layer is not compressing, `layer_compr_kwargs` is None.
        - Else, `layer_compr_kwargs` is a dictionary with keyword arguments defining a layer compression schedule. 
        
        | `layer_compr_kwargs` = {
        |    `decomposition` : *(str)*,
        |    `rank_selection` : *(str)*,
        |    `manual_rank` : *(list of (int or iterable) or None)*,
        |    `parameter_reduction_rate` : *(int or None)*,
        |    `vbmf_weakenen_factor` : *(float or None)*,
        |    `curr_compr_iter` : *(int)*,
        | },
        | where
        
        - `decomposition` *(str)* is a type of tensor method applied to approximate nn.Conv2d or nn.Linear kernel at the compression step.
        
            - For nn.Conv2d with 1x1 spacial size and nn.Linear layers `decomposition` = 'svd'.
            - For nn.Conv2d with nxn (n>1) spacial size `decomposition` takes value from {'cp3', 'cp4', 'tucker2'}, default is 'tucker2'.
            
        - `rank_selection` *(str)* is a method to estimate rank of tensor decompositions, which is applied to nn.Conv2d and nn.Linear layers. `rank_selection` takes a value from {'vbmf', 'param_reduction', 'manual'}.
        
        - `manual_rank` *list of (int or iterable) or None*.
        
            - `manual_rank` is None if the kernel of the corresponding layer is approximated using automatically defined rank value (i.e. `rank_selection` != 'manual').
            - `manual_rank` is *list of (int or iterable)* if the kernel of the corresponding layer is approximated using a manually defined rank value. When the layer is compressed for the i-th time, i-th element in the list defines the rank of decomposition.
            
        - `param_reduction_rate` *(int or None)* is a reduction factor by which the number of layer's parameters decrease after the compression step. 
        
            - if `rank_selection` != 'param_reduction', then `param_reduction_rate` is None.
            - if `rank_selection` == 'param_reduction', then  default is 2.
                
        - `vbmf_weakenen_factor` *(float or None)* is a weakenen factor used to increase tensor rank found via EVMBF.
            
                - if `rank_selection` != 'vbmf', then `vbmf_weakenen_factor` is None.
                - if `rank_selection` == 'vbmf', then `vbmf_weakenen_factor` takes a value from ``[0, 1]``, default is 0.8.
         
        - `curr_compr_iter` *(int)* is a counter for compression iterations for the given layer.
        
    lnames : list of str
        Names of all compressing nn.Conv2d and nn.Linear layers from the initial model.
    ranks : collections.defaultdict
        A dictionary ``{lname : rank for lname in lnames}``, where `rank` is *(int or iterable)* and defines a rank of decomposition used to compress a layer.
    ft_every : int
        Indicates how many layers to compress before next model fine-tuning (default is ``len(self.lnames)``).
        For example, if `ft_every` == 3, then the compression schedule is as follows: compress 3 layers, fine-tine,
        compress another 3 layers, fine-tune, etc.
    nglobal_compress_iters : int
        Indicates how many times each layer should be compressed (default is 1).
    niters : int
        Total number of compression iterations, ``niters = nglobal_compress_iters * len(lnames)) // ft_every``,
        where `lnames` is a list of compressing layers' names.
    curr_iter : int
        A counter for compression iterations (default is 0).
    curr_ncompr_layers : int
        A counter for the number of compressed layers (default is 0).
    done : bool
        Indicates whether the compression is terminated (default is False).
        When ``curr_iter >= niters``, `done` is set to be True.
        
        
    Examples
    --------
    Let us perform ResNet18 multi-stage compression by compressing 3 layers 
    
        - 'layer3.1.conv2' is compressed using Tucker2 low-rank approximation with manually selected rank.
        - 'layer2.1.conv2' is compressed using Tucker2 low-rank approximation with EVBMF rank selection.
        - 'layer1.1.conv2' is compressed using CP4 low-rank approximation with rank selection based on layers' parameter reduction rate.
        
     Assume that each layer is compressed twice (`nglobal_compress_iters` = 2) and that at each compression step 3 layers are compressed (`ft_every` = 3). The fine-tuning step restores an accuracy after the compression step. Untill each layer is compressed twice, we compress 3 layers, fine-tine, compress another 3 layers, fine-tune, etc.

    ::
    
        from torchvision.models import resnet18
        from flopco import FlopCo
        from musco.pytorch import Compressor
        import copy

        device = 'cuda'
        
        # Load the model
        model = resnet18(pretrained=True).to(device)

        # Collect initial model statistics
        model_stats = FlopCo(model,
                             img_size = (1, 3, 128, 128),
                             device = device)

        # Set a model compression schedule
        model_compr_kwargs = {
            'layer3.1.conv2': {'decomposition': 'tucker2',
                               'rank_selection': 'manual',
                               'manual_rank': [(32, 32), (16, 16)],
                               'curr_compr_iter': 0
                              },
            'layer2.1.conv2': {'decomposition': 'tucker2',
                               'rank_selection': 'vbmf',
                               'vbmf_weakenen_factor': 0.9,
                               'curr_compr_iter': 0
                              },
            'layer1.1.conv2': {'decomposition': 'cp4',
                              'rank_selection': 'param_reduction',
                              'param_reduction_rate': 4,
                              'curr_compr_iter': 0
                              },
        }


        # Initialize a compressor
        compressor = Compressor(copy.deepcopy(model),
                                model_stats,
                                ft_every=3,
                                nglobal_compress_iters=2,
                                model_compr_kwargs = model_compr_kwargs,
                               )


        # Alernate compression and fine-tuning steps, while compression is not done
        # (i.e., until each compressing layer is compressed `nglobal_compress_iters` times)
        while not compressor.done:
                    # Compress layers
                    compressor.compression_step()

                    # Fine-tune compressor.compressed_model
                    
        # Replace custom layers with standard nn.Module layers.
        standardize_model(compressor.compressed_model)
        
        # compressor.compressed_model is our final compressed and standardized model.
        

    See Also
    --------
    CompressorVBMF
    CompressorPR
    CompressorManual
    """

    def __init__(self,
                 model,
                 model_stats,
                 model_compr_kwargs = None,
                 ft_every=1,
                 nglobal_compress_iters=None,
                 all_algo_kwargs=None):
        """

        Parameters
        ----------
        model : torch.nn.Module
            Initial model to compress.
        model_stats : FlopCo
            Model statistics (FLOPs, params, input/output layer shapes) collected using FlopCo package.
        model_compr_kwargs: collection.defaultdict(dict)
            A dictionary to update default `self.model_compr_kwargs`
        ft_every : int
            Initial value for `self.ft_every`.
        nglobal_compress_iters : int
            Initial value for `self.nglobal_compress_iters`.
        all_algo_kwargs :  collections.defaultdict(dict)
            A dictionary ``{decomposition : algo_kwargs}``, where `decomposition` states for the approximation type and `algo_kwargs` is a dictionary containing parameters for the approximation algorithm. For the available list of algorithm  parameters,
                - see ``tensorly.decomposition.parafac()`` arguments, if `decomposition` takes values from {'cp3', 'cp4'};
                - see ``sktensor.tucker.hooi()`` arguments, if `decomposition` is 'tucker2';
                - see ``np.linalg.svd()`` arguments, if `decomposition` is 'svd'.
        """
        self.model_stats = model_stats
        
        self.init_model_compr_kwargs(model_stats, model_compr_kwargs)

        self.init_all_algo_kwargs(all_algo_kwargs)
            
        self.lnames = [lname for lname in self.model_compr_kwargs.keys() if
                       self.model_compr_kwargs[lname] is not None]

        self.ranks = defaultdict()

        self.ft_every = len(self.lnames)
        if ft_every:
            self.ft_every = min(ft_every, len(self.lnames))

        self.nglobal_compress_iters = 1
        if nglobal_compress_iters:
            self.nglobal_compress_iters = nglobal_compress_iters

        self.niters = (self.nglobal_compress_iters * len(self.lnames)) // self.ft_every
        self.niters += ((self.nglobal_compress_iters * len(self.lnames)) % self.ft_every != 0)

        self.curr_iter = 0
        self.curr_ncompr_layers = 0
        self.done = False

        self.compressed_model = copy.deepcopy(model)

    def __str__(self):
        print_info = "\n".join([str({k: v}) for k, v in self.__dict__.items() if k != "ltypes"])

        return str(self.__class__) + ": \n" + print_info

    def compression_step(self):
        start = (self.curr_iter * self.ft_every) % len(self.lnames)
        end = start + self.ft_every

        lnames = self.lnames[start: end]

        self.curr_iter += 1

        if self.curr_iter >= self.niters:
            self.done = True

        if end > len(self.lnames):
            lnames += self.lnames[0: end - len(self.lnames)]

        for i, lname in enumerate(lnames):

            self.compressed_model, new_ranks = get_compressed_model(
                self.compressed_model,
                layer_names=[lname],
                model_stats=self.model_stats,
                return_ranks=True,
                all_algo_kwargs = self.all_algo_kwargs,
                model_compr_kwargs = self.model_compr_kwargs)
            self.ranks.update(new_ranks)

            self.curr_ncompr_layers += 1
            if self.curr_ncompr_layers >= self.nglobal_compress_iters * len(self.lnames):
                break
                
    def init_all_algo_kwargs(self, all_algo_kwargs):
        """Initializes hyper parameters for approximation algorithms.
        
         Performs default initialization, then updates it using a passed argument `all_algo_kwargs`.
        
        See Also
        --------
        musco.pytorch.compressor.layers.utils.get_all_algo_kwargs
        
        """
        self.all_algo_kwargs = get_all_algo_kwargs()  

        if all_algo_kwargs is not None:
            self.all_algo_kwargs.update(all_algo_kwargs)            
                
    def _init_layer_compr_kwargs(self, linfo):
        """Defines how to compress a model's layer.

        Parameters
        ----------
        linfo : defaultdict
            Information about layer, such as 'type', 'kernel_size', 'groups' (last two fields are defined only for nn.Conv2d)

        Returns
        -------
        defaultdict or None
            If None, then the layer is not compressed.
            Else, returns a dictionary with keyword arguments defining a layer compression schedule. Keyword arguments include
                - 'decomposition'
                - 'rank_selection'
                - 'manual_rank'
                - 'param_reduction_rate'
                - 'vbmf_weakenen_factor'
                
        See Also
        --------
        model_compr_kwargs
        """
        pass
    

    def init_model_compr_kwargs(self, model_stats, model_compr_kwargs):
        """Defines how to compress each layer in the model.
        
        Performs default initialization for the model compression schedule, then updates it using a passed argument `model_compr_kwargs`.

        Parameters
        ----------
        model_stats : FlopCo
            Model statistics (FLOPs, params, input/output layer shapes, layers' types) collected using FlopCo package.

        Returns
        -------
        deafaultdict(defaultdict)
            A dictionary ``{lname : layer_compr_kwargs}`` that maps each layer in the initial model to a dictionary of parameters, which define a compression schedule for the layer.
            
        See Also
        --------
        model_compr_kwargs
        """
        self.model_compr_kwargs = defaultdict(defaultdict)

        for lname, linfo in model_stats.ltypes.items():
            self.model_compr_kwargs[lname] = self._init_layer_compr_kwargs(linfo)
            
        if model_compr_kwargs is not None:
            self.model_compr_kwargs.update(model_compr_kwargs)

            
class CompressorVBMF(Compressor):
    """Multi-stage compression of a neural network using low-rank approximations with automated rank selection based on EVBMF.
    
        - nn.Conv2d layers with nxn (n > 1) spacial kernels are compressed using **Tucker2 low-rank approximation** with **EVBMF rank selection**.
        - nn.Conv2d layers with 1x1 spacial kernels and nn.Linear layers are compressed using **SVD low-rank approximation** with **EVBMF rank selection**.
        - We compress the model by alternating compression and fine-tuning steps. 
        
    **By default all nn.Conv2d and nn.Linear layers are compressed. Default `vbmf_wekenen_factor` is 0.8**. 
        
    Examples
    --------
    Let us perform multi-stage compression by compressing all nn.Conv2d and nn.Linear layers.
        
    Assume that each layer is compressed twice (`nglobal_compress_iters` = 2) and that at each compression step 5 layers are compressed (`ft_every` = 5). The fine-tuning step restores an accuracy after the compression step. Untill each layer is compressed twice, we compress 5 layers, fine-tine, compress another 5 layers, fine-tune, etc.

    ::
    
        from flopco import FlopCo
        from musco.pytorch import CompressorVBMF
        import copy

        device = 'cuda'
        # Load the model
        model = ...
        model.to(device)

        # Collect initial model statistics
        model_stats = FlopCo(model,
                             img_size = (1, 3, 128, 128),
                             device = device)


        # Initialize a compressor
        compressor = CompressorVBMF(copy.deepcopy(model),
                                    model_stats,
                                    ft_every=5,
                                    nglobal_compress_iters=2,
                                   )

        # Alernate compression and fine-tuning steps, while compression is not done
        # (i.e., until each compressing layer is compressed `nglobal_compress_iters` times)
        while not compressor.done:
            # Compress layers
            compressor.compression_step()

            # Fine-tune compressor.compressed_model

        # Replace custom layers with standard nn.Module layers.
        standardize_model(compressor.compressed_model) 
    
        # compressor.compressed_model is our final compressed and standardized model.
    
    
    See Also
    --------
    musco.pytorch.compressor.rank_estimation.estimator.estimate_vbmf_ranks
    CompressorPR
    CompressorManual
    
    """
    def __init__(self,
                 model,
                 model_stats,
                 model_compr_kwargs=None,
                 ft_every=None,
                 nglobal_compress_iters=1,
                 all_algo_kwargs=None,
                 ):
        
        
        super(CompressorVBMF, self).__init__(
             model,
             model_stats,
             model_compr_kwargs = model_compr_kwargs,
             ft_every=ft_every,
             nglobal_compress_iters=nglobal_compress_iters,
             all_algo_kwargs=all_algo_kwargs)
        
        
    def _init_layer_compr_kwargs(self, linfo):
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
        return layer_compr_kwargs

        

class CompressorPR(Compressor):
    """Multi-stage compression of a neural network using low-rank approximations with automated rank selection based on layers' parameter reduction rate.
    
        - nn.Conv2d layers with nxn (n > 1) spacial kernels are compressed using CP3/CP4/Tucker2 low-rank approximation with **rank selection based on layers' parameter reduction rate**.
        - nn.Conv2d layers with 1x1 spacial kernels and nn.Linear layers are compressed using **SVD low-rank approximation** with **rank selection based on layers' parameter reduction rate**.
        - We compress the model by alternating compression and fine-tuning steps. 
        
    **By default all nn.Conv2d and nn.Linear layers are compressed. Default `param_reduction_rate` is 2. Default `decomposition` for nn.Conv2d layers with nxn (n > 1) spacial kernels is Tucker2**.
    
        
    Examples
    --------
    Let us perform multi-stage compression by compressing all nn.Conv2d and nn.Linear layers.
        
    Assume that each layer is compressed twice (`nglobal_compress_iters` = 2) and that at each compression step 5 layers are compressed (`ft_every` = 5). The fine-tuning step restores an accuracy after the compression step. Untill each layer is compressed twice, we compress 5 layers, fine-tine, compress another 5 layers, fine-tune, etc.

    ::
    
        from flopco import FlopCo
        from musco.pytorch import CompressorPR
        import copy

        device = 'cuda'
        # Load the model
        model = ...
        model.to(device)

        # Collect initial model statistics
        model_stats = FlopCo(model,
                             img_size = (1, 3, 128, 128),
                             device = device)


        # Initialize a compressor
        compressor = CompressorPR(copy.deepcopy(model),
                                  model_stats,
                                  ft_every=5,
                                  nglobal_compress_iters=2,
                                 )

        # Alernate compression and fine-tuning steps, while compression is not done
        # (i.e., until each compressing layer is compressed `nglobal_compress_iters` times)
        while not compressor.done:
            # Compress layers
            compressor.compression_step()

            # Fine-tune compressor.compressed_model

        # Replace custom layers with standard nn.Module layers.
        standardize_model(compressor.compressed_model) 
    
        # compressor.compressed_model is our final compressed and standardized model.


    See Also
    --------
    musco.pytorch.compressor.rank_estimation.estimator.estimate_rank_for_compression_rate
    CompressorVBMF
    CompressorManual
    """
    def __init__(self,
                 model,
                 model_stats,
                 model_compr_kwargs=None,
                 ft_every=None,
                 nglobal_compress_iters=1,
                 all_algo_kwargs=None):
        
        super(CompressorPR, self).__init__(
             model,
             model_stats,
             model_compr_kwargs = model_compr_kwargs,
             ft_every=ft_every,
             nglobal_compress_iters=nglobal_compress_iters,
             all_algo_kwargs=all_algo_kwargs,
             )
        
    
    def _init_layer_compr_kwargs(self, linfo):
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
        return layer_compr_kwargs

        
        
class CompressorManual(Compressor):
    """Compression of a neural network using low-rank approximations with manually defined ranks.

        - nn.Conv2d layers with nxn (n > 1) spacial kernels are compressed using Tucker2 low-rank approximation with a **manually defined rank**. (You can use CP3 or CP4 instead of Tucker2.)
        - nn.Conv2d layers with 1x1 spacial kernels and nn.Linear layers are compressed using **SVD low-rank approximation** with a **manually defined rank**.
        - We compress the model by alternating compression and fine-tuning steps. 
        - Assume that each layer is compressed twice (`nglobal_compress_iters` = 2) and that at each compression step 5 layers are compressed (`ft_every` = 5). The fine-tuning step restores an accuracy after the compression step. Untill each layer is compressed twice, we compress 5 layers, fine-tine, compress another 5 layers, fine-tune, etc.

    **By default none of the layers is compressed**.
        
    See Also
    --------
    CompressorVBMF
    CompressorPR
    """
    def __init__(self,
                 model,
                 model_stats,
                 model_compr_kwargs=None,
                 ft_every=None,
                 nglobal_compress_iters=1,
                 all_algo_kwargs=None): 

        super(CompressorManual, self).__init__(
                model,
                model_stats,
                model_compr_kwargs=model_compr_kwargs,
                ft_every=ft_every,
                nglobal_compress_iters=nglobal_compress_iters,
                all_algo_kwargs=all_algo_kwargs)