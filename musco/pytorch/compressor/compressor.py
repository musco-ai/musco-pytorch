import torch.nn as nn
from collections import defaultdict
import copy

from musco.pytorch.compressor.utils import get_compressed_model


class Compressor():
    """Multi-stage compression of a neural network using low-rank approximations.

        - nn.Conv2d layers with 1x1 spacial kernels and nn.Linear layers are compressed by approximating weights with truncated SVD.
        - nn.Conv2d layers with kxk (k>1) spacial kernels are compressed by approximating weights with Tucker2, CP3 or CP4 decomposition.
        - Compression is performed iteratively by alternating compression and fine-tuning steps. During the compression step several layers can be compressed. The fine-tuning step restores an accuracy after the compression step.

    Attributes
    ----------
    compressed_model : torch.nn.Module
    ltypes : collections.defaultdict(dict)
        A dictionary ``{lname : linfo}`` that for each layer from the initial model contains its name `lname` and  corresponding information, such as 'type', 'kernel_size', 'groups' (last two fields are defined only for nn.Conv2d).
    lnames : list of str
        Names of all nn.Conv2d and nn.Linear layers from the initial model.
        After `ranks` is initialized, all non-compressing layers are removed from `lnames` using `pop_noncompressing_lnames()` method.
    rank_selection  : {'vbmf', 'param_reduction', 'manual'}
        A method to estimate rank of tensor decompositions, which is applied to nn.Conv2d and nn.Linear layers.
    ranks : collections.defaultdict
        A dictionary ``{lname : rank for lname in lnames}``, where `rank` is *(int or iterable)* or None.
            - `rank` is None if a corresponding layer is not compressed.
            - `rank` is -1 if the kernel of the corresponding layer is approximated using automatically defined rank value.
            - `rank` is *(int or iterable)* and rank != -1 if the kernel of the corresponding layer is approximated using a manually defined rank value.
            - default is ``{lname : -1 for lname in lnames}``.
    vbmf_wfs : collections.defaultdict(float) or None
        A dictionary ``{lname : vbmf_wf for lname in lnames}``, where `vbmf_wf` is a weakenen factor used to increase tensor rank found via EVMBF. `lnames` is a list of compressing layers' names.
            - if `rank_selection` != 'vbmf', then `vbmf_wfs` = None.
            - if `rank_selection` == 'vbmf', then default is ``{lname : 0.8 for lname in lnames}``.
    param_rrs : collections.defaultdict(float) or None
        A dictionary ``{lname : param_rr for lname in lnames}``, where `param_rr` is a reduction factor by which the number of layer's parameters decrease after the compression step. `lnames` is a list of compressing layers' names.
            - if `rank_selection` != 'param_reduction', then `param_rrs` = None.
            - if `rank_selection` == 'param_reduction', then  default is ``{lname : 2 for lname in lnames}``.
    decompositions : collections.defaultdict(str)
        A dictionary ``{lname : decomposition for lname in lnames}``, where decomposition is a type of tensor method applied to approximate nn.Conv2d or nn.Linear kernel at the compression step. `lnames` is a list of compressing layers' names.
            - For nn.Conv2d with 1x1 spacial size and nn.Linear layers `decomposition` = 'svd'.
            - For nn.Conv2d with nxn (n>1) spacial size `decomposition` takes value from {'cp3', 'cp4', 'tucker2'}.
    ft_every : int
        Indicates how many layers to compress before next model fine-tuning (default is ``len(lnames)``).
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

    See Also
    --------
    CompressorVBMF
    CompressorPR
    CompressorManual
    """

    def __init__(self,
                 model,
                 model_stats,
                 rank_selection=None,
                 conv2d_nn_decomposition=None,
                 ranks=None,
                 vbmf_weakenen_factors=None,
                 param_reduction_rates=None,
                 ft_every=None,
                 nglobal_compress_iters=None,
                 all_algo_kwargs=None):
        """

        Parameters
        ----------
        model : torch.nn.Module
            Initial model to compress.
        model_stats : FlopCo
            Model statistics (FLOPs, params, input/output layer shapes) collected using FlopCo package.
        rank_selection  : {'vbmf', 'param_reduction', 'manual'}
            Initial value for `self.rank_selection`.
        conv2d_nn_decomposition : {'cp3', 'cp4', 'tucker2'}
            Tensor decomposition applied to approximate nn.Conv2d kernel with nxn (n>1) spacial size.
            nn.Conv2d kernels with 1x1 spacial size and nn.Linear weights are approximated using 'svd'.
            Used to initialize `self.decompositions`.
        vbmf_weakenen_factors : collections.defaultdict(float)
            A dictionary to update `self.vbmf_wfs`.
        param_reduction_rates : collections.defaultdict(float)
            A dictionary to update `self.param_rrs`.
        ft_every : int
            Initial value for `self.ft_every`.
        nglobal_compress_iters : int
            Initial value for `self.nglobal_compress_iters`.
        ranks : collections.defaultdict(float)
            A dictianary to update `self.ranks`.
        all_algo_kwargs :  collections.defaultdict(dict)
            A dictionary ``{decomposition : algo_kwargs}``, where `decomposition` states for the approximation type and `algo_kwargs` is a dictionary containing parameters for the approximation algorithm. For the available list of algorithm  parameters,
                - see ``tensorly.decomposition.parafac()`` arguments, if `decomposition` takes values from {'cp3', 'cp4'};
                - see ``sktensor.tucker.hooi()`` arguments, if `decomposition` is 'tucker2';
                - see ``np.linalg.svd()`` arguments, if `decomposition` is 'svd'.
        """
        self.all_algo_kwargs = all_algo_kwargs
        
        self.ltypes = model_stats.ltypes
        self.lnames = [lname for lname in self.ltypes.keys() if
                       self.ltypes[lname]['type'] in [nn.Conv2d, nn.Linear]]

        self.rank_selection = rank_selection
        self.conv2d_nn_decomposition = conv2d_nn_decomposition

        self.ranks = None
        self.initialize_ranks(ranks)

        self.pop_noncompressing_lnames()

        self.vbmf_wfs = None
        self.param_rrs = None

        if self.rank_selection == 'vbmf':
            self.initialize_vbmf_wfs(vbmf_weakenen_factors)

        elif self.rank_selection == 'param_reduction':
            self.initialize_param_rrs(param_reduction_rates)

        self.decompositions = None
        self.initialize_decompositions()

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
                ranks=self.ranks,
                decompositions=self.decompositions,
                rank_selection=self.rank_selection,
                vbmf_weaken_factors=self.vbmf_wfs,
                param_reduction_rates=self.param_rrs,
                layer_types=self.ltypes,
                return_ranks=True,
                all_algo_kwargs = self.all_algo_kwargs)
            self.ranks.update(new_ranks)

            self.curr_ncompr_layers += 1
            if self.curr_ncompr_layers >= self.nglobal_compress_iters * len(self.lnames):
                break

    def pop_noncompressing_lnames(self):
        for lname in self.lnames:
            if self.ranks[lname] is None:
                self.ranks.pop(lname)
        self.lnames = list(self.ranks.keys())

    def initialize_param_rrs(self, param_rrs):
        self.param_rrs = {k: 2 for k in self.lnames}

        if param_rrs is not None:
            updates = {k: v for k, v in param_rrs.items() if k in self.lnames}

            self.param_rrs.update(updates)

    def initialize_vbmf_wfs(self, vbmf_wfs):

        self.vbmf_wfs = {k: 0.8 for k in self.lnames}

        if vbmf_wfs is not None:
            updates = {k: v for k, v in vbmf_wfs.items() if k in self.lnames}

            self.vbmf_wfs.update(updates)

    def initialize_ranks(self, ranks):

        self.ranks = {k: -1 for k in self.lnames}

        if ranks is not None:
            updates = {k: v for k, v in ranks.items() if k in self.lnames}

            self.ranks.update(updates)

    def initialize_decompositions(self):
        self.decompositions = defaultdict()

        for lname in self.lnames:
            options = self.ltypes[lname]

            if options['type'] == nn.Conv2d and options['kernel_size'] != (1, 1):
                self.decompositions[lname] = self.conv2d_nn_decomposition
            else:
                self.decompositions[lname] = 'svd'


class CompressorVBMF(Compressor):
    """Multi-stage compression of a neural network using low-rank approximations with automated rank selection based on EVBMF.

    Examples
    --------
    Let us perform ResNet50 multi-stage compression by compressing all nn.Conv2d and nn.Linear layers.
    
        - nn.Conv2d layers with nxn (n > 1) spacial kernels are compressed using **Tucker2 low-rank approximation** with **EVBMF rank selection**.
        - nn.Conv2d layers with 1x1 spacial kernels and nn.Linear layers are compressed using **SVD low-rank approximation** with **EVBMF rank selection**.
        - We compress the model by alternating compression and fine-tuning steps. 
        - Assume that each layer is compressed twice (`nglobal_compress_iters` = 2) and that at each compression step 5 layers are compressed (`ft_every` = 5). The fine-tuning step restores an accuracy after the compression step. Untill each layer is compressed twice, we compress 5 layers, fine-tine, compress another 5 layers, fine-tune, etc.

    ::
    
        from flopco import FlopCo
        from musco.pytorch import CompressorVBMF
        
        from musco.pytorch.compressor.layers.utils import get_all_algo_kwargs
        from musco.pytorch.compressor.utils import standardize_model
        
        from torchvision.models import resnet50
        import copy
        
        device = 'cuda'
        
        # Load the model
        model = resnet50(pretrained = True).to(device)
        
        # Collect model statistics (FLOPs, params, layer input/output shapes)
        model_stats = FlopCo(model, img_size=(1, 3, 128, 128), device=device)  
        
        # Set parameters for approximation algorithms
        all_algo_kwargs = get_all_algo_kwargs()
        
        # Create a compressor
        compressed_model = copy.deepcopy(model)
        compressor = CompressorVBMF(compressed_model,
                                    model_stats,
                                    ft_every=5, 
                                    nglobal_compress_iters=2,
                                    all_algo_kwargs=all_algo_kwargs)
        
        # Alernate compression and fine-tuning steps, while compression is not done
        # (i.e., until each compressing layer is compressed `nglobal_compress_iters` times)
        while not compressor.done:
            # Compress layers
            compressor.compression_step()
        
            # Fine-tune compressor.compressed_model
        
        # Compressed model is saved at compressor.compressed_model.
        compressed_model = compressor.compressed_model
        
        # Replace custom layers with standard nn.Module layers.
        standardize_model(compressed_model)
    
    See Also
    --------
    musco.pytorch.compressor.rank_estimation.estimator.estimate_vbmf_ranks
    CompressorPR
    CompressorManual
    
    """
    
    def __init__(self,
                 model,
                 model_stats,
                 ranks=None,
                 vbmf_weakenen_factors=None,
                 ft_every=None,
                 nglobal_compress_iters=1,
                 all_algo_kwargs=None):
        
        super(CompressorVBMF, self).__init__(
             model,
             model_stats,
             rank_selection='vbmf',
             conv2d_nn_decomposition='tucker2',
             ranks=ranks,
             vbmf_weakenen_factors=vbmf_weakenen_factors,
             ft_every=ft_every,
             nglobal_compress_iters=nglobal_compress_iters,
             all_algo_kwargs=all_algo_kwargs)
        

class CompressorPR(Compressor):
    """Multi-stage compression of a neural network using low-rank approximations with automated rank selection based on layers' parameter reduction rate.
    
    Examples
    --------
    Let us perform ResNet50 multi-stage compression by compressing 2 layers ('layer1.2.conv1', 'layer1.2.conv2').
    
        - nn.Conv2d layers with nxn (n > 1) spacial kernels are compressed using CP3 low-rank approximation with **rank selection based on layers' parameter reduction rate**. (You can use CP4 or Tucker2 instead of CP3.)
        - nn.Conv2d layers with 1x1 spacial kernels and nn.Linear layers are compressed using **SVD low-rank approximation** with **rank selection based on layers' parameter reduction rate**.
        - We compress the model by alternating compression and fine-tuning steps. 
        - Assume that each layer is compressed twice (`nglobal_compress_iters` = 2) and that at each compression step 5 layers are compressed (`ft_every` = 5). The fine-tuning step restores an accuracy after the compression step. Untill each layer is compressed twice, we compress 5 layers, fine-tine, compress another 5 layers, fine-tune, etc.
        
    ::
    
        from flopco import FlopCo
        from musco.pytorch import CompressorPR
        
        from musco.pytorch.compressor.layers.utils import get_all_algo_kwargs
        from musco.pytorch.compressor.utils import standardize_model
        
        from torchvision.models import resnet50
        import copy

        device = 'cuda'

        # Load the model
        model = resnet50(pretrained = True).to(device)

        # Collect initial model statistics
        model_stats = FlopCo(model, img_size=(1, 3, 128, 128), device=device)

        # Get names of all convolutional and fully-connected layers
        lnames = list(model_stats.flops.keys())

        # Choose layers you want to compress
        compressing_lnames = ['layer1.2.conv1', 'layer1.2.conv2']
        noncompressing_lnames = list(filter(lambda lname : lname not in compressing_lnames,
                                            lnames))

        # Set tensor decomposition type for convolutional layers
        conv2d_nn_decomposition = 'cp3' 

        # Set rank = None for layers we don't compress
        ranks = {lname : None for lname in noncompressing_lnames}

        # Choose parameter reduction rate for each layer you compress
        param_reduction_rates = {'layer1.2.conv1' : 2, 'layer1.2.conv2' : 3}
        
        # Set parameters for approximation algorithms
        all_algo_kwargs = get_all_algo_kwargs()

        # Initialize a compressor
        compressed_model = copy.deepcopy(model)
        compressor = CompressorPR(compressed_model,
                                  model_stats,
                                  ranks=ranks,
                                  ft_every=len(compressing_lnames), 
                                  nglobal_compress_iters=2,
                                  param_reduction_rates=param_reduction_rates,
                                  conv2d_nn_decomposition=conv2d_nn_decomposition,
                                  all_algo_kwargs=all_algo_kwargs)
                                  
        # Alernate compression and fine-tuning steps, while compression is not done
        # (i.e., until each compressing layer is compressed `nglobal_compress_iters` times)
        while not compressor.done:
            # Compress layers
            compressor.compression_step()
        
            # Fine-tune compressor.compressed_model
        
        # Compressed model is saved at compressor.compressed_model.
        compressed_model = compressor.compressed_model
        
        # Replace custom layers with standard nn.Module layers.
        standardize_model(compressed_model)
    
    See Also
    --------
    musco.pytorch.compressor.rank_estimation.estimator.estimate_rank_for_compression_rate
    CompressorVBMF
    CompressorManual
    
    """
    def __init__(self,
                 model,
                 model_stats,
                 conv2d_nn_decomposition=None,
                 ranks=None,
                 param_reduction_rates=None,
                 ft_every=None,
                 nglobal_compress_iters=1,
                 all_algo_kwargs=None):
        
        super(CompressorPR, self).__init__(
             model,
             model_stats,
             rank_selection='param_reduction',
             conv2d_nn_decomposition=conv2d_nn_decomposition,
             ranks=ranks,
             param_reduction_rates=param_reduction_rates,
             ft_every=ft_every,
             nglobal_compress_iters=nglobal_compress_iters,
             all_algo_kwargs=all_algo_kwargs)
        
        
class CompressorManual(Compressor):
    """Compression of a neural network using low-rank approximations with manually defined ranks.
    
    Examples
    --------
    Let us perform ResNet50 one-stage compression by compressing 2 layers ('layer1.2.conv1', 'layer1.2.conv2').
    
        - nn.Conv2d layers with nxn (n > 1) spacial kernels are compressed using Tucker2 low-rank approximation with a **manually defined rank**. (You can use CP3 or CP4 instead of Tucker2.)
        - nn.Conv2d layers with 1x1 spacial kernels and nn.Linear layers are compressed using **SVD low-rank approximation** with a **manually defined rank**.
        - We compress the model by alternating compression and fine-tuning steps. 
        - Assume that each layer is compressed twice (`nglobal_compress_iters` = 2) and that at each compression step 5 layers are compressed (`ft_every` = 5). The fine-tuning step restores an accuracy after the compression step. Untill each layer is compressed twice, we compress 5 layers, fine-tine, compress another 5 layers, fine-tune, etc.
        
    ::
    
        from flopco import FlopCo
        from musco.pytorch import CompressorManual
        
        from musco.pytorch.compressor.layers.utils import get_all_algo_kwargs
        from musco.pytorch.compressor.utils import standardize_model
        
        from torchvision.models import resnet50
        import copy

        device = 'cuda'
        
        # Load the model
        model = resnet50(pretrained = True).to(device)

        # Collect initial model statistics
        model_stats = FlopCo(model, img_size=(1, 3, 128, 128), device=device)

        # Get names of all convolutional and fully-connected layers
        lnames = list(model_stats.flops.keys())

        # Choose layers you want to compress
        compressing_lnames = ['layer1.2.conv1', 'layer1.2.conv2']
        noncompressing_lnames = list(filter(lambda lname : lname not in compressing_lnames,
                                            lnames))

        # Set tensor decomposition type for convolutional layers
        conv2d_nn_decomposition = 'tucker2' 

        # Set rank = None for layers we don't compress
        ranks = {lname : None for lname in noncompressing_lnames}

        # Set rank != None for layers we compress 
        ranks['layer1.2.conv2'] = (8, 8) # Tucker2 rank
        ranks['layer1.2.conv1'] = 11 # SVD rank
        
        # Set parameters for approximation algorithms
        all_algo_kwargs = get_all_algo_kwargs()

        # Initialize a compressor
        compressed_model = copy.deepcopy(model)
        compressor = CompressorManual(compressed_model,
                                      model_stats,
                                      ranks=ranks,
                                      ft_every=len(compressing_lnames), 
                                      nglobal_compress_iters=1,
                                      conv2d_nn_decomposition=conv2d_nn_decomposition,
                                      all_algo_kwargs=all_algo_kwargs)

        # Alernate compression and fine-tuning steps, while compression is not done
        # (i.e., until each compressing layer is compressed `nglobal_compress_iters` times)
        while not compressor.done:
            # Compress layers
            compressor.compression_step()
        
            # Fine-tune compressor.compressed_model
        
        # Compressed model is saved at compressor.compressed_model.
        compressed_model = compressor.compressed_model
        
        # Replace custom layers with standard nn.Module layers.
        standardize_model(compressed_model)
        
    See Also
    --------
    CompressorVBMF
    CompressorPR

    """
    def __init__(self,
                 model,
                 model_stats,
                 conv2d_nn_decomposition=None,
                 ranks=None,
                 ft_every=None,
                 nglobal_compress_iters=1,
                 all_algo_kwargs=None): 

        super(CompressorManual, self).__init__(
                model,
                model_stats,
                rank_selection='manual',
                conv2d_nn_decomposition=conv2d_nn_decomposition,
                ranks=ranks,
                vbmf_weakenen_factors=None,
                param_reduction_rates=None,
                ft_every=ft_every,
                nglobal_compress_iters=nglobal_compress_iters,
                all_algo_kwargs=all_algo_kwargs)
