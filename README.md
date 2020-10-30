# ![](https://user-images.githubusercontent.com/11778655/66068156-bef1a880-e555-11e9-8d26-094071133a11.png) MUSCO: MUlti-Stage COmpression of neural networks

This repository contains supplementary code for the paper [Automated Multi-Stage Compression of Neural Networks](http://openaccess.thecvf.com/content_ICCVW_2019/html/LPCV/Gusak_Automated_Multi-Stage_Compression_of_Neural_Networks_ICCVW_2019_paper.html). 
It demonstrates how a neural network with convolutional and fully connected layers can be compressed using iterative tensor decomposition of weight tensors.

## Requirements
```
numpy
scipy
scikit-tensor-py3
absl-py
flopco-pytorch
tensorly==0.4.5
pytorch
```

## Installation
```
pip install musco-pytorch
```

## Quick Start
```python
from torchvision.models import resnet50
from flopco import FlopCo
from musco.pytorch import Compressor
from musco.pytorch.compressor.utils import standardize_model

model = resnet50(pretrained = True)
model_stats = FlopCo(model, device = device)

compressor = Compressor(model,
                        model_stats,
                        ft_every=5, 
                        nglobal_compress_iters=2б
                        config_type = 'vbmf')

# Compressor decomposes 5 layers on each iteration.
# Compressed model is saved at compressor.compressed_model.
# You have to fine-tune model after each iteration to restore accuracy.
while not compressor.done:
    # Compress layers
    compressor.compression_step()
    
    # Fine-tune compressor.compressed_model

# Convert custom layers to PyTorch layers
standardize_model(compressor.compressed_model)

compressed_model = compressor.compressed_model
```
Please, find more examples in musco/pytorch/examples folder

## Model compression
To perform multi-stage model compression via low-rank approximations, you need to permorm the following steps.

#### Load a pre-trained model
```
model = ...
```

#### Compute model statistics
```
from flopco import FlopCo

model_stats = FlopCo(model, img_size = (1, 3, 128, 128), device = device)
```
`model_stats` contains model statistics (FLOPs, params, input/output layer shapes, layers' types) collected using [FlopCo](https://github.com/juliagusak/flopco-pytorch) package.

#### Define a model compression schedule

To compress the model you need to define a compession schedule for each layer. You can do that by creating `model_compr_kwargs` dictionary.

`model_compr_kwargs` is a dictionary ``{lname : layer_compr_kwargs}`` that maps each layer in the initial model to a dictionary of parameters, which define a compression schedule for the layer.

   - If the layer is not compressing, `layer_compr_kwargs` is None.
   - Else, `layer_compr_kwargs` is a dictionary with keyword arguments defining a layer compression schedule. 

```
layer_compr_kwargs = {
   decomposition : str,
   rank_selection : str,
   manual_rank : list of (int or iterable) or None,
   parameter_reduction_rate : int or None,
   vbmf_weakenen_factor : float or None,
   curr_compr_iter : int,
}
 ```
   where

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
  
For example, you can have the following compression schedule for ResNet18
```
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
```
  
#### Create a *Compressor*
Assume that each layer is compressed twice (`nglobal_compress_iters` = 2) and that at each compression step 3 layers are compressed (`ft_every` = 3). Then the compressor is initialized as follows.

```
from musco.pytorch import Compressor
import copy

compressor = Compressor(copy.deepcopy(model),
                        model_stats,
                        ft_every=3,
                        nglobal_compress_iters=2,
                        model_compr_kwargs = model_compr_kwargs,
                       )
```

#### Compress
Alernate compression and fine-tuning steps, while compression is not done (i.e., until each compressing layer is compressed `nglobal_compress_iters` times).
In our example, untill each layer is compressed twice, we compress 3 layers, fine-tine, compress another 3 layers, fine-tune, etc.

```
while not compressor.done:
  # Compress layers
  compressor.compression_step()

  # Fine-tune compressor.compressed_model
```

#### Convert custom layers to PyTorch standard layers
Post-process the compressed model to get rid of custom layers used during the compression.
```
from musco.pytorch.compressor.utils import standardize_model

standardize_model(compressor.compressed_model)
```
Thus, `compressor.compressed_model` is a compressed model that is build from PyTorch standard layers.


#### Define a model compression schedule using default configs 
To compress  nn.Conv2d and nn.Linear layers using default compression settings, you can use the follwing config generation utils
```
from musco.pytorch.compressor.config_gen import generate_model_compr_kwargs

config = generate_model_compr_kwargs(model_stats, config_type = 'vbmf')

```
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
```
layer_compr_kwargs = {
    decomposition : None,
    rank_selection : None,
    manual_rank : None,
    parameter_reduction_rate : None,
    vbmf_weakenen_factor : None,
    curr_compr_iter : 0,
}
```
You can modify the generated config by modifying a Python dictionary
```
config.update({'conv1': None,
               'fc': None,
               'layer2.1.conv2': {
                   'decomposition': 'cp4',
                   'rank_selection': 'param_reduction',
                   'param_reduction': 1.5,
                   'curr_compr_iter': 0
                  },})
                  
model_compr_kwargs = config
```
Or you can save the generated config to `.yaml` file, manually modify `.yaml` file, and load the resulting config
```
import collections
import yaml
from yaml.representer import Representer
yaml.add_representer(collections.defaultdict, Representer.represent_dict)

# Write a generated config to .yaml file
yaml_file = "config.yaml"

with open(yaml_file, "w") as f:  
    yaml.dump(config, f)
    
# Modify .yaml file if needed
# Read from .yaml file
with open(yaml_file, "r") as f:
    model_compr_kwargs = yaml.load(f, Loader=yaml.SafeLoader)
```
To validate the correctness of the resulting config run
```
from musco.pytorch.compressor.config_gen import  validate_model_compr_kwargs

validate_model_compr_kwargs(model_compr_kwargs)
```
        
 
## Compiling the documentation
Install Sphinx: ``` apt-get install python3-sphinx```.

To build the documentation, from ``docs`` folder run:

  - ```make html```  - for HTML doc
  - ```make pdf``` - for pdf doc  (``rst2pdf`` should be installed, ```pip install rst2pdf```)

## Citing
If you used our research, we kindly ask you to cite the corresponding [paper](https://arxiv.org/abs/1903.09973).

```
@inproceedings{gusak2019automated,
  title={Automated Multi-Stage Compression of Neural Networks},
  author={Gusak, Julia and Kholiavchenko, Maksym and Ponomarev, Evgeny and Markeeva, Larisa and Blagoveschensky, Philip and Cichocki, Andrzej and Oseledets, Ivan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={0--0},
  year={2019}
}
```

## License
Project is distributed under [BSD-3-Clause License](https://github.com/musco-ai/musco-tf/blob/master/LICENSE).
