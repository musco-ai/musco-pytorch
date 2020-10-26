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
from musco.pytorch import CompressorVBMF, CompressorPR, CompressorManual

model = resnet50(pretrained = True)
model_stats = FlopCo(model, device = device)

compressor = CompressorVBMF(model,
                            model_stats,
                            ft_every=5, 
                            nglobal_compress_iters=2)

while not compressor.done:
    # Compress layers
    compressor.compression_step()
    
    # Fine-tune compressor.compressed_model

compressed_model = compressor.compressed_model

# Compressor decomposes 5 layers on each iteration.
# Compressed model is saved at compressor.compressed_model.
# You have to fine-tune model after each iteration to restore accuracy.

```
Please, find more examples in musco/pytorch/examples folder

## Compress the model

You can compress the model using diffrenet strategies depending on rank selection method.

- Using any of the below listed compressors, you can optionally specify:
     - which layers will NOT be compressed (```ranks = {lname : None for lname in noncompressing_lnames}```)
     - how many layers to compress before next model fine-tuning (```ft_every = 3```, i.e. compression schedule is as follows: compress 3 layers, fine-tine, compress another 3 layers, fine-tune, ... )
     - how many times to compress each layer (```nglobal_iters = 2```, by default 1)
        

- **CompressorVBMF**:  ranks are determined  by a global analytic solution of variational Bayesian matrix factorization (EVBMF)
    - Tucker2 decomposition is used for nn.Conv2d layers with kernels (n, n), n > 1
    - SVD is used for nn.Linear and nn.Conv2d with kernels (1, 1)
    - You can optionally specify:
        - weakenen factor for VBMF rank(```vbmf_weakenen_factors = {lname : factor for lname in lnames}```)



- **CompressorPR**: ranks correspond to chosen fixed parameter reduction rate (specified for each layer, default: 2x for all layers)

    - Tucker2/CP3/CP4 decomposition is used for nn.Conv2d layers with kernels (n, n), n > 1
    - SVD is used for nn.Linear and nn.Conv2d with kernels (1, 1)
    - You can optionally specify:
        - which decomposition to use for nn.Conv2d layers with kernels (n, n), n > 1 (```conv2d_nn_decomposition = cp3```)
        - parameter reduction rate (```param_reduction_rates``` argument), can be different for each layer



- **CompressorManual**: manualy specified ranks are used

    - Tucker2/CP3/CP4 decomposition is used for nn.Conv2d layers with kernels (n, n), n > 1
    - SVD is used for nn.Linear and nn.Conv2d with kernels (1, 1)
    - You can optionally specify:
        - which decomposition to use for nn.Conv2d layers with kernels (n, n), n > 1 (```conv2d_nn_decomposition = tucker2```)
        - which ranks to use (```ranks = {lname : rank for lname in lnames}```, if you don't want to compress layer set ```None``` instead ```rank``` value)
        
 
## Compiling the documentation
To build the documentation, from ``docs`` folder run

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
