# MUSCO: Multi-Stage COmpression of neural networks

This repository contains supplementary code for the paper [MUSCO: Multi-Stage COmpression of neural networks](https://arxiv.org/pdf/1903.09973.pdf). 
It demonstrates how a neural network with convolutional and fully connected layers can be compressed using iterative tensor decomposition of weight tensors.

```tensor_decomposition```  folder contains code to decompose convolutional and fully connected weights using  Tucker-2/CP3/CP4/ tensor decompositions and SVD decomposition, respectively. 

```demo``` folder contains several notebooks and scripts that demonstrate how to
 
  -  compress convolutional/fully connected layers of any neural network using different tensor decomposition,
  - iteratively compress neural network model by alternating compression and fine-tunning steps.
  
## Instructions

**Docker container**

To avoid package installation routine, create docker container (titeled ```my_container```, for example) to work in using docker image https://hub.docker.com/r/jgusak/tensor_compression_od. 
Use port ```4567``` (or any port you like) to run jupyter notebook at. 

```bash
nvidia-docker run --name my_container -it -v musco:/workspace/musco -v <datasets_dir>:/workspace/raid -p 4567:8888  jgusak/tensor_compression_od
```
In this example  ```/workspace/musco``` is the folder inside the container, where the content of the current repository will be stored (```PWD``` variable from ```model_utils/load_utils.py``` defines path to this folder). 

```/workspace/raid``` is a directory where datasets and models folders are stored at a docker container. 

```<datasets_dir>``` is a folder at a host machine where datasets and models are stored.

```-p 4567:8888``` a port 8888 from docker container is mapped to 4567 on a host machine by this option. If you want to map all ports from a docker container to the corresponding ports at a host machine use ```--net="host"``` instead.



  **Data preparation**
  
Prepare the dataset by storing it in  ```DATA_ROOT```  folder (```DATA_ROOT``` can be specified in ```model_utils/load_utils.py```).
 
 
  **Model compression**
  
Please set path to your working directory by changing ```PWD``` variable inside ```model_utils/load_utils.py```.
  
Pretrained models are needed. Please, download them and specify path to the pretrained models by setting ```PATH_TO_PRETRAINED``` variable from ```model_utils/load_utils.py```.
  
  Follow the instructions in ```demo/compression.ipynb``` to apply tensor decomposition to the selected convolutional layers. Compression can be applied to any custom loaded neural network.
  
  
  **Iterative compression**
  
  Notebook ```demo/iterative_finetuning.ipynb```  demonstrates how to perform iterative compression.
  
  Checkpoints are saved in  ```results``` folder in working directory (to modify default save path, go to ```model_utils/load_utils.py```  and change ```SAVE_ROOT``` variable).

## Citing
If you used our research, we kindly ask you to cite the corresponding [paper](https://arxiv.org/abs/1903.09973).

```
@article{gusak2019one,
  title={One time is not enough: iterative tensor decomposition for neural network compression},
  author={Gusak, Julia and Kholyavchenko, Maksym and Ponomarev, Evgeny and Markeeva, Larisa and Oseledets, Ivan and Cichocki, Andrzej},
  journal={arXiv preprint arXiv:1903.09973},
  year={2019}
}
```
