# Reproduction：AdaPtive Experience Sampling(APES)<br>
Original paper: https://ieeexplore.ieee.org/abstract/document/9832486?casa_token=gfMcjybRnTQAAAAA:rXl2teqGPw082-Gvvm-dPBdQwPHDywXZkqGmaXp47neKSJKcbL-PXTCzux-iGJ0foba35-YC<br><br>
>Based on the method of this article, we perform motion planning in two-dimensional space with the aim of generating a better sampling distribution to optimize the performance of the RRTConnect sampler(based on OMPL). 

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Using](#using)
- [Conclusion](#conclusion)


## Background<br>
>When a planner work baesd on RRTConnect, random sampling is an important part in this planner, Higher quality sampling can improve planner performance.because higher quality sampling enables the planner to find solution path with fewer iterations.When we know the instance of the robotic arm (start point, target point, and occupancy grid) we can transform the instance into the joint space (Figure 1)<br><br>
<div align=center>
<img src="https://github.com/KGWANG2049/apes/blob/main/png/obstacle.jpg" width="50%" height="50%">Figure1 — Instance in joint space
</div><br><br>

>the green area is clearly a better region to sample, so the problem is how to find the sutiable region, and make the region have a higher sampling probability than the other regions. When we give start, endpoint and the obstacle to APES, APES can generate this distribution, which is a Gaussian mixture model(GMM).

## Installation
Environment:

Install Ubuntu 20.04

Install miniconda

Install Nvidia-Cuda
1. Clean up the environment:
```sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt remove --autoremove nvidia-*
sudo rm -rf /usr/local/cuda*
sudo apt-get purge nvidia*
sudo apt-get update
sudo apt-get autoremove
sudo apt-get autoclean
sudo apt-key del 7fa2af80
```
2. Get Current Cuda - Example code - For actual version: 
```https://developer.nvidia.com/cuda-downloads``` 
can find
```wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.0
sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

3. Reboot

Setup PyTorch environment

1. Create a virtual environment for pytorch using conda
Remove the old environment if needed
```
conda remove --name pytorch_env
conda create -n pytorch_env -c pytorch pytorch=1.11 torchvision python=3.9
```
2. Activate the virtual environment
```
conda activate pytorch_env
```
3. Install the torch_geometric
```
conda install pyg -c pyg # -c for --channel
```
4. Test the installation in the python
```
python
from torch_geometric.data import Data
```
5. Install stable_baselines3, 1.6.1
```
https://spinningup.openai.com/en/latest/user/installation.html
```
6. install OMPL
```
https://ompl.kavrakilab.org/installation.html
```

## Structure

>It's the overall structure of APES, which has five important working parts(fig.2)

<div align=center>
<img src="https://github.com/KGWANG2049/apes/blob/main/png/frame%20work.jpg" width="50%" height="50%"> Figure2
</div><br><br>

>1. Generator(Fig.3) accepts instance and output fifty weights: coefficients Wi.
<div align=center>
<img src="https://github.com/KGWANG2049/apes/blob/main/png/Gen%20NN.jpg" width="50%" height="50%"> Figure3
</div><br><br>

>2. Assign coefficients Wi as weights to the 50 paths to form GMM. 50 paths have been prepared.(Fig.4)
<div align=center>
<img src="https://github.com/KGWANG2049/apes/blob/main/png/GMM.jpg" width="50%" height="50%"> Figure4
</div><br><br>

>3. transfer the gmm to planner, The planner can sample based on GMM, when planner successful planning a path, the total number of planner iterations is defined as the performance of the planner, in this paper (called value) and planner output value Vi to critic.<br>
>4. Through learning, the critic can estimate the value Vi: and out put value estimate V^.(Fig.5)

<div align=center>
<img src="https://github.com/KGWANG2049/apes/blob/main/png/Critic%20NN.jpg" width="50%" height="50%"> Figure5
</div><br><br>

>5. The critic passes the gradient information of value estimate to the generator, Through gradient information of value estimate V^, the generator can be optimized to output better coefficients in order to get a better GMM.(Fig.6)
<div align=center>
<img src="https://github.com/KGWANG2049/apes/blob/main/png/appro.jpg" width="50%" height="50%"> Figure6
</div><br><br>

## using
We need [planning.py](https://github.com/Xi-HHHM/mp2d/tree/gmm/scripts) and [utilities.py](https://github.com/Xi-HHHM/mp2d/tree/gmm/scripts) Thanks my supervisor  [__Xi-Huang__](https://github.com/Xi-HHHM) for guiding and for the functions available in planning and utilities.<br><br>

>Using [train.py](https://github.com/KGWANG2049/apes/blob/main/train.py) to train the neural network generator and critic, which are instantiated in [model.py](https://github.com/KGWANG2049/apes/blob/main/model.py)
> 1. After initialization, the entropy of GMM is about -144.6, so the target entropy should be smaller than it (recommended -180 ~ -230).<br>
> 2. A larger batchsize leads to more stable convergence of the generator (GMM entropy changes more slowly), and Epoch can be increased appropriately.<br>
> 3. If 2. leads to too long computation time, the entropy regularization factor log_alpha can be reduced appropriately (e.g., use -12 instead of -8).<br>
> 4. Learning rate: 1e-4 ~ 6e-4 is appropriate.<br><br>


## conclusion

>In order to visualize which regions have higher sampling probability, I make a random sampling of 2000 points under the GMM generated by APES. I found that the density of blue points is higher in the green region, which proves that the probability of sampling in this region is higher than the other regions, and APES got a GMM we wanted.
>In APES, the critic transfers the gradient information of the planner's VALUE to the generator, and then the generator can learn effectively and generate an effective GMM distribution.

<div align=center>
<img src="https://github.com/KGWANG2049/apes/blob/main/png/6.png" width="50%" height="50%"> result
</div>


