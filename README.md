# Reproduction：AdaPtive Experience Sampling(APES)<br>
Original paper: https://ieeexplore.ieee.org/abstract/document/9832486?casa_token=gfMcjybRnTQAAAAA:rXl2teqGPw082-Gvvm-dPBdQwPHDywXZkqGmaXp47neKSJKcbL-PXTCzux-iGJ0foba35-YC<br><br>
>Based on the method of this article, we perform motion planning in two-dimensional space with the aim of generating a better sampling distribution to optimize the performance of the RRTConnect sampler(based on OMPL). 

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [如何使用](#如何使用)
- [结论](#结论)

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

说明如何使用该项目，可以包括安装、配置、使用说明和示例代码等。这一部分应该尽可能详细，以便新用户可以快速上手。

## 结论

总结该项目的一些结果，可以包括项目的收获、成果和不足之处。还可以列出该项目的未来规划和改进方向。

## 贡献者

列出该项目的贡献者，感谢他们的付出和贡献。可以包括他们的 GitHub 链接或其他联系方式。

## 许可证

说明该项目的许可证信息，以及如何使用该项目的代码、文档和其他资源。可以包括版权信息和授权说明。

## 联系我们

如果您对该项目有任何疑问、建议或反馈，请在此部分提供联系方式，以便与开发团队进行沟通。可以包括电子邮件、Slack、论坛等等。
