




The network_2d and APES_2D_train are used to test the performance of the 3D occupancy map simplified to 2D in the paper, 
after which the training set and model will be updated for 3D.


The re-parametric sampling implementation method can be found in:

https://arxiv.org/abs/2011.03813

note:https://sprout-cheque-498.notion.site/Algorithms-38558127bfd54a6da52fe0b28728ec31

# Reproduction：AdaPtive Experience Sampling(APES)



Original paper: https://ieeexplore.ieee.org/abstract/document/9832486?casa_token=gfMcjybRnTQAAAAA:rXl2teqGPw082-Gvvm-dPBdQwPHDywXZkqGmaXp47neKSJKcbL-PXTCzux-iGJ0foba35-YC

>Based on the method of this article, we perform motion planning in two-dimensional space with the aim of generating a better sampling distribution to optimize the performance of the RRTConnect sampler(based on OMPL). 

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [如何使用](#如何使用)
- [结论](#结论)

## Background

When a planner work baesd on RRTConnect, random sampling is an important part in this planner, Higher quality sampling can improve planner performance.because higher quality sampling enables the planner to find solution path with fewer iterations.When we know the instance of the robotic arm (start point, target point, and occupancy grid) we can transform the instance into the joint space (Figure 1)
<div style="text-align:center">
  <img src="图片路径" alt="图片描述" width="宽度">
</div>



## Installation
Environment:

Install Ubuntu 20.04

Install miniconda

Install Nvidia-Cuda

描述项目的具体结构，可以包括文件和文件夹的布局、项目中包含的组件、模块、库等等。此外，还可以描述项目的开发进度和项目的未来规划。

## 如何使用

说明如何使用该项目，可以包括安装、配置、使用说明和示例代码等。这一部分应该尽可能详细，以便新用户可以快速上手。

## 结论

总结该项目的一些结果，可以包括项目的收获、成果和不足之处。还可以列出该项目的未来规划和改进方向。

## 贡献者

列出该项目的贡献者，感谢他们的付出和贡献。可以包括他们的 GitHub 链接或其他联系方式。

## 许可证

说明该项目的许可证信息，以及如何使用该项目的代码、文档和其他资源。可以包括版权信息和授权说明。

## 联系我们

如果您对该项目有任何疑问、建议或反馈，请在此部分提供联系方式，以便与开发团队进行沟通。可以包括电子邮件、Slack、论坛等等。
