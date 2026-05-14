# PACENet
This repository includes the official implementation our paper “PACEAttention: Principled and Adaptive Feature Compression-Expansion Grounded in the Geometry of $\text{MCR}^2$”[ICML 2026, [paper link](https://openreview.net/forum?id=O64UziRDnr)]

Inspired by the **geometric interpretation** of the gradients of the maximal coding rate reduction ($\text{MCR}^2$) objective, we propose a principled and interpretable attention mechanism that facilitates feature compression and expansion.

# Lay Summary
Deep neural networks are often powerful but difficult to interpret, making it unclear why they work well or how their architectures should be designed. In this work, we propose PACEAttention, a new attention mechanism motivated by geometric principles from representation learning theory.

Our method views feature learning as a progressive process that compresses features belonging to the same category while separating features from different categories in a high-dimensional space. This process is guided by the intrinsic structure of the data, which we capture through a randomized mechanism based on random matrices.

The resulting PACENet is both principled and efficient. The introduced randomized mechanism naturally guides feature updates within a low-dimensional subspace induced by the data structure, enabling linear computational complexity while preserving strong representation capability.

# One Layer of PACENet
![df](figs/FrameworkofECA6_1.jpg)

<!-- # Reference
Please consider citing our work if you find it helpful to yours:
```
@inproceedings{
anonymous2026paceattention,
title={{PACEA}ttention: Principled and Adaptive Feature Compression-Expansion Grounded in the Geometry of \${\textbackslash}text\{{MCR}\}{\textasciicircum}2\$},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=O64UziRDnr}
}
``` -->