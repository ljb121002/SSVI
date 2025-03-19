## Training Bayesian Neural Networks with Sparse Subspace Variational Inference

This repository contains code for the paper [Training Bayesian Neural Networks with Sparse Subspace Variational Inference](https://arxiv.org/abs/2402.11025), accepted in International Conference on Learning Representations (ICLR), 2024.

```
@inproceedings{litraining,
  title={Training Bayesian Neural Networks with Sparse Subspace Variational Inference},
  author={Li, Junbo and Miao, Zichen and Qiu, Qiang and Zhang, Ruqi},
  booktitle={The Twelfth International Conference on Learning Representations}
}
```

### Introduction

Bayesian neural networks (BNNs) offer uncertainty quantification but come with the downside of substantially increased training and inference costs. Sparse BNNs have been investigated for efficient inference, typically by either slowly introducing sparsity throughout the training or by post-training compression of dense BNNs. The dilemma of how to cut down massive training costs remains, particularly given the requirement to learn about the uncertainty. To solve this challenge, we introduce Sparse Subspace Variational Inference (SSVI), the first fully sparse BNN framework that maintains a consistently highly sparse Bayesian model throughout the training and inference phases. Starting from a randomly initialized low-dimensional sparse subspace, our approach alternately optimizes the sparse subspace basis selection and its associated parameters. While basis selection is characterized as a non-differentiable problem, we approximate the optimal solution with a removal-and-addition strategy, guided by novel criteria based on weight distribution statistics. Our extensive experiments show that SSVI sets new benchmarks in crafting sparse BNNs, achieving, for instance, a 10-20x compression in model size with under 3\% performance drop, and up to 20x FLOPs reduction during training compared with dense VI training. Remarkably, SSVI also demonstrates enhanced robustness to hyperparameters, reducing the need for intricate tuning in VI and occasionally even surpassing VI-trained dense BNNs on both accuracy and uncertainty metrics.

### Dependencies
```shell
pip install -r requirements
```

### Training

The following script will run SSVI on CIFAR-10 dataset. The default criteria is ```SNR_mean_abs```.
 
```
bash ./scripts/ssvi_cifar.sh
```
