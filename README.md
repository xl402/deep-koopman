# Deep Koopman
*CUED IIB Master's Thesis Project*

## Introduction: 
The Koopman operator framework is becoming increasingly popular for obtaining linear representations of nonlinear systems from data. This project aims to **optimally control input non-affine nonlinear systems**, utilizing Deep Learning (DL) to discover the Koopman invariant subspace, bridging the gap between **DL based Koopman eigenfunction discovery and optimal predictive control.**

## Networks Overview:
Script `models/networks.py` contains all networks discussed in the thesis, including:

**LREN** : **L**inearly **R**ecurrent **E**ncoder **N**etwork

**DENIS**: **D**eep **E**ncoder with **I**nitial **S**tate Parameterisation

**DEINA**: **D**eep **E**ncoder for **I**nput **N**on-**A**ffine systems

 
|               **LREN**               |               **DENIS**              |               **DEINA**              |
|:------------------------------------:|:------------------------------------:|:------------------------------------:|
| ![](https://i.imgur.com/zk0sbWV.png) | ![](https://i.imgur.com/dTgpnbo.png) | ![](https://i.imgur.com/4lvGkWC.png) |

## Koopman Operator Optimal Control
By lifting system state dimensions, system dynamics become globally linear, where LQR is readily applied. This technique is compared against the iterative LQR (iLQR). Video below shows our models controlling a pendulum it its vertical upright position.

**Ours VS. iLQR**            |  **Effect of Latent Size**
:-------------------------:|:-------------------------:
![](https://i.imgur.com/cEslwIS.gif)  |  ![](https://i.imgur.com/c0X2hVD.gif)


## Pendulum Example
Left: Predicted trajectories overlaying ground truth. Right: Top two Koopman eigenfunctions magnitudes (which together, convey the Hamiltonian energy of the system).

![Pendulum2](https://i.imgur.com/j83vGxn.gif)

## Training
To train the network, first configure a training json file inside `training\configs`. Then inside `training` directory, open terminal:

`python trainer.py --config_dir PATH_TO_CONFIG --viz`

This should locally initialize a model, trained with data specified in the configuration file.


