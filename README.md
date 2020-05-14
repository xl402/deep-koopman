# Deep Koopman
*CUED IIB Master's Thesis Project*

## Introduction: 
The Koopman operator framework is becoming increasingly popular for obtaining linear representations of nonlinear systems from data. This project aims to **optimally control input non-affine nonlinear systems**, utilizing Deep Learning (DL) to discover the Koopman invariant subspace, bridging the gap between **DL based Koopman eigenfunction discovery and optimal predictive control.**

## Networks Overview:
Script `models/networks.py` contains all networks discussed in the thesis, including:
- **LREN**: **L**inearly **R**ecurrent **E**ncoder **N**etwork
- **DENIS**: **D**eep **E**ncoder with **I**nitial **S**tate Parameterisation
- **DEINA**: **D**eep **E**ncoder for **I**nput **N**on-**A**ffine systems

## Koopman Operator Optimal Control
By lifting system state timensions, system dynamics become globally linear, where LQR is readily applied. This technique is compared against iterative local linearization approach, i.e. Iterative LQR (iLQR). Hamiltonian energy increases from top left to bottom right.

<img src="https://i.imgur.com/jy9Knpv.gif" width="700"/>

## Pendulum Example
Left: Predicted trajectories overlaying ground truth. Right: Top two Koopman eigenfunctions magnitudes (which together, convey the Hamiltonian energy of the system).

![Pendulum2](https://i.imgur.com/j83vGxn.gif)

## Fluid Flow Example
Left: Predicted trajectories overlaying ground truth. Right: Top Koopman eigenfunction magnitude and phase plot.

<img src="https://i.imgur.com/5MuBOFo.gif" width="280"/> <img src="https://i.imgur.com/Y35ktWl.gif" width="570"/> 

