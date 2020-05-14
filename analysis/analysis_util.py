import numpy as np
from pathlib import Path
import json
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm


def get_errors(model, x_true, n_shifts):
    """
    Calculates error metrics along prediction time horizon

    Args:
        model: (NN) trained network
        x_true: (ndarray) true trajectories
        n_shifts: (int) number of time horizon to be evaluated
    Returns:
        x: (N ndarray) time axis
        mse: (N ndarray) mean squared error across the batch
        mad: (N ndarray) mean absolute deviation
        qt: (2 by N array) lower and upper quartiles for absolute errors
    """
    x_pred = model.predict(x_true, n_shifts)
    t_end = np.min((x_true.shape[1], x_pred.shape[1]))
    se = (x_pred[:, 1:t_end, :] - x_true[:, 1:t_end, :])**2
    se = np.mean(se, axis=-1)
    se = np.swapaxes(se, 0, 1)
    mse = np.mean(se, axis=1)
    ae = np.abs(x_pred[:, 1:t_end, :] - x_true[:, 1:t_end, :])
    ae = np.mean(ae, axis=-1)
    ae = np.swapaxes(ae, 0, 1)
    mad = np.mean(ae, axis=1)
    mad = np.median(ae, 1)
    x = np.arange(0, len(mad), 1)
    lq = np.quantile(ae, 0.25, 1)
    uq = np.quantile(ae, 0.75, 1)
    qt = np.array([lq, uq])
    return x, mse, mad, qt


def load_models(network, root_path, model_names):
    """
    Load a list of models (for experiment evaluations)
    Args:
        network: network class, can be DENIS or LREN
        root_path: (str) model root directory
        model_names: (lst) list of names
    Returns:
        models: list of models
        model_configs: list of model configurations
    """
    models = []
    model_configs = []
    for name in model_names:
        model_path = Path(root_path, "{}.pt".format(name))
        model_config_path = Path(root_path, "{}.json".format(name))
        with open(model_config_path) as f:
            model_config = json.load(f)
        model_configs.append(model_config)
        model = network(model_config)
        model.load_state_dict(torch.load(model_path))
        models.append(model)
    return models, model_configs


def plot_mse_mad(models, model_labels, x_true, n_shifts, t_end):
    mses, mads, mqts = [], [], []
    for model in tqdm(models):
        x, mse, mad, mqt =  get_errors(model, x_true, t_end)
        mses.append(mse)
        mads.append(mad)
        mqts.append(mqt)
    mses = np.array(mses)
    mads = np.array(mads)
    mqts = np.array(mqts)

    plt.rcParams.update({'font.size': 15})
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["grey", "purple", "k", "m", "orange"])


    fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=100, facecolor='white')
    for idx, label in enumerate(model_labels):
        p = ax[0].plot(x[:n_shifts], mses[idx, :n_shifts],
                       linewidth=2, label=label)
        ax[0].plot(x[n_shifts-1:], mses[idx, n_shifts-1:], '--',
                   color=p[0].get_color(), linewidth=2)

        ax[0].set_ylabel(r'MSE')


        p = ax[1].errorbar(x[:n_shifts], mads[idx, :n_shifts],
                   yerr=mqts[idx, :, :n_shifts], errorevery=len(x)//5,
                   capsize=2, capthick=2, label=label)

        ax[1].errorbar(x[n_shifts:], mads[idx, n_shifts:],
                       yerr=mqts[idx, :, n_shifts:], errorevery=len(x)//5,
                       capsize=2, capthick=2, ls='--', color=p[0].get_color())
        ax[1].set_ylabel(r'MAD')
    for i in range(2):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_xlabel(r'$k$')
        ax[i].set_xlim([1, t_end])

        ax[i].set_yscale('log')

    ax[0].legend()
    fig.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.3)
    plt.show()
    return x, mses, mads, mqts


def plot_trajectories(model, n_shifts, x_true, setlim=None):
    """
    Plot trajectory and phase plots for ground truth and prediction
    Args:
        model: (NN) trained network
        model_config: (dict) model configuration dictionary
        x_true: (ndarray) ground truth
        setlim: (float) axis limit (auto if left none)
    """
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(1, 3, figsize=(18, 4), facecolor='white', dpi=100)
    lw = 3
    x_pred = model.predict(x_true, x_true.shape[1])
    t_end = x_true.shape[1]

    time = np.arange(0, t_end, 1)

    for i in range(5):
        ax[2].plot(x_pred[i, :n_shifts, 0], x_pred[i, :n_shifts, 1], color='g', linewidth=lw)
        ax[2].plot(x_pred[i, n_shifts:t_end, 0], x_pred[i, n_shifts:t_end, 1], '--', color='g', linewidth=lw)
        ax[2].plot(x_true[i, :t_end, 0], x_true[i, :t_end, 1], '--', alpha=0.5, color='r', linewidth=lw)
        ax[2].scatter(x_pred[i, 0, 0], x_pred[i, 0, 1], color='g', s=50)
        ax[2].set_xlabel(r'$\theta$')
        ax[2].set_ylabel(r'$\omega$')
        ax[2].set_title(r'Phase Portrait', pad=20)

    for i in range(5):
        ax[0].plot(time[:n_shifts], x_pred[i, :n_shifts, 0], color='g', linewidth=lw)
        ax[0].plot(time[n_shifts:], x_pred[i, n_shifts:t_end, 0], '--', color='g', linewidth=lw)
        ax[0].plot(time, x_true[i, :t_end, 0], '--', alpha=0.5, color='r', linewidth=lw)
        ax[0].set_xlabel(r'$k$')
        ax[0].set_ylabel(r'$\theta$')
        ax[0].set_title(r'$\theta$ Trajectory', pad=20)

    for i in range(5):
        ax[1].plot(time[:n_shifts], x_pred[i, :n_shifts, 1], color='g', linewidth=lw)
        ax[1].plot(time[n_shifts:], x_pred[i, n_shifts:t_end, 1], '--', color='g', linewidth=lw)
        ax[1].plot(time, x_true[i, :t_end, 1], '--', alpha=0.5, color='r', linewidth=lw)
        ax[1].set_xlabel(r'$k$')
        ax[1].set_ylabel(r'$\omega$')
        ax[1].set_title(r'$\omega$ Trajectory', pad=20)

    if setlim is not None:
        ax[0].set_ylim([setlim, -setlim])
        ax[1].set_ylim([setlim, -setlim])
        ax[2].set_ylim([setlim, -setlim])
        ax[2].set_xlim([setlim, -setlim])
    for i in range(3):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

    fig.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.3)
    plt.show()
    return x_pred
