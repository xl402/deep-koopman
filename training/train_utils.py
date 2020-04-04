import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def count_parameters(net):
    from functools import reduce
    from collections import defaultdict
    import pandas as pd
    p = list(net.parameters())
    total = sum([reduce(lambda x, y: x*y, i.shape) for i in p])

    inp = [(i[0].split('_')[0], reduce(lambda x, y: x*y, i[1].shape)) for i in list(net.named_parameters())]
    output = defaultdict(int)
    for k, v in inp:
        output[k] += v

    df = pd.DataFrame(dict(output), index=[0])
    df['total'] = total
    return df


def affine_model_configurer(config):
    model_config = {"enc_shape": config["enc_shape"], "aux_shape": config.get("aux_shape"),
         "n_shifts": config["n_shifts"], "dt": config.get("dt"),
         "use_rbf": config.get("use_rbf"), "drop_prob": config.get("drop_prob"),
         "aux_rbf": config.get("aux_rbf")}
    return model_config


def get_lr_scheduler(optimizer, config):
    step = config.get('lr_sch_step') or 1000
    gamma = config.get('lr_sch_gamma') or 0.1
    return optim.lr_scheduler.StepLR(optimizer, step, gamma=gamma)


def get_batch(data, btach_size, config):
    batch = data[torch.randint(len(data), size=(btach_size,))]
    if config['zero_loss'] != 0.:
        zero = torch.zeros((1, *batch.shape[1:]))
        batch = torch.cat((zero, batch), dim=0)
    return batch

def get_l2_weights(model):
    l2_reg = 0.
    for W in model.parameters():
        l2_reg = l2_reg + W.norm(2)
    return l2_reg

def koopman_loss(xy, xy_pred, config, model):
    dim = config['enc_shape'][0]
    x, x_pred = xy[:, :, :dim], xy_pred[:, :, :dim]
    y, y_pred = xy[:, :, dim:], xy_pred[:, :, dim:]
    loss = 0.
    idx = 0
    zero_loss = 0.
    if config['zero_loss'] != 0.:
        zero_loss = torch.mean((xy[0] - xy_pred[0])**2)
        idx = 1
    x_error = x[idx:] - x_pred[idx:]
    y_error = y[idx:] - y_pred[idx:]
    x_mse = torch.mean(x_error**2)
    y_mse = torch.mean(y_error**2)
    min_idx = np.min((config['n_shifts'], 5))
    x_mse_1 = torch.mean(x_error**2, dim=-1)[:, :int(min_idx)]
    x_inf_mse = torch.mean(x_mse_1.norm(p=float('inf'), dim=-1))
    reg_loss = get_l2_weights(model)

    loss += config['state_loss'] * x_mse    \
          + config['latent_loss'] * y_mse   \
          + config['inf_loss'] * x_inf_mse  \
          + config['zero_loss'] * zero_loss \
          + config['reg_loss'] * reg_loss

    r_arry = np.array([loss.detach().numpy(),
              x_mse.detach().numpy(),
              y_mse.detach().numpy(),
              x_inf_mse.detach().numpy(),
              zero_loss.detach().numpy(),
              reg_loss.detach().numpy()])
    return loss, r_arry.astype(np.double())


def print_metrics(train_metrics_mean, val_metrics_mean, epoch, lr):
    print('{{"metric": "Train Loss", "value": {}, "epoch": {}}}'.format(
        train_metrics_mean[0], epoch))
    print('{{"metric": "Val Loss", "value": {}, "epoch": {}}}'.format(
            val_metrics_mean[0], epoch))
    print('{{"metric": "Val State MSE", "value": {}, "epoch": {}}}'.format(
            train_metrics_mean[1], epoch))
    print('{{"metric": "Val Latent MSE", "value": {}, "epoch": {}}}'.format(
            val_metrics_mean[2], epoch))
    print('{{"metric": "Val Infinity MSE", "value": {}, "epoch": {}}}'.format(
            val_metrics_mean[3], epoch))
    print('{{"metric": "Val Zeros MSE", "value": {}, "epoch": {}}}'.format(
            val_metrics_mean[4], epoch))
    print('{{"metric": "Regularization", "value": {}, "epoch": {}}}'.format(
            val_metrics_mean[5], epoch))
    print('{{"metric": "Learning Rate", "value": {}, "epoch": {}}}'.format(
            lr, epoch))
