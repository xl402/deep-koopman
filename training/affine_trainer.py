#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from datetime import datetime
import json
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy import linalg as LA
sys.path.append('model/')
sys.path.append('../model/')
from networks import *
from train_utils import *
from tqdm import tqdm


name2model = {"lren": LREN, "denis": DENIS, "denis_jbf": DENIS_JBF}

parser = argparse.ArgumentParser("DKoopman", description = 'Training Affine Models')
parser.add_argument('--config_dir', type=str, help='config file directory')
parser.add_argument('--name', type=str, help='model name', default='model')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--viz', action='store_true',
                    help='visualize the process')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--val_feq', type=int, default=4,
                    help='number of times to validate per epoch')
parser.add_argument('--dump_dir', type=str, default='../saved/logs',
                    help='directory to save outputs')
parser.add_argument('--v', action='store_true',
                    help='verbosity')
parser.add_argument('--random', action='store_true',
                    help='randomize visualization')
parser.add_argument('--load_weights', type=str, default=None,
                    help='dir for a model checkpoint (.pt file)')
parser.add_argument('--record', action='store_true',
                    help='save figure at every iteration')
args = parser.parse_args()

now = datetime.now().strftime("%m%d-%H-%M-%S%f")
EPOCHS = args.epochs
MODEL_NAME = args.name + "-{}".format(now)
BATCH_SIZE = args.batch_size
VAL_FEQ = args.val_feq
DUMP_DIR = args.dump_dir
VERBOSE = args.v
VIS = args.viz
DEVICE = 'cpu'

SUMMARY_DIR = "{}/summary.csv".format(args.dump_dir)
MODEL_DIR = "{}/models".format(args.dump_dir)
FIGURE_DIR = "{}/figures".format(args.dump_dir)

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

if not os.path.exists(FIGURE_DIR):
    os.mkdir(FIGURE_DIR)

if args.viz:
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    plt.rcParams.update({'font.size': 10})
    ax_phase = fig.add_subplot(332, frameon=False)
    ax_traj = fig.add_subplot(331, frameon=False)
    ax_x_mse = fig.add_subplot(334, frameon=False)
    ax_y_mse = fig.add_subplot(335, frameon=False)
    ax_x_inf = fig.add_subplot(336, frameon=False)
    ax_0_mse = fig.add_subplot(337, frameon=False)
    ax_reg = fig.add_subplot(338, frameon=False)
    ax_ko = fig.add_subplot(333, frameon=False)
    ax_lr = fig.add_subplot(339, frameon=False)
    plt.show(block=False)

def summary_writer(train_val, val_val, lr):
    if not os.path.isfile(SUMMARY_DIR):
        df = pd.DataFrame(columns=['model_name', 'iter', 'lr', 'train_loss',
                                   'state_loss_train', 'latent_loss_train',
                                   'inf_loss_train', 'zero_loss_train', 'reg_loss_train',
                                   'val_loss' ,'state_loss_val', 'latent_loss_val',
                                   'inf_loss_val', 'zero_loss_val', 'reg_loss_val'])

        df.to_csv(SUMMARY_DIR, index=False)
    df = pd.read_csv(SUMMARY_DIR)
    if not df['model_name'].str.contains(MODEL_NAME).any():
        df.loc[len(df)+1] = [MODEL_NAME, 0, lr, *train_val, *val_val]
    else:
        idx = df[df['model_name']==MODEL_NAME]['iter'].max()
        df.loc[len(df)+1] = [MODEL_NAME, idx+1, lr, *train_val, *val_val]
    df.to_csv(SUMMARY_DIR, index=False)

def save_model(val_loss):
    #saves model weights if validation loss is the lowest ever
    df = pd.read_csv(SUMMARY_DIR)
    min_val = np.min(df[df['model_name']==MODEL_NAME]['val_loss'])
    if val_loss <= min_val:
        model_dir = os.path.join(MODEL_DIR, "{}.pt".format(MODEL_NAME))
        torch.save(model.state_dict(), model_dir)
        json_dir = os.path.join(MODEL_DIR, "{}.json".format(MODEL_NAME))
        with open(json_dir, 'w') as fp:
            json.dump(config, fp)

def zero_pad(data):
    # pad first location with zero input (for calculating zero loss)
    zero = torch.zeros((1, *data.shape[1:])).to(DEVICE)
    data_w_zeros = torch.cat((zero, data), dim=0)
    return data_w_zeros

def visualize(model, data, iter):
    if args.viz:
        df = pd.read_csv(SUMMARY_DIR)
        df = df[df['model_name']==MODEL_NAME]
        with torch.no_grad():
            outputs = model.forward(data, return_ko=True)
        xy_gt, xy_pred, kos = outputs
        xy_gt, xy_pred = xy_gt.numpy(), xy_pred.numpy()
        kos = kos.numpy()

        random = args.random

        idx = np.random.randint(len(xy_gt)) if random else 1
        ax_traj.cla()
        ax_traj.plot(xy_gt[idx, :, 0], '--', color='r', linewidth=2, alpha=0.5)
        ax_traj.plot(xy_pred[idx, :, 0], color='r', linewidth=2)
        ax_traj.plot(xy_gt[idx, :, 1], '--', color='g', linewidth=2, alpha=0.5)
        ax_traj.plot(xy_pred[idx, :, 1], color='g', linewidth=2)
        ax_traj.set_ylim(np.min(xy_gt)*1.1, np.max(xy_gt)*1.1)
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('Time Steps')
        ax_traj.set_ylabel(r'$\mathbf{x}$')

        ax_phase.cla()
        idicies = np.random.randint(len(xy_gt), size=10) if random else range(10)
        for i in idicies:
            ax_phase.plot(xy_gt[i, :, 0], xy_gt[i, :, 1], '--', color='r', linewidth=2, alpha=0.5)
            ax_phase.plot(xy_pred[i, :, 0], xy_pred[i, :, 1], color='g', linewidth=2)
            ax_phase.scatter(xy_gt[i, 0, 0], xy_gt[i, 0, 1], color='g', s=15)
        ax_phase.spines['top'].set_visible(False)
        ax_phase.spines['right'].set_visible(False)
        ax_phase.set_xlim(np.min(xy_gt[:, :, 0]*1.1), np.max(xy_gt[:, :, 0]*1.1))
        ax_phase.set_ylim(np.min(xy_gt[:, :, 1]*1.1), np.max(xy_gt[:, :, 1]*1.1))
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel(r'$x_1$')
        ax_phase.set_ylabel(r'$x_2$')

        x_axis = np.array(df['iter'])
        x_mse_train = np.array(df['state_loss_train'])
        x_mse_val = np.array(df['state_loss_val'])
        y_mse_train = np.array(df['latent_loss_train'])
        y_mse_val = np.array(df['latent_loss_val'])
        x_inf_train = np.array(df['inf_loss_train'])
        x_inf_val = np.array(df['inf_loss_val'])
        zero_train = np.array(df['zero_loss_train'])
        zero_val = np.array(df['zero_loss_val'])
        reg_train = np.array(df['reg_loss_train'])
        reg_val = np.array(df['reg_loss_val'])
        reg_val = np.array(df['reg_loss_val'])
        lrs = np.array(df['lr'])

        ax_x_mse.cla()
        ax_x_mse.plot(x_axis, x_mse_train, label='train', color='r', linewidth=2)
        ax_x_mse.scatter(x_axis[-1], x_mse_train[-1], color='r', s=25)
        ax_x_mse.plot(x_axis, x_mse_val, label='val', color='g', linewidth=2)
        ax_x_mse.scatter(x_axis[-1], x_mse_val[-1], color='g', s=25)
        ax_x_mse.set_title('State MSE')
        ax_x_mse.set_yscale('log')
        ax_x_mse.set_xlabel('Iterations')
        ax_x_mse.set_ylim([None, np.min((np.max(x_mse_train), 2))])
        ax_x_mse.legend()

        ax_y_mse.cla()
        ax_y_mse.plot(x_axis, y_mse_train, label='train', color='r', linewidth=2)
        ax_y_mse.scatter(x_axis[-1], y_mse_train[-1], s=25, color='r')
        ax_y_mse.plot(x_axis, y_mse_val, label='val', color='g', linewidth=2)
        ax_y_mse.scatter(x_axis[-1], y_mse_val[-1], s=25, color='g')
        ax_y_mse.set_title('Latent MSE')
        ax_y_mse.set_yscale('log')
        ax_y_mse.set_xlabel('Iterations')
        ax_y_mse.set_ylim([None, np.min((np.max(y_mse_train), 1))])
        ax_y_mse.legend()

        ax_x_inf.cla()
        ax_x_inf.plot(x_axis, x_inf_train, label='train', color='r', linewidth=2)
        ax_x_inf.scatter(x_axis[-1], x_inf_train[-1], s=25, color='r')
        ax_x_inf.plot(x_axis, x_inf_val, label='val', color='g', linewidth=2)
        ax_x_inf.scatter(x_axis[-1], x_inf_val[-1], s=25, color='g')
        ax_x_inf.set_title('Max State Deviation')
        ax_x_inf.set_yscale('log')
        ax_x_inf.set_xlabel('Iterations')
        ax_x_inf.legend()

        ax_0_mse.cla()
        ax_0_mse.plot(x_axis, zero_train, label='train', color='r', linewidth=2)
        ax_0_mse.scatter(x_axis[-1], zero_train[-1], s=25, color='r')
        ax_0_mse.plot(x_axis, zero_val, label='val', color='g', linewidth=2)
        ax_0_mse.scatter(x_axis[-1], zero_val[-1], s=25, color='g')
        ax_0_mse.set_title('Zero Loss')
        ax_0_mse.set_yscale('log')
        ax_0_mse.set_xlabel('Iterations')
        ax_0_mse.legend()
        ax_0_mse.set_ylim([np.min(zero_train)+1e-11, None])

        ax_reg.cla()
        ax_reg.plot(x_axis, reg_train, label='train', color='r', linewidth=2)
        ax_reg.scatter(x_axis[-1], reg_train[-1], s=25, color='r')
        ax_reg.plot(x_axis, reg_val, label='val', color='g', linewidth=2)
        ax_reg.scatter(x_axis[-1], reg_val[-1], s=25, color='g')
        ax_reg.set_title('Regularization Loss')
        ax_reg.set_yscale('log')
        ax_reg.set_xlabel('Iterations')
        ax_reg.legend()

        ax_lr.cla()
        ax_lr.plot(x_axis, lrs, color='r', linewidth=2)
        ax_lr.set_title('Learning Rate')
        ax_lr.set_xlabel('Iterations')

        idx = np.random.randint(len(xy_gt)) if random else 1
        w, v = LA.eig(kos[idx])
        ax_ko.cla()
        ax_ko.scatter(np.real(w), np.imag(w), marker='+', color='g', s=30)
        ax_ko.set_title('Koopman Eigenvalues')
        ax_ko.set_xlabel(r'$\mathcal{R}(\lambda)$')
        ax_ko.set_ylabel(r'$\mathcal{I}(\lambda)$')

        fig.tight_layout()
        plt.suptitle("{}:\n{}".format(MODEL_NAME, config["description"]))

        plt.subplots_adjust(top=0.9, hspace=0.5, wspace=0.5)
        if args.record:
            plt.savefig("{}/{}_{}.png".format(FIGURE_DIR, MODEL_NAME, iter))
        else:
            plt.savefig("{}/{}.png".format(FIGURE_DIR, MODEL_NAME))
        plt.draw()
        plt.pause(0.001)


if __name__ == "__main__":
    with open(args.config_dir, 'r') as fp:
        config = json.load(fp)

    model_config = affine_model_configurer(config)
    model = name2model[config["network"]](model_config)
    if args.load_weights is not None:
        model.load_state_dict(torch.load(args.load_weights))
        print("All keys match, loading from {}\n".format(args.load_weights))
    if args.v:
        print("Entering training loop\n")
        print("Model configuration: {}\n".format(model_config))
        print(count_parameters(model))
        print("\n")


    optimizer = optim.Adam(model.parameters(), config['lr'])
    lr_scheduler = get_lr_scheduler(optimizer, config)

    train_x = torch.Tensor(np.load(config['train_data']))
    val_x = torch.Tensor(np.load(config['val_data']))
    train_x = train_x.to(DEVICE)
    val_x = val_x.to(DEVICE)

    N = len(train_x)
    for epoch in range(EPOCHS):
        # dont show tqdm on floydhub
        if config.get("mode")!="floyd":
            pbar = tqdm(range(0, N, BATCH_SIZE))
        else:
            pbar = range(0, N, BATCH_SIZE)
        # permute training samples across each epoch
        permutation = torch.randperm(N)

        # training starts
        for idx, i in enumerate(pbar):
            train_metrics_mean = []
            train_loss_mean = []
            model.zero_grad()
            indices = permutation[i:i+BATCH_SIZE]
            batch_x = train_x[indices]
            if config['zero_loss'] != 0.:
                batch_x = zero_pad(batch_x)

            outputs = model(batch_x)
            loss, loss_array = koopman_loss(*outputs, config, model)
            train_metrics_mean.append(loss_array)
            train_loss_mean.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()

            # validation starts
            if (idx+1) % (len(pbar) // VAL_FEQ) == 0:
                model.eval()
                val_metrics_mean = []
                val_loss_mean = []
                with torch.no_grad():
                    # batching validation set
                    for j in range(0, len(val_x), BATCH_SIZE):
                        batch_val = val_x[j: j+BATCH_SIZE]
                        if config['zero_loss'] != 0.:
                            batch_val = zero_pad(batch_val)
                        val_outputs = model(batch_val)
                        val_loss, val_loss_array = koopman_loss(*val_outputs,
                                                                config, model)
                        val_metrics_mean.append(val_loss_array)
                        val_loss_mean.append(val_loss.numpy())
                # average loss components across validation batches
                train_metrics_mean = np.mean(np.array(train_metrics_mean), axis=0)
                val_metrics_mean = np.mean(np.array(val_metrics_mean), axis=0)
                val_loss_mean = np.mean(val_loss_mean)
                train_loss_mean = np.mean(train_loss_mean)
                lr = lr_scheduler.get_lr()[0]
                summary_writer(train_metrics_mean, val_metrics_mean, lr)
                save_model(val_metrics_mean[0])
                visualize(model, batch_val, epoch)

                # update progress bar
                desc = "loss: {:.4f}/{:.4f}, state_mse: {:.4f}/{:.4f}".format(
                                                        train_loss_mean,
                                                        val_loss.numpy(),
                                                        train_metrics_mean[0],
                                                        val_metrics_mean[0])
                if  config.get("mode")!="floyd":
                    pbar.set_description(desc)
                    pbar.refresh()

        if  config.get("mode")=="floyd":
            print_metrics(train_metrics_mean, val_metrics_mean, epoch, lr)

        lr_scheduler.step()
