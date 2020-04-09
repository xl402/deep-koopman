import numpy as np
from numpy import trapz
from kooc_fns import KOOC
from ilqr_fns import ILQR
from tqdm import tqdm

def get_u_cost(us, t):
    """Calculate total controller cost
    \int \|u(t)\|_2  dt

    Args:
        us: (array) controller inputs
        t: (array) timesteps corresponding to these controller inputs
    """
    us = np.array(us)
    if len(us.shape)==1:
        us = np.expand_dims(us, 1)
    return trapz(np.mean(us**2, -1), np.array(t))

def get_x_cost(xs, x_goal, t):
    """Calculate total state cost
    \int \|x(t)-x_goal\|_2  dt

    Args:
        xs: (array) system state at each timestep
        x_goal: (array) desired system state (reference)
        t: (array) timesteps corresponding to these states
    """
    if len(xs)!=len(t):
        # assume we ditch the initial condition
        xs = xs[1:]
    se = (xs - x_goal)**2
    mse = np.mean(se, axis=-1)
    return trapz(mse, t)

def kooc_eval(model, init_conds, x_goal, T, Qs, r=1):
    kooc = KOOC('pendulum', model, 0.1)
    x_cost_m, u_cost_m = [], []
    for Q in tqdm(Qs):
        x_kooc, u_kooc, t, t_kooc = kooc.simulate(init_conds, x_goal,
                                                  T, Q, r, show=False)
        x_costs = [get_x_cost(x, x_goal, t) for x in x_kooc]
        u_costs = [get_u_cost(u, t) for u in zip(u_kooc, t_kooc)]
        x_cost_m.append(np.median(x_costs))
        u_cost_m.append(np.median(u_costs))
    return np.array(x_cost_m), np.array(u_cost_m)

def ilqr_eval(init_conds, x_goal, T, Qs, r=1):
    ilqr = ILQR('pendulum', 0.1)
    x_cost_m, u_cost_m = [], []
    for Q in tqdm(Qs):
        x_ilqr, u_ilqr, t = ilqr.simulate(init_conds, x_goal, T, Q, r,
                                          show=False)
        x_costs = [get_x_cost(x, x_goal, t) for x in x_ilqr]
        u_costs = [get_u_cost(u, t) for u in u_ilqr]
        x_cost_m.append(np.median(x_costs))
        u_cost_m.append(np.median(u_costs))
    return np.array(x_cost_m), np.array(u_cost_m)


def animate(labels, fn, *args, interval=50):
    """Animates 4x4 dufing or pendulum trajectories across different models
    Args:
        labels: (lsft of strings) model labels
        fn: (std) full path to file to be saved
        *args: trajectories (batch x time x state dimension) to be animated
        interval: (int) animation interval
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(4, 4, figsize=(20, 20), facecolor='w')
    plt.rcParams['font.size'] = 15
    t = args[0].shape[1]
    colors = ['k', 'r', 'b', 'c','g']

    for i in range(4):
        for j in range(4):
            ax[i, j].set_xlim([-1, 1])
            ax[i, j].set_ylim([-2, 2])
            ax[i, j].set_yticks([])
            ax[i, j].set_xticks([])

    lines = []
    lns = [[] for i in range(t)]

    for i in range(4):
        for j in range(4):
            idx = j+i*4
            for k in range(t):
                for m, arg in enumerate(args):
                    ln, = ax[i, j].plot([0, np.sin(arg[idx, k, 0])],
                                [0, -np.cos(arg[idx, k, 0])], color=colors[m], lw=4)
                    lns[k].append(ln)

                    if idx==15 and k==t-1:
                        lines.append(ln)
            ax[i, j].set_aspect('equal', 'datalim')
    plt.figlegend(lines, labels, loc = 'upper left', labelspacing=0. , fontsize=30)
    ani = animation.ArtistAnimation(fig, lns, interval=interval, repeat=True)
    ani.save(fn+'.mp4',writer='ffmpeg')
    plt.close()
