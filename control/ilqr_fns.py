import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from ilqr import iLQR
from ilqr.dynamics import AutoDiffDynamics
from ilqr.dynamics import BatchAutoDiffDynamics
from ilqr.cost import QRCost
from tqdm import tqdm

def ilqr_duffing(dt=0.1):
    x = T.dscalar("x")  # Position.
    x_dot = T.dscalar("x_dot")  # Velocity.
    u = T.dscalar("u")

    x_dot_dot = x - x**3 - u

    f = T.stack([
        x + (x_dot) * dt,
        x_dot + x_dot_dot * dt,
    ])

    x_inputs = [x, x_dot]
    u_inputs = [u]

    dynamics = AutoDiffDynamics(f, x_inputs, u_inputs)
    return dynamics


def ilqr_pend(dt=0.1):
    x = T.dscalar("x")  # Position.
    x_dot = T.dscalar("x_dot")  # Velocity.
    u = T.dscalar("u")  # Force.
    x_dot_dot = -np.sin(x) - u

    f = T.stack([
        x + (x_dot) * dt,
        x_dot + x_dot_dot * dt,
    ])

    x_inputs = [x, x_dot]  # State vector.
    u_inputs = [u]  # Control vector.

    dynamics = AutoDiffDynamics(f, x_inputs, u_inputs)
    return dynamics

class ILQR():
    def __init__(self, dynamics, dt):
        if dynamics == 'pendulum':
            self.dynamics = ilqr_pend(dt)
        elif dynamics == 'duffing':
            self.dynamics = ilqr_duffing(dt)
        else:
            print("Unknown dynamics")
            raise
        self.dt = dt

    def simulate(self, init, x_goal, T, Q=1, R=1, show=True):
        cost = global_cost(x_goal, Q, R)
        assert len(init.shape) == 3, "Must batch initial conditions"
        xses, uses = [], []
        if show:
            p_bar = tqdm(range(init.shape[0]))
        else:
            p_bar = range(init.shape[0])
        for i in p_bar:
            x0 = init[i]
            us_init = np.random.uniform(-1, 1, (T, 1))
            ilqr = iLQR(self.dynamics, cost, T)
            xs, us = ilqr.fit(x0, us_init)
            t = np.arange(0, self.dt*T, self.dt)
            xses.append(xs)
            uses.append(us)
        xses = np.array(xses)
        uses = np.array(uses)
        return xses, uses, t


def global_cost(x_goal = [0.0, 0.0], Q=1, R=1):
    if np.isscalar(Q):
        Q = Q * np.eye(2)
    else:
        Q = np.diag(Q)
    R = R * np.eye(1)
    x_goal = np.array(x_goal)
    cost = QRCost(Q, R, x_goal=x_goal)
    return cost
