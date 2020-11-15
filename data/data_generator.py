import os

import numpy as np
import matplotlib.pyplot as plt
from data_helper_fns import *

def is_constraint_violated(x):
    x0, x1 = x
    potential = 0.5 * x1**2 - np.cos(x0)
    return potential > .99

def dynamics(y, t, u):
    x1, x2 = y
    dydt = [x2, -np.sin(x1) + u[1]]
    return dydt

if __name__ == "__main__":
    
    N = 25000
    n_steps = 50
    dt = 0.01
    generator = Data_Generator(pendulum_dynamics, dt, 2)

    params = {
        "theta": 2,
        "sigma": 0,
        'keep_zero': True,
        "max_u": [0, 1],
        "max_x": [3.14, 2.5]
    }

    data = generator.generate(N, n_steps, params, is_constraint_violated)
    data.save('raw_data/pendulum_large')