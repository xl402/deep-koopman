import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dataclasses import dataclass
import torch

@dataclass
class KoopmanData:
    us: np.ndarray
    xs: np.ndarray
        
    def shuffle(self):
        p = np.random.permutation(len(self.xs))
        return self.xs[p], self.us[p]
    
    def train_test_split(self, ratio=0.7, return_numpy=False):
        xs, us = self.shuffle()
        idx = int(len(xs)*ratio)
        train_x, train_u = xs[:idx], us[:idx]
        test_x, test_u = xs[idx:], us[idx:]
        return torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_u, dtype=torch.float32), torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_u, dtype=torch.float32)
        
    def save(self, f_name):
        np.save("{}_inputs.npy".format(f_name), self.us)
        np.save("{}_traj.npy".format(f_name), self.xs)
        
    def load(self, f_name):
        self.xs = np.load("{}_traj.npy".format(f_name))
        self.us = np.load("{}_inputs.npy".format(f_name))

        
class Data_Generator:
    """
    Generates driven dynamics using a ODE solver
    
    Args:
        gradient: function compatable with the scipy's ODE solver,
                  i.e. taking in (x, t, u), and returns dydt
        dt: sampling time
        dim: system's dimension
        
    Methods: 
        generate: runs a set of OU process simulations and returns
                  a Koopman dataclass
    """
    
    def __init__(self, gradient, dt, dim):
        self.gradient = gradient
        self.dt = dt
        self.dim = dim
        self.data = KoopmanData(None, None)
        
    def _ou_generator(self, N, n_steps, params):
        theta, sigma, max_u = params
        
        assert len(max_u) == self.dim

        X_n = (np.random.rand(N, self.dim) - 0.5) * 2 * max_u
        mus = (np.random.rand(N, self.dim) - 0.5) * 2 * max_u
        results = [X_n]
        
        for i in range(n_steps-1):
            noise = sigma * np.random.randn(N, self.dim) * np.sqrt(self.dt)
            X_n = X_n + theta * (mus - X_n) * self.dt + noise
            results.append(X_n)
        results = np.array(results)
        
        for idx, i in enumerate(max_u):
            if i == 0:
                results[:, :, idx] = 0
                
        return np.swapaxes(results, 0, 1)
    
    def _simulate(self, x0, t, u):
        traj = [x0]
        for i in range(1, len(t)):
            tspan = [t[i-1], t[i]]
            x0 = odeint(self.gradient, x0, tspan, args=(u[i-1],))[1]
            traj.append(x0)
        return np.array(traj)
    
    def _apply_constraint(self, result, constraint):
        if constraint is None:
            return True
        a = sum([constraint(i) for i in result])
        if a > 0:
            return False
        else:
            return True
        
    def _get_initial_cond(self, max_x, constraint):
        while True:
            xs = (np.random.rand(self.dim) - 0.5) * 2 * max_x
            if self._apply_constraint([xs], constraint):
                return xs
            
    
    def generate(self, N, n_steps, params, constraint=None):
        """
        Args:
            N: (int) number of trajectories simulated
            n_steps: (int) number of time-steps for each simulation
            params: (dict) configuration dictionary
                    "theta": (float), parameter for OU process
                    "sigma": (float), parameter for OU process
                    "keep_zero": (bool) half of trajectories generated will be autonomous
                    "max_u": (float) maximum input magnitude
                    "max_x"" (float) maximum state magnitude
            constraint: function which returns true if a constraint was violated for the
                        current state
        """
        
        theta = params.get('theta') or 2
        sigma = params.get('sigma') or 1
        keep_zero = params.get('keep_zero')
        if keep_zero is None:
            keep_zero = True
        max_u = params.get('max_u')
        max_x = params.get('max_x')
        
        assert len(max_x) == len(max_u) == self.dim
        
        t = np.arange(0, self.dt*n_steps, self.dt)
        
        traj_n_input = []
        traj_w_input = []
        inputs = []
        
        for i in tqdm(range(N)):
            
            flag = True
            while flag:
                # Generate a random initial condition
                xs = self._get_initial_cond(max_x, constraint)
                # Generate random inputs
                us = self._ou_generator(1, n_steps, [theta, sigma, max_u])
                
                # run input dynamics
                r0 = self._simulate(xs, t, us[0])
                
                # check input dynamics constraint
                c1 = self._apply_constraint(r0, constraint)
                
                if keep_zero:
                    u0 = np.zeros(us.shape)
                    r1 = self._simulate(xs, t, u0[0])
        
                    # check free dynamics constraint
                    c2 = self._apply_constraint(r1, constraint)
                    flag = not (c1 and c2)
                else:
                    flag = not c1
            if keep_zero:
                traj_n_input.append(r1)
                
            traj_w_input.append(r0)
            inputs.append(us[0])
                                
        xs = np.asarray(traj_w_input)
        us = np.asarray(inputs)
        
        if keep_zero:
            xs = np.concatenate((xs, np.asarray(traj_n_input)), axis=0)
            us = np.concatenate((us, np.zeros(us.shape)), axis=0)
        self.data.us = us
        self.data.xs = xs
                
        return self.data
    