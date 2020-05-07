import harold
from control import *
from scipy.integrate import odeint
import numpy as np
from tqdm import tqdm

def pend_dynamics(model):
    def wrapper(x0, t, K, x_goal=np.array([0, 0]), us = [], ts = []):
        # us is used to store inputs for evaluation
        phi, omega = x0
        x0 = x0.reshape((1, 1, 2))
        x_goal = x_goal.reshape((1, 1, 2))
        y = model.encode(x0-x_goal)[0].T
        
        u = (K@y)[0, 0]
        us.append(u)
        ts.append(t)
        dydt = [omega, -np.sin(phi) - u]
        return dydt
    return wrapper


class KOOC():
    """
    Koopman Operator Optimal Control currently supporting only Denis.
    Uses ODE45 to simulate the actual dynamics obtained through the gain matrix
    calculated using the DENIS model

    Args:
        system: (str or func) dynamical system gradient (or pre-defined strings)
                 , must be compatiable with scipy's ode solver
        model: ML model
        dt: (float) sampling frequency used to train the model
        B: (ndarray) controller input matrix

    Methods:
        d2c: internal discrete to continous time system converter
        simulate: run a batch of dynamics simulated using ode45

    """
    def __init__(self, system, model, dt, B=None):
        self.model = model
        self.dt = dt
        if system == 'pendulum':
            self.dynamics = pend_dynamics(model)
        else:
            self.dynamics = system
        if B==None:
            self.B = np.array([[0], [1]])
        else:
            self.B = B

    def d2c(self, dim, A):
        B = self.B
        dt = self.dt
        ldim = len(A)-dim
        u_dim = B.shape[1]
        B_hat = np.append(B, np.zeros((1, ldim)))[:, np.newaxis]
        C = np.append(np.ones((1, dim)), np.zeros((1, ldim)))
        G = harold.State(A, B_hat, C, .0, dt)
        sys = harold.undiscretize(G, 'tustin')
        A_c = sys.a
        return A_c, B_hat
    

    def simulate(self, init_conds, x_goal, T, Q=1, r=1, show=True):
        """Simulate closed loop dynamics
        Args:
            init_conds: (ndarray) list of starting points
            x_goal: (ndarray) target state
            Q: LQR state cost matrix, if scalar, then assume np.eye()*Q cost
            r: (floar) LQR controller cost

        Returns:
            traj: (ndarray) list of simualted closed-loop trajectories
            u_kooc: (ndarray) controller inputs (varying intervals)
            t: (ndarray) time axis (uniform intervals)
            t_kooc: (ndarray) time steps used for the solver
        """
        x_goal = np.array(x_goal)
        results = self.model.predict(init_conds, 1, return_ko=True)
        kos = results[-1]
        dim = init_conds.shape[-1]
        us = []
        ts = []
        traj = []
        if show:
            p_bar = tqdm(range(len(init_conds)))
        else:
            p_bar = range(len(init_conds))
        for i in p_bar:
            x0 = init_conds[i]
            ko = kos[i]
            A = ko.T
            A_c, B_hat = self.d2c(dim, A)
            if np.isscalar(Q):
                Q = Q * np.eye(dim)
            else:
                Q = np.diag(Q)
            Q_hat = np.zeros((A.shape))
            Q_hat[:dim, :dim] = Q
            K, _, _ = lqr(A_c, B_hat, Q_hat, r)
            t = np.arange(0, T*self.dt, self.dt)
            u = []
            _t = []
            sol = odeint(self.dynamics, init_conds[i, 0], t,
                         args=(K, x_goal, u, _t))
            us.append(u)
            ts.append(_t)
            traj.append(sol)
        traj, us, ts = np.array(traj), np.array(us), np.array(ts)
        return traj, us, t, ts
