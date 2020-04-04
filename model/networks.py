import torch
import torch.nn as nn
import torch.nn.functional as F
from listmodule import ListModule
from torch_rbf import *


def get_encoder(params, shape, name='enc'):
    # returns encoder given its configuration and shape
    use_rbf = params.get('use_rbf') or False
    kernel = params.get('kernel') or None
    config = {'shape': shape, 'use_rbf':use_rbf, 'kernel':kernel}
    return Encoder(config, name)


class Encoder(nn.Module):
    """
    General block for lifting the state-space dimension:
    y = h(x), where Dim(y)>Dim(x)
    
    Arguments:
        params: configuration dictionary
                requires:
                    "shape": (list of int)
                optional"
                    "drop_prob": (float) dropout probability
                    "use_rbf": (Bool) whether to use RBF as first layer
                    "kernel": type of kernel for RBF layer}
                 
        name:   (str) encoder name    
    """
    def __init__(self, params, name='encoder'):
        super().__init__()
        self.params = params
        self.shape = params['shape']
        self.dim = self.shape[0]
        self.drop_prob = params.get('drop_prob') or 0.
        
        self.dropout = nn.Dropout(p=self.drop_prob)
        
        # Can use a rbf kernel for the first layer
        self.use_rbf = params.get('use_rbf') or False
        if self.use_rbf:
            kernel_type = params.get('kernel') or 'gaussian'
            str2base_func = basis_func_dict()
            kernel = str2base_func[kernel_type]
            self.rbf = RBF(self.dim, self.shape[1], kernel)
            self.shape = self.shape[1:]
            
        # the (rest) of the layers are linear    
        self.aux_layers = ListModule(self, '{}_'.format(name))
        
        for j in range(len(self.shape)-1):
            bias = bool(j+2!=len(self.shape))
            self.aux_layers.append(nn.Linear(self.shape[j],
                                             self.shape[j+1],
                                             bias=bias))
            
    def forward(self, x):
        if self.use_rbf:
            if len(x.size()) == 3:
                # handle 3D data (batch, timestep, dim)
                N, T, D = x.size()
                x = self.rbf(x.contiguous().view((N * T, D)))
                x = x.contiguous().view((N, T, x.shape[-1]))
            else:
                x = self.rbf(x)
            x = F.relu(x)
            
        for idx, layer in enumerate(self.aux_layers):
            if idx <= len(self.aux_layers)-2:
                x = F.relu(layer(x))
            else:
                # only do dropout on the last layer
                x = self.dropout(layer(x))
        return x
    
    
class LREN(nn.Module):
    """
    Linearly Recurrent Encoder Network
    y_n = [x_n, g(x_n)]^T
    y_{n+1} = K y_n
    
    Arguments:
        params: configuration dictionary,
                requires:
                    'enc_shape': (list of int), 
                    'n_shifts': number of time shifts
                optional:
                    'use_rbf': (bool)
                    'kernel'(str): type of RBF kernel
                    'drop_prob': (float)
                    'return_ko': (bool)
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        enc_shape = params['enc_shape']
        self.dim = params['enc_shape'][0]
        
        self.encoder = get_encoder(params, enc_shape, 'enc_x')
        self.koopman = nn.Linear(enc_shape[-1]+self.dim,
                            enc_shape[-1]+self.dim,
                            bias=False)
        
    def forward(self, x, return_ko=False):
        # generate ground truth
        x = x[:, :self.params['n_shifts'], :]
        y = self.encoder(x)
        # append observable state
        y = torch.cat((x, y), dim=-1)
        
        # generate trajectories from initial state
        y_pred = y[:, 0:1, :].clone()
        for i in range(self.params['n_shifts']-1):
            y_next = self.koopman(y_pred[:, -1:, :])
            y_pred = torch.cat((y_pred, y_next), axis=1)
            
        if return_ko:
            koopman = self.koopman.parameters()
            koopman = torch.Tensor(list(koopman)[0])
            return y, y_pred, koopman
        else:
            return y, y_pred
        
        
class DENIS(nn.Module):
    """
    Deep Encoder Network with Initial State parameterisation
    y_n = [x_n, g(x_n)]^T
    K = h(x_0)
    y_{n+1} = K y_n
    
    Arguments:
        params: configuration dictionary,
                requires:
                    'enc_shape': (list of int), 
                    'aux_shape': (list of int), 
                    'n_shifts': number of time shifts
                optional:
                    'use_rbf': (bool)
                    'kernel'(str): type of RBF kernel
                    'drop_prob': (float)
                    'return_ko': (bool)
    """
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.dim = params['enc_shape'][0]
        self.ldim = params['aux_shape'][-1]

        enc_shape = params['enc_shape']
        aux_shape = params['aux_shape'].copy()
        self.aux_rbf = params.get('aux_rbf') or False
        

        assert aux_shape[-1] == enc_shape[0] + enc_shape[-1],\
               "Invalid auxiliary network's output shape"
        assert aux_shape[0] == enc_shape[0],\
               "Input state dimensions must match"
        
        aux_shape[-1] = aux_shape[-1]*aux_shape[-1]
        
        self.encoder = get_encoder(params, enc_shape, 'enc_x')
        
        aux_params = params.copy()
        aux_params['use_rbf'] = self.aux_rbf
        self.aux_net = get_encoder(aux_params, aux_shape, 'aux_')
        

    def forward(self, x, return_ko=False):
        # Generate parameterized Koopman operator
        x0 = x[:, 0, :].clone()
        koopman = self.aux_net(x0)
        koopman = koopman.view((koopman.shape[0], self.ldim, self.ldim))

        # generate ground truth
        x = x[:, :self.params['n_shifts'], :]
        y = self.encoder(x)
        # append observable state
        y = torch.cat((x, y), dim=-1)

        # generate trajectories from initial state
        y_pred = y[:, 0:1, :].clone()
        for i in range(self.params['n_shifts']-1):
            y_next = torch.bmm(y_pred[:, -1:, :], koopman)
            y_pred = torch.cat((y_pred, y_next), axis=1)
            
        if return_ko:
            return y, y_pred, koopman
        else:
            return y, y_pred
    
    
class DENIS_JBF(nn.Module):
    """
    Deep Encoder Network with Initial State parameterisation
    In Jordon block diagonal form, i.e. direct eigenfunction learning
    y_n = g(x_n)^T
    B_n = h_n(x_0)
    K = [[B_1, 0, ..., 0], [0, B_2, ...0], [0, ..., B_N]]
    y_{n+1} = K y_n
    x_{n+1} = C y_{n+1}
    
    Arguments:
        params: configuration dictionary,
                requires:
                    'enc_shape': (list of int), 
                    'aux_shape': (list of int), 
                    'n_shifts': number of time shifts
                    'dt': sampling time
                optional:
                    'use_rbf': (bool)
                    'kernel'(str): type of RBF kernel
                    'drop_prob': (float)
                    'return_ko': (bool)
                    'aux_rbf': (bool), whether aux nets use RBF layers
                    
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        self.dim = params['enc_shape'][0]
        self.ldim = params['enc_shape'][-1]
        self.dt = params['dt']
        self.aux_rbf = params.get('aux_rbf') or False

        enc_shape = params['enc_shape']
        aux_shape = params['aux_shape'].copy()
        
        self.n_aux = int(enc_shape[-1] // 2)
        self.koopman = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        assert aux_shape[0] == enc_shape[0], 'Invalid state dimensions'
        assert aux_shape[-1] == 2, 'Each aux net must output omega and mu'
        assert self.ldim % 2 == 0, 'Encoder output shape must be even'
        
        # initialize encoder network
        self.encoder = get_encoder(params, enc_shape, 'enc_x')
        
        # initialize auxiliary networks (each one represent one Jacobian block)
        self.aux_nets = ListModule(self, 'auxNets_')
        # option to not to use rbf for aux nets
        enc_params = params.copy()
        enc_params['use_rbf'] = self.aux_rbf
        for k in range(self.n_aux):
            self.aux_nets.append(get_encoder(enc_params, aux_shape, 'auc_'))
            
        # initialize C matrix for casting to observed state
        self.C = nn.Linear(self.ldim, self.dim,bias=False)
        
        
    def block_diag(self, m):
        def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
            return v.reshape(
                torch.Size([1] * n_dim_to_prepend)
                + v.shape
                + torch.Size([1] * n_dim_to_append))
        if type(m) is list:
            m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

        d = m.dim()
        n = m.shape[-3]
        siz0 = m.shape[:-3]
        siz1 = m.shape[-2:]
        m2 = m.unsqueeze(-2)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        eye = attach_dim(torch.eye(n).unsqueeze(-2).to(device), d - 3, 1)
        return (m2 * eye).reshape(
            siz0 + torch.Size(torch.tensor(siz1) * n)
        )
            
    def forward(self, x, return_ko=False):
        # Generate Koopman operator (block parameters)
        x0 = x[:, 0, :].clone()
        aux_out = torch.Tensor(x0.shape[0], self.n_aux, 2).to(self.device)
        for idx, net in enumerate(self.aux_nets):
            aux_out[:, idx, :] = net(x0)
            
        # Form Koopman operator from Jordon blocks
        scale = torch.exp(aux_out[:, :, 0] * self.dt)
        cos = torch.cos(aux_out[:, :, 1] * self.dt) * scale
        sin = torch.sin(aux_out[:, :, 1] * self.dt) * scale
        row1 = torch.stack((cos, sin), dim=2)
        row2 = torch.stack((-sin, cos), dim=2)
        koopman = torch.stack((row1, row2), dim=2)
        koopman = list(map(self.block_diag, torch.unbind(koopman, 0)))
        koopman = torch.stack(koopman, 0)
        self.koopman = koopman
        
        # generate ground truth
        n_shifts = self.params['n_shifts']
        x = x[:, :n_shifts, :]
        y = self.encoder(x)
        y = torch.cat((x, y), dim=-1)
                
        # generate trajectories from initial state
        y_pred = y[:, 0:1, self.dim:].clone()
        x_pred = y[:, 0:1, :self.dim].clone()
            
        for i in range(n_shifts-1):
            y_next = torch.bmm(y_pred[:, -1:, :], koopman)
            y_pred = torch.cat((y_pred, y_next), dim=1)
            x_next = self.C(y_next)
            x_pred = torch.cat((x_pred, x_next), dim=1)

        y_pred = torch.cat((x_pred, y_pred), dim=2)

        if return_ko:
            return y, y_pred, koopman
        else:
            return y, y_pred
        
        
class DEINA(nn.Module):
    """
    Deep Encoder for Input Non-Affine Systems
    y_n = [x_n, g(x_n)]^T
    y_{n+1} = K y_n + B u_n
    or
    y_{n+1} = K y_n + B [u_n + h(u_n)]^T
    
    Arguments:
        params: configuration dictionary,
                requires:
                    'enc_x_shape': (list of int),
                    'n_shifts': number of time shifts
                optional:
                    'use_rbf': (bool),
                    'use_enc_u', (bool) whether to add
                                 non-linearities in u
                    'enc_u_shape': (list of int), 
                    'kernel'(str): type of RBF kernel
                    'drop_prob': (float)
                    'return_ko': (bool)
                    'enc_rbf': (bool), whether  use RBF layers
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        self.enc_x_shape = params['enc_x_shape']
        self.use_enc_u = params['use_enc_u'] or False
        
        self.dim = self.enc_x_shape[0]        
        self.ldim = self.enc_x_shape[-1] + self.dim
                
        self.encoder_x = get_encoder(params, self.enc_x_shape, 'enc_x')
        
        # option to add non-linearities in u
        if self.use_enc_u:
            self.enc_u_shape = params['enc_u_shape']
            # option to add a rbf layer
            enc_params = params.copy()
            enc_params['use_rbf'] = params.get('enc_rbf') or False
            self.encoder_u = get_encoder(enc_params, self.enc_u_shape, 'enc_u')
            dim = self.enc_u_shape[-1] + self.dim
            self.B = nn.Linear(dim, self.ldim, bias=False)
        else:
            self.B = nn.Linear(self.dim, self.ldim, bias=False)
                        
        self.koopman = nn.Linear(self.ldim, self.ldim)
        
        
    def forward(self, x, u):
                
        # generate latent ground truth
        x = x[:, :self.params['n_shifts'], :]
        y = self.encoder_x(x)
        y = torch.cat((x, y), dim=-1)
        
        # generate encoded inputs
        u = u[:, :self.params['n_shifts'], :]
        if self.use_enc_u:
            v = self.encoder_u(u)
            u = torch.cat((u, v), dim=-1)
        Bu = self.B(u)
        
        # generate predicted trajectories
        y_pred = y[:, 0:1, :].clone()
        for i in range(self.params['n_shifts']-1):
            add = Bu[:, i, :].unsqueeze_(1)
            y_next = self.koopman(y_pred[:, -1:, :]) + add
            y_pred = torch.cat((y_pred, y_next), dim=1)
        
        return y, y_pred
   