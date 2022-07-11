
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.stats import norm

def cov_matrix(c,n):
    return (1-c) * np.diag(np.ones(n)) + c * np.ones((n,n))

def corr_init_matrix(in_features,out_features,c):
    
    cdf_func = lambda x: norm.cdf(x,loc=0,scale=np.sqrt(2)/2)
    n        = max(in_features,out_features)
    
    W  = np.random.multivariate_normal(np.zeros(n),cov_matrix(c,n),n)
    W += np.random.multivariate_normal(np.zeros(n),cov_matrix(c,n),n).T
    W  = 0.5*W
    W  = cdf_func(W)
    
    W  = (2*W-1)*np.sqrt(1.0/in_features)
    W = W[:out_features,:in_features]
    
    return torch.nn.Parameter(torch.FloatTensor(W))

# The following method is from https://github.com/danielkelshaw/ConcreteDropout
def concrete_regulariser(model: nn.Module) -> nn.Module:

    """Adds ConcreteDropout regularisation functionality to a nn.Module.

    Parameters
    ----------
    model : nn.Module
        Model for which to calculate the ConcreteDropout regularisation.

    Returns
    -------
    model : nn.Module
        Model with additional functionality.
    """

    def regularisation(self) -> torch.Tensor:

        """Calculates ConcreteDropout regularisation for each module.

        The total ConcreteDropout can be calculated by iterating through
        each module in the model and accumulating the regularisation for
        each compatible layer.

        Returns
        -------
        Tensor
            Total ConcreteDropout regularisation.
        """

        total_regularisation = 0
        for module in filter(lambda x: isinstance(x, ConcreteDropout), self.modules()):
            total_regularisation += module.regularisation

        return total_regularisation

    setattr(model, 'regularisation', regularisation)

    return model
    
    
    
# Fully connected neural network with three hidden layers (with dropout)
class Net(nn.Module):
    def __init__(self, net_params, train_params):
        super(Net, self).__init__()
        
        self.n_input      = net_params['n_input']
        self.layer_width  = net_params['layer_width']
        self.num_layers   = net_params['num_layers']
        self.n_output     = net_params['n_output']
        self.nonlinearity = net_params['nonlinearity']
            
        self.drop_bool    = train_params['drop_bool']
        self.drop_bool_ll = train_params['drop_bool_ll']
        self.drop_p       = train_params['drop_p']
        
        self.net_params = net_params
        self.train_params = train_params 
        
        self.layers       = nn.ModuleList()
        
        if self.num_layers == 0:
            self.layers.append(nn.Linear(self.n_input,self.n_output))
        else:
            self.layers.append(nn.Linear(self.n_input,self.layer_width))
            for _ in range(self.num_layers-1):
                self.layers.append(nn.Linear(self.layer_width,self.layer_width))
            self.layers.append(nn.Linear(self.layer_width,self.n_output))
    
        for layer in self.layers:
            layer.weight = corr_init_matrix(layer.in_features,layer.out_features,net_params['init_corrcoef'])
    
    def forward(self, x, drop_bool=None):
        
        # drop_bool controls whether last layer dropout is used (True/False) or if values from the constructor shall be used (None)
        if drop_bool is None:
            drop_bool    = self.drop_bool
            drop_bool_ll = self.drop_bool_ll
        elif drop_bool is False:
            drop_bool_ll = False
        elif drop_bool is True:
            drop_bool_ll = True
        
        if self.num_layers == 0:
            x = F.dropout(x, p=self.drop_p, training=drop_bool_ll)
            x = self.layers[-1](x)
        else:
            for layer in self.layers[:-2]:
                x = layer(x)
                x = self.nonlinearity(x)
                x = F.dropout(x, p=self.drop_p, training=drop_bool)

            x = self.layers[-2](x)
            x = self.nonlinearity(x)
            x = F.dropout(x, p=self.drop_p, training=drop_bool_ll)

            x = self.layers[-1](x)
        
        return x
    
    def sample(self, X, n_samples=1):
        
        X = np.array(X)
        if len(X.shape) == 1:
            X = X[None, :]
            
        with torch.no_grad():
            X = torch.FloatTensor(X)
            samples = np.array([self(X).cpu().numpy() for _ in range(n_samples)])
        
        # samples has shape n_samples x len(X) x n_output 
        samples = np.swapaxes(samples, 0, 1)
        return samples
        
        
    
# network for parametric uncertainty (PU)
# x[:, 0]: Network output, x[:, 1] uncertainty estimate (1D)
# x[:, :n_output] Network mean, x[:, n_output:].reshape((-1, n_output, n_output)) is the covariance matrix (nD)
class Net_PU(Net):
    def __init__(self, net_params, train_params):
        super(Net_PU, self).__init__(net_params=net_params, train_params=train_params)
        self.softplus    = nn.Softplus()
        
        if self.n_output > 1:
            # compute gt dim
            self.n_output_y = int(-1.5 + np.sqrt((1.5)**2 + 2*self.n_output)) # solution of: 0 = x**2 + 3x - 2n_output
    
    def forward(self, x, drop_bool=None):
        
        if drop_bool is None:
            drop_bool = self.drop_bool
        
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.nonlinearity(x)
            x = F.dropout(x, p=self.drop_p, training=drop_bool)
        x = self.layers[-1](x)
        
        if self.n_output == 1: # TODO: This condition is obsolete
            x = torch.stack([x[:,0], self.softplus(x[:,1])], dim=1)
        
        else: # Parametrizing lower triangular matrix for ensuring PSD
            
            batch_size = x.shape[0]
            mu = x[:, :self.n_output_y]
            cov = torch.zeros((batch_size, self.n_output_y, self.n_output_y))
            tril_idxs = torch.tril_indices(self.n_output_y, self.n_output_y)
            cov[:, tril_idxs[0], tril_idxs[1]] = x[:, self.n_output_y:]
            
            # ensure diagonal to be positive using softplus
            cov[:, torch.arange(self.n_output_y), torch.arange(self.n_output_y)] = self.softplus(cov[:, torch.arange(self.n_output_y), torch.arange(self.n_output_y)]) 
            cov = torch.matmul(cov, torch.transpose(cov, 1, 2)) # c.f. cholesky decomp
            x = torch.cat((mu, cov.reshape((-1, self.n_output_y**2))), dim=1)
            
        return x


# Fully connected neural network with three hidden layers (with concrete dropout)
# Difference to Net in terms of architechture: Also dropping inputs
@concrete_regulariser
class CDNet(Net):
    def __init__(self, net_params, train_params):
        super(CDNet, self).__init__(net_params, train_params)

        w, d = train_params['sml_loss_params']
        self.cd_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.cd_layers.append(ConcreteDropout(weight_regulariser=w, dropout_regulariser=d))

    def forward(self, x):

        if self.num_layers == 0:  # case analogous to Net class (drops input)
            x = self.cd_layers[-1](x, self.layers[-1])  # The cd_layers drop first and then apply the layer
        else:  # do not drop input otherwise
            x = nn.Sequential(self.layers[0], self.nonlinearity)(x)
            for i, layer in enumerate(self.layers[1:-1]):
                x = self.cd_layers[i](x, nn.Sequential(layer, self.nonlinearity))
            x = self.cd_layers[-1](x, self.layers[-1])

        return x

    def sample(self, X, n_samples=1):
        raise NotImplementedError()

class Net_Evidential(Net):
    
    def __init__(self, net_params, train_params):
        super(Net_Evidential, self).__init__(net_params=net_params, train_params=train_params)
        
        self.softplus = nn.Softplus()
        self.n_output_y = int(self.n_output / 4)
        
    def forward(self, x):
        
        x = super(Net_Evidential, self).forward(x)
        
        res = torch.cat([
            x[:, :self.n_output_y],
            self.softplus(x[:, self.n_output_y:self.n_output_y*2]), # nu
            self.softplus(x[:, self.n_output_y*2:self.n_output_y*3]) + 1, # alpha
            self.softplus(x[:, self.n_output_y*3:]) # beta
        ], dim=1)
        
        return res
        
    
class Net_SWA(nn.Module):
    
    def __init__(self, net_params, train_params, base_model=Net):

        super(Net_SWA, self).__init__()
        
        self.base_model = base_model(net_params, train_params)
        self.add_module('base_model', self.base_model)
        
        self.n_output = self.base_model.n_output
        self.net_params = net_params
        self.train_params = train_params
        
        self.n_params = np.sum([np.prod(layer_weights.size()) for layer_weights in self.parameters()])
        
        self.mean_weights = [torch.zeros(layer_weights.size()) for layer_weights in self.parameters()]
        self.update_count = 0
    
    def forward(self, x):
        return self.base_model(x)
    
    def update(self):
        
        self.update_count += 1
        for i, (name, layer_weights) in enumerate(self.state_dict().items()):
            self.mean_weights[i] = (layer_weights + self.update_count * self.mean_weights[i])/(self.update_count + 1)

class Net_SWAG(Net_SWA):
    
    def __init__(self, net_params, train_params, base_model=Net):
        
        super(Net_SWAG, self).__init__(net_params=net_params, train_params=train_params, base_model=base_model)
        
        self.num_col = net_params['num_col']
        
        self.sec_moment_weights = [torch.zeros(layer_weights.size()) for layer_weights in self.state_dict().values()]
        self.latest_weight_deviations = []
        
    def update(self):
        super(Net_SWAG, self).update()
        
        cur_weight_deviations = []
        for i, (name, layer_weights) in enumerate(self.state_dict().items()):
            self.sec_moment_weights[i] = (layer_weights**2 + self.update_count * self.sec_moment_weights[i])/(self.update_count + 1)
            cur_weight_deviations.append(layer_weights - self.mean_weights[i])
            
        self.latest_weight_deviations.append(cur_weight_deviations)
        n_col = len(self.latest_weight_deviations)
        if n_col > self.num_col:
            start = n_col - self.num_col
            self.latest_weight_deviations = self.latest_weight_deviations[start:]
            
    
    def sample_weights(self, n_samples=1):
        
        if self.update_count < self.num_col:
            raise Exception("Cannot sample from model, since the number of update is not sufficient. You have to run it for more epochs.")
        
        z_2 = torch.normal(0, 1, (self.num_col, n_samples))
        
        weights_samples = []
        for i, (name, layer_weights) in enumerate(self.state_dict().items()):
            
            n_dims = len(layer_weights.size())
            weight_deviations = torch.stack([w[i] for w in self.latest_weight_deviations]).permute(list(range(1, n_dims+1))+[0])
            
            diag = self.sec_moment_weights[i] - self.mean_weights[i]**2 # shape: layer_weights.size
            
            z_1 = torch.normal(0, 1, (n_samples, *layer_weights.size()))
            
            weights = self.mean_weights[i] + (1./np.sqrt(2)) * torch.sqrt(diag) * z_1 # shape: n_samples x layer_weights.size
            weights += (1./np.sqrt(2*(self.num_col - 1))) * torch.matmul(weight_deviations, z_2).permute([n_dims] + list(range(n_dims)))
            weights_samples.append(weights)
            
        return weights_samples # shape: layer x n_samples x layer_weights.shape
        
# The following code is from https://github.com/danielkelshaw/ConcreteDropout

class ConcreteDropout(nn.Module):

    """Concrete Dropout.

    Implementation of the Concrete Dropout module as described in the
    'Concrete Dropout' paper: https://arxiv.org/pdf/1705.07832
    """

    def __init__(self,
                 weight_regulariser: float,
                 dropout_regulariser: float,
                 init_min: float = 0.1,
                 init_max: float = 0.1) -> None:

        """Concrete Dropout.

        Parameters
        ----------
        weight_regulariser : float
            Weight regulariser term.
        dropout_regulariser : float
            Dropout regulariser term.
        init_min : float
            Initial min value.
        init_max : float
            Initial max value.
        """

        super().__init__()

        self.weight_regulariser = weight_regulariser
        self.dropout_regulariser = dropout_regulariser

        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.p = torch.sigmoid(self.p_logit)

        self.regularisation = 0.0

    def forward(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:

        """Calculates the forward pass.

        The regularisation term for the layer is calculated and assigned to a
        class attribute - this can later be accessed to evaluate the loss.

        Parameters
        ----------
        x : Tensor
            Input to the Concrete Dropout.
        layer : nn.Module
            Layer for which to calculate the Concrete Dropout.

        Returns
        -------
        Tensor
            Output from the dropout layer.
        """

        output = layer(self._concrete_dropout(x))

        sum_of_squares = 0
        for param in layer.parameters():
            sum_of_squares += torch.sum(torch.pow(param, 2))

        weights_reg = self.weight_regulariser * sum_of_squares / (1.0 - self.p)

        dropout_reg = self.p * torch.log(self.p)
        dropout_reg += (1.0 - self.p) * torch.log(1.0 - self.p)
        dropout_reg *= self.dropout_regulariser * x[0].numel()

        self.regularisation = weights_reg + dropout_reg

        return output

    def _concrete_dropout(self, x: torch.Tensor) -> torch.Tensor:

        """Computes the Concrete Dropout.

        Parameters
        ----------
        x : Tensor
            Input tensor to the Concrete Dropout layer.

        Returns
        -------
        Tensor
            Outputs from Concrete Dropout.
        """

        eps = 1e-7
        tmp = 0.1

        self.p = torch.sigmoid(self.p_logit)
        u_noise = torch.rand_like(x)

        drop_prob = (torch.log(self.p + eps) -
                     torch.log(1 - self.p + eps) +
                     torch.log(u_noise + eps) -
                     torch.log(1 - u_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / tmp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        x = torch.mul(x, random_tensor) / retain_prob

        return x
