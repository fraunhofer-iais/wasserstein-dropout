import numpy as np
from scipy.special import loggamma

import torch

def new_exact_wasserstein_dropout_loss(net, data, loss_params, eps=1e-8):
    
    inputs, labels = data
    n_samples, beta, det = loss_params
    
    outputs_mc = torch.stack([net(inputs) for _ in range(n_samples)])
    mean_outputs_mc = torch.mean(outputs_mc, dim=0)
    mean_outputs_mc_sq = torch.mean(outputs_mc**2, dim=0)
    
    var_w = mean_outputs_mc_sq - mean_outputs_mc**2
    sigma_w = torch.sqrt(var_w + eps)
    var_y_bootstrap = (mean_outputs_mc - labels)**2 + var_w
    sigma_y_bootstrap = torch.sqrt(var_y_bootstrap + eps)
    
    loss = torch.mean((mean_outputs_mc - labels)**2 + (sigma_w - sigma_y_bootstrap)**2)
    return loss
    
def nll_floored(y_pred,y_gt, EPS=1e-10):  # only for training of parametric uncertainty model
    mu    = y_pred[:,0]
    sigma = y_pred[:,1]
    y_gt  = torch.squeeze(y_gt)
    
    nll = torch.log(sigma**2 + EPS)/2 + ((y_gt-mu)**2)/(2*sigma**2 + EPS)
    nll[nll<-100]=-100 
    nll = nll.mean()  
    
    return nll

def nll_floored_nd(y_pred, y_gt, EPS=1e-10):
    
    n_output = y_gt.shape[-1]
    mu = y_pred[:, :n_output]
    cov = y_pred[:, n_output:].reshape((-1, n_output, n_output))
    
    norm = 0.5 * torch.log(torch.det(cov) + EPS)
    cov_eps = cov + torch.diag(EPS*torch.ones(n_output))[None, :, :]
    nll = norm + 0.5 * torch.matmul((y_gt - mu)[:, None, :], torch.matmul(torch.inverse(cov_eps), (y_gt - mu)[:, :, None]))
    nll[nll<-100] = -100
    nll = nll.mean()
    return nll

def nll(mu,sigma,y):
    eps = 1e-10
    return np.log(eps+sigma**2)/2 + ((y-mu)**2)/(eps+2*sigma**2)

def nll_nd(mean, cov, y, treat_invalid='raise', EPS=1e-10):
    dim = mean.shape[-1]
    const = (dim/2.) * np.log(2*np.pi)
    norm = 0.5 * np.log(np.linalg.det(cov) + EPS)
    cov_eps = cov + np.diag(EPS*np.ones(dim))[None, :, :]
    res = norm + 0.5 * np.matmul((y - mean)[:, None, :], np.matmul(np.linalg.inv(cov_eps), (y-mean)[:, :, None]))

    if treat_invalid == 'raise' and np.any(np.isnan(res)):
        raise ArithmeticError("Result contains NaN values")

    return res[:, 0, 0]

def evidential_loss(y_pred, y_gt, lmbda=0.001, eps=1e-10):

    n_output_y = y_gt.shape[-1]
    gamma = y_pred[:, :n_output_y]
    nu = y_pred[:, n_output_y:n_output_y*2]
    alpha = y_pred[:, n_output_y*2:n_output_y*3]
    beta = y_pred[:, n_output_y*3:]
    omega = 2*beta*(1+nu)

    nll = 0.5*(np.pi - torch.log(nu + eps)) 
    nll -= alpha*torch.log(omega + eps) 
    nll += (alpha + 0.5)*torch.log((y_gt - gamma)**2*nu + omega + eps)
    nll += torch.lgamma(alpha)
    nll -= torch.lgamma(alpha + 0.5)

    reg = torch.abs(y_gt - gamma) * (2*nu + alpha)

    return torch.mean(nll + lmbda*reg)

def evidential_nll(gamma, nu, alpha, beta, y, eps=1e-10):
    
    omega = 2*beta*(1+nu)
    nll = 0.5*(np.pi - np.log(nu + eps))
    nll -= alpha*np.log(omega + eps)
    nll += (alpha + 0.5)*np.log((y - gamma)**2 *nu + omega + eps)
    nll += loggamma(alpha)
    nll -= loggamma(alpha + 0.5)
    
    if y.shape[-1] == 1:
        return nll[:, 0]
    
    return nll

def concrete_dropout_loss(outputs, labels, reg):
    return torch.nn.MSELoss(reduction='mean')(outputs, labels) + reg