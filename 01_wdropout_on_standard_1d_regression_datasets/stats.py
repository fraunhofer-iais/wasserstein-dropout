import copy
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import norm, wasserstein_distance, kstest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score

import torch

from functional import nll, nll_nd, evidential_nll

def calc_ece(pred_error_quantiles):
    bins = np.linspace(0.1, 0.9, 9)
    n    = len(pred_error_quantiles)
    
    digitized = np.digitize(pred_error_quantiles, bins)
    ece       = np.abs(((pd.Series(digitized).value_counts()/n)-0.1)).sum()
    
    return ece

def calc_ece_and_iso_reg(pred_error_quantiles):
    
    bins = np.linspace(-0.0001,1.0001,21)
    rel_freqs = np.zeros(len(bins)-1)
    n = len(pred_error_quantiles)
        
    digitized = np.digitize(pred_error_quantiles, bins)
    digitized = pd.Series(digitized).value_counts()/n
    
    for i in digitized.index:
        rel_freqs[i-1] = digitized[i]
    
    ece = np.abs(rel_freqs-0.05).sum()
    
    model_quantiles = bins[1:]-0.025
    emp_quantiles   = np.add.accumulate(rel_freqs)
    iso_reg         = IsotonicRegression(out_of_bounds='clip').fit(model_quantiles,emp_quantiles)

    return ece, iso_reg

def net_gradient_norm(datapoint,net):
    test_in  = torch.tensor(datapoint,requires_grad=True,dtype=torch.float32)
    test_out = net(test_in)
    return torch.autograd.grad(test_out, test_in)[0].norm().item()

# better use isotropic Gaussian for const density on eps-sphere
def random_perturb_hull(eps,dim):
    per = np.random.uniform(2*eps,5*eps,dim)
    if norm(per) > eps:
        per = eps*per/norm(per)
    return per

def random_perturb_ball(eps,dim):     
    return norm(np.random.multivariate_normal(np.zeros(dim),0.005*eps*np.diag(np.ones(dim))))

def calc_datapoint_statistics(net,data,method, iso_reg=None, train_N=None):
    
    def _insert_mean_cov(df, pred_y_samples):
        pred_mean = np.mean(pred_y_samples, axis=0)
        df['pred_mean'] = list(pred_mean)
        pred_cov = [np.cov(pred_y_sample) for pred_y_sample in np.transpose(pred_y_samples, [1, 2, 0])]
        df['pred_cov'] = pred_cov
        return pred_mean, np.array(pred_cov)
        
    X,y = data
    pred_y_samples, pred_sigma_samples = [], []
    eps = 1e-10
    
    n_output = y.shape[-1]

    if n_output == 1:
        df = pd.DataFrame(y.flatten()).rename(columns={0:'gt'})
        df['x'] = X.tolist()
    else:
        df = pd.DataFrame()
        df['gt'] = list(y) # rows of lists with n_output entries
        df['x'] = list(X) # rows of lists with n_features entries
    
    with torch.no_grad(): 
        
        # Compute mean and std from network outputs   
        if 'mc' in method or 'new_wdrop' in method or 'concrete_dropout' in method: # Get predictions with deactivated dropout and multiple predictions per input point with activated dropout
            
            if n_output == 1:
                if 'concrete_dropout' not in method:
                    net_out = net(torch.FloatTensor(X),drop_bool=False).cpu().numpy()

                    if method == 'mc_pu':
                        pred_y_no_mc = net_out[:, 0].flatten()
                        pred_sigma_no_mc = net_out[:, 1].flatten()
                    else:
                        pred_y_no_mc = list(net_out.flatten())
                
                for _ in range(200):
                    net_out = net(torch.FloatTensor(X)).cpu().numpy()
                    
                    if method == 'mc_pu':
                        pred_y_samples.append(list(net_out[:, 0].flatten()))
                        pred_sigma_samples.append(net_out[:, 1].flatten())
                    else:
                        pred_y_samples.append(list(net_out.flatten()))
                        
                df['pred_mean'] = pd.DataFrame(pred_y_samples).mean()
                
                pred_var = pd.DataFrame(pred_y_samples).var()
                if 'mc' in method:
                    wd = net.train_params['weight_decay']
                    p = float(net.drop_p)
                    l = 1
                    wd = float(wd) if wd is not None else None
                    if wd is not None and wd > 0:   
                        tau_inv = (2*train_N*wd)/(p*l**2)
                        pred_var += tau_inv
                df['pred_std'] = np.sqrt(pred_var)

            else:
                if method in ['mc_pu'] or 'new_wdrop' in method:
                    raise Exception('nd output is not implemented for mc_pu / wdrop yet')
                                              
                pred_y_no_mc = net(torch.FloatTensor(X), drop_bool=False).cpu().numpy() # shape: len(X) x n_output
                pred_y_samples = [net(torch.FloatTensor(X)).cpu().numpy() for _ in range(200)] # shape: 200 x len(X) x n_output
                pred_mean, pred_cov = _insert_mean_cov(df, pred_y_samples)
            
        
        elif method == 'de':
            
            if n_output == 1:
                for i in range(len(net)):
                    pred_y_samples.append(list((net[i](torch.FloatTensor(X)).cpu().numpy()).flatten()))
                df['pred_mean'] = pd.DataFrame(pred_y_samples).mean()
                df['pred_std']  = pd.DataFrame(pred_y_samples).std()
            else:
                pred_y_samples = [net[i](torch.FloatTensor(X)).cpu().numpy() for i in range(len(net))] # shape n_net x len(X) x n_output
                pred_mean, pred_cov = _insert_mean_cov(df, pred_y_samples) 
            
        
        elif method == 'pu':
            if n_output == 1:
                df[['pred_mean','pred_std']] = pd.DataFrame(net(torch.FloatTensor(X)).cpu().numpy())
            else:
                net_out = net(torch.FloatTensor(X)).cpu().numpy()
                pred_mean = net_out[:, :n_output]
                pred_cov = net_out[:, n_output:].reshape((-1, n_output, n_output))
                
                df['pred_mean'] = list(pred_mean)
                df['pred_cov'] = list(pred_cov)
                
        elif method == 'evidential':
            
            net_out = net(torch.FloatTensor(X)).cpu().numpy()
            
            gamma = net_out[:, :n_output] # E[mu] = gamma
            nu = net_out[:, n_output:n_output*2]
            alpha = net_out[:, n_output*2:n_output*3]
            beta = net_out[:, n_output*3:]
            
            df['params'] = list(np.concatenate((nu, alpha, beta), axis=1))
            
            pred_var_aleatoric = beta / (alpha-1 + eps) # E[sigma^2] = beta / (alpha-1)
            pred_var_epistemic = pred_var_aleatoric / (nu + eps) # Var[mu] = E[sigma^2]/nu
            
            df['pred_mean'] = gamma
            df['pred_std'] = np.sqrt(pred_var_aleatoric + pred_var_epistemic + eps)
            
        elif 'swag' in method:
            
            net_copy = type(net)(net_params=net.net_params,
                                train_params=net.train_params) 
            sample_weights = net.sample_weights(n_samples=50)
            
            outputs = []
            for sample in range(50):
                
                sample_state_dict = []
                for i, (name, layer_weights) in enumerate(net.state_dict().items()):
                    sample_state_dict.append((name, sample_weights[i][sample, ...]))
                
                net_copy.load_state_dict(OrderedDict(sample_state_dict))
                outputs.append(net_copy(torch.FloatTensor(X)).cpu().data.numpy())
            
            outputs = np.array(outputs)
            if method == 'swag':
                df['pred_mean'] = outputs.mean(axis=0)
                df['pred_std'] = outputs.std(axis=0)
            
            
            # TODO: n_output > 1 
            
            
        elif method == 'pu_de':
            
            if n_output == 1:
                mus = []
                sigmas = []
                for net_ in net:
                    net_mu_sigma = net_(torch.FloatTensor(X)).cpu().data.numpy()    
                    mus.append(net_mu_sigma[:,0])
                    sigmas.append(net_mu_sigma[:,1])

                mus    = np.array(mus)
                sigmas = np.array(sigmas)
                df['pred_mean'] = mus.mean(axis=0)
                df['pred_std']  = np.sqrt( (sigmas**2 + mus**2).mean(axis=0) - df['pred_mean']**2 )
            else:
                mus = []
                covs = []
                for net_ in net:
                    net_out = net_(torch.FloatTensor(X)).cpu().numpy()
                    mus.append(net_out[:, :n_output])
                    covs.append(net_out[:, n_output:].reshape((-1, n_output, n_output)))
                
                mus, covs = np.array(mus), np.array(covs) # shapes: len(net) x len(X) x n_output / len(net) x len(X) x n_output x n_output
                pred_mean = np.mean(mus, axis=0) # shape: len(X) x n_output
                df['pred_mean'] = list(pred_mean)
                pred_cov = np.sum(covs + mus[:, :, None, :] * mus[:, :, :, None], axis=0) - pred_mean[:, None, :] * pred_mean[:, :, None]
                df['pred_cov'] = list(pred_cov)
        
        elif method == 'vanilla':
            pred_y_no_mc = net(torch.FloatTensor(X), drop_bool=False).cpu().numpy()
            df['pred_no_mc'] = pred_y_no_mc
            df['pred_mean'] = pred_y_no_mc

        if 'mc' in method or 'new_wdrop' in method:
            df['pred_no_mc'] = list(pred_y_no_mc)
            df['spread']     = df['pred_mean'] - df['pred_no_mc']


        # Further metrics: nll (of gt in model under gaussian assumption), residual (i.e. mean - gt), error quantile (quantile of gt in normalized prediction distribution)       
        if method == 'mc_pu':
            
            if n_output == 1:
                pred_var_samples = np.array(pred_sigma_samples)**2
                mean_pred_var_samples = np.mean(pred_var_samples, axis=0)
                var_pred_y_samples = df['pred_std'].to_numpy()**2
                total_std = np.sqrt(var_pred_y_samples + mean_pred_var_samples)
                 
                df['total_std'] = total_std
                df['nll'] = df.apply(lambda x: nll(x['pred_mean'], x['total_std'], x['gt']), axis=1)
                df['pred_residual'] = df['pred_mean'] - df['gt']
                
            else:
                raise Exception('mc_pu has not been implemented for n_output > 1')
        
        elif method == 'vanilla':
            df['pred_residual'] = df['pred_no_mc'] - df['gt']
            
        elif method == 'evidential':
            
            df['pred_residual'] = df['pred_mean'] - df['gt']
            df['nll'] = list(evidential_nll(gamma, nu, alpha, beta, y))
            df['total_std'] = df['pred_std']
        
        else:
            
            if n_output == 1:
                df['total_std'] = df['pred_std']
                df['nll'] = df.apply(lambda x: nll(x['pred_mean'],x['pred_std'],x['gt']),axis=1)
            else:
                df['total_std'] = [np.diag(cov) for cov in df['pred_cov']]
                df['nll'] = list(nll_nd(pred_mean, pred_cov, y, treat_invalid='ignore'))
                for i in range(n_output):
                    df['nll_%d' % i] = list(nll(pred_mean[:, i], np.sqrt(pred_cov[:, i, i]), y[:, i]).flatten())
                
            df['pred_residual']        = df['pred_mean']-df['gt']

    if method not in ['vanilla']:
        
        df['pred_residual_normed'] = df['pred_residual']/(df['total_std']+eps)   
        df['error_quantile']       = df['pred_residual_normed'].apply(lambda x: np.round(norm.cdf(x),2))
        
        if n_output == 1:
            if 'mc' in method and method != 'mc_pu':
                df['net_gradient_norm'] = pd.DataFrame(X).apply(lambda x: net_gradient_norm(x,net),axis=1)
            else:
                df['net_gradient_norm'] = 1e10

            _, iso_reg_ = calc_ece_and_iso_reg(df['error_quantile'])
            if iso_reg is not None:
                if isinstance(iso_reg, list):
                    if len(iso_reg) == 0:
                        iso_reg.append(iso_reg_)
                        df['error_quantile_calibrated'] = iso_reg_.predict(df['error_quantile'])
                    else:
                        df['error_quantile_calibrated'] = iso_reg[0].predict(df['error_quantile'])

        else:

            err_quantile = np.array([item for item in df['error_quantile']])
            do_iso_reg = (iso_reg is not None) and (isinstance(iso_reg, list))
            reuse_iso_reg = do_iso_reg and (len(iso_reg) == n_output) 

            err_quantile_calibrated = []
            for i in range(n_output):
                _, iso_reg_ = calc_ece_and_iso_reg(err_quantile[:, i])

                if do_iso_reg and not reuse_iso_reg: # do isotonic regression from scratch if list is empty
                    iso_reg.append(iso_reg_)
                    err_quantile_calibrated.append(iso_reg_.predict(err_quantile[:, i]))
                elif reuse_iso_reg: # reuse learned regressor if list has same size as n_output
                    err_quantile_calibrated.append(iso_reg[i].predict(err_quantile[:, i]))

            df['error_quantile_calibrated'] = list(np.transpose(err_quantile_calibrated))

    return df

def calc_global_statistics(df, n_output=1):
    
    rmse = np.sqrt((df['pred_residual']**2).mean()).tolist()
    
    nll = None
    if 'nll' in df.columns.values:
        nll  = df['nll'].mean()
    
    if n_output == 1:
        
        r2   = r2_score(df['gt'],df['pred_mean'])
        
        ece = None
        if 'error_quantile' in df.columns.values:
            ece, _ = calc_ece_and_iso_reg(df['error_quantile']) 
       
        ws_dist, ks_dist = None, None
        if 'pred_residual_normed' in df.columns.values:
            ws_dist = wasserstein_distance(df['pred_residual_normed'],np.random.randn(100000))
            ks_dist = kstest(df['pred_residual_normed'].values,'norm')[0]
        
        res = {'rmse':rmse,'r2':r2,'nll':nll,'ece':ece, 'ks_dist':ks_dist,'ws_dist':ws_dist}
        
        if ('error_quantile_calibrated' in df.columns.values):
            res['ece_calib'], _ = calc_ece_and_iso_reg(df['error_quantile_calibrated'])
    else:
        y = np.array([item for item in df['gt']])
        pred_mean = np.array([item for item in df['pred_mean']])
        r2 = [r2_score(y[:, i], pred_mean[:, i]) for i in range(n_output)]
        
        res = {'rmse':rmse, 'nll': nll}
        
        ece = None
        if 'error_quantile' in df.columns.values:
            err_quantile = np.array([item for item in df['error_quantile']])
            ece = [calc_ece_and_iso_reg(err_quantile[:, i])[0] for i in range(n_output)]
            res.update({'ece_%d' % i: ece[i] for i in range(n_output)})
        
        ws_dists, ks_dists = None, None
        if 'pred_residual_normed' in df.columns.values:
            pred_res_normed = np.array([item for item in df['pred_residual_normed']])
            ws_dists = [wasserstein_distance(pred_res_normed[:, i], np.random.randn(100000)) for i in range(n_output)] 
            ks_dists = [kstest(pred_res_normed[:, i], 'norm') for i in range(n_output)]
            res.update({'ws_dist_%d' % i: ws_dists[i] for i in range(n_output)})
            res.update({'ks_dist_%d' % i: ks_dists[i] for i in range(n_output)})
        
        if nll is not None:
            nlls = [df['nll_%d' % i].mean() for i in range(n_output)]
            res.update({'nll_%d' % i: nlls[i] for i in range(n_output)})
        
        if ('error_quantile_calibrated' in df.columns.values):
            err_quant_calib = np.array([item for item in df['error_quantile_calibrated']])
            res.update({'ece_calib_%d' % i: calc_ece_and_iso_reg(err_quant_calib[:, i])[0]})
    
    return res