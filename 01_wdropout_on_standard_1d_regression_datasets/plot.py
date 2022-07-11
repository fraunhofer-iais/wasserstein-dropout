import re
import os
from collections import OrderedDict
import itertools as itert

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from decimal import Decimal

from data import available_datasets, get_dir_files, load_method_dict, load_global_stats

default_ident_offsets = {'train': -0.2, 'val': -0.18, 'test': -0.12, 
                     'label_test_interpolate': -0.04, 'label_test_extrapolate': 0.04,
                     'pca_test_interpolate': 0.12, 'pca_test_extrapolate': 0.2}

method_to_marker = {'mc': 'D', 'mc_ll': 'd', 
                   'pu': '.',  'pu_de': '*', 'mc_pu': 'o', 
                    'de': 'x', 'swag': '^',
                    'evidential': 'p',
                   'evidential_lmbda=0.5': 's', 'evidential_lmbda=0.1': '*', 'evidential_lmbda=0.01': 'o', 'evidential_lmbda=0.001': 'd', 'evidential_lmbda=0.0001': '.',
                   'new_wdrop_exact_l=5': '+',
                   'new_wdrop_exact_l=4': 's', 'new_wdrop_exact_l=8': '*', 'new_wdrop_exact_l=10': 'o', 'new_wdrop_exact_l=20': '.', 'new_wdrop_exact_l=5_det': 'd' }

ident_to_color = {'train': 'g', 'val': 'lightblue', 'test': 'b', 'label_test_interpolate': 'r', 'label_test_extrapolate': 'lightcoral',
                  'pca_test_interpolate': 'y', 'pca_test_extrapolate': 'orange'}
dataset_to_marker = {'toy': ',', 'toy_noise': '.', 'yacht': '+', 'diabetes': 'x', 'boston': '|', 
                     'energy': '_', 'concrete': '1', 'wine_red': '3', 
                    'abalone': 'o', 'naval': 'v', 'power': '^', 'california': 's', 'superconduct': 'P', 
                     'protein': 'D', 'year': '*'}
fontP = FontProperties()
fontP.set_size('xx-small')

dataset_to_size = {'boston': 506, 'kin8nm': 8000, 'wine_red': 1599, 'concrete': 1030, 'toy_noise': 10000, 'abalone': 4176, 'energy': 768,
                   'year': 515345, 'protein': 45730, 'california': 20640, 'superconduct': 21263, 'diabetes': 442, 'naval': 11934, 
                   'power': 9568, 'yacht': 308, 'toy_hf': 1000, 'toy_modulated': 20000}


def name_map(name):
    if name == 'concrete_dropout':
        return 'con-mc'
    if name == 'evidential':
        return 'pu_ev'
    if name.startswith('mc'):
        pat = re.compile(r'mc(_p=(0.[0-9]+))?(_wd=(0.[0-9]+))?')
        mat = pat.match(name)
        res = 'mc'
        if mat is not None:
            grps = mat.groups()
            if grps[1] is not None or grps[3] is not None:
                res += ' ('
                if grps[1] is not None:
                    res += 'p={:}'.format(str(float(grps[1])))
                if grps[3] is not None:
                    if float(grps[3]) == 0.000001:
                        return 'mc'
                    res += '$\lambda$={:.0e}'.format(float(grps[3]))
                res += ')'
        return res
    if name == 'mc_ll':
        return 'mc_drop_ll'
    if name.startswith('new_wdrop_exact'):
        match = re.compile(r'new_wdrop_exact_l=([1-9][0-9]*)(_p=(0.[0-9]+))?((\Z)|_(\w+))').match(name)
        if match is not None:
            grps = match.groups()
            if grps[0] is not None:
                l = int(grps[0])
                res = 'w_drop (L=%d' % l
                if grps[2] is not None:
                    p = float(grps[2])
                    res += ', p=%.2f' % p
                if grps[5] is None:
                    res += ')'
                    return res
    if name == 'mc_pu':
        return 'pu_mc'
    if name in ['pca_test_extrapolate', 'label_test_extrapolate']:
        return 'test (extrap.)'
    if name in ['pca_test_interpolate', 'label_test_interpolate']:
        return 'test (interp.)'
    if name == 'ws_dist':
        return 'Wasserstein distance'

    return name

def plot_densitymap(x, y, ax):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    x_range, y_range = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([x_range.ravel(), y_range.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    density = np.reshape(kernel(positions).T, x_range.shape)

    ax.imshow(np.rot90(density), cmap=plt.cm.gist_heat_r, extent=[xmin, xmax, ymin, ymax], aspect='equal')
    ax.plot(x, y, 'k.', markersize=1, alpha=0.1)

def plot_results(method_dict, file=None, with_valset=False):
    # set min-/max-values for all subplots in the next cell
    
    plt.clf()

    concatted = pd.DataFrame()
    split_idx_list = [0, 1, 2] if with_valset else [0, 1]
    test_idx = 2 if with_valset else 1
        
    means = [method_dict[key][i]['pred_mean'].values for key in method_dict for i in split_idx_list if 'pred_mean' in method_dict[key][i].columns.values]
    stds = [method_dict[key][i]['pred_std'].values for key in method_dict for i in split_idx_list if 'pred_std' in method_dict[key][i].columns.values]
    residuals = [method_dict[key][i]['pred_residual'].values for key in method_dict for i in split_idx_list if 'pred_residual' in method_dict[key][i].columns.values]
    normed_residuals = [method_dict[key][i]['pred_residual_normed'].values for key in method_dict for i in split_idx_list if 'pred_residual_normed' in method_dict[key][i].columns.values]
    
    def _check_number(x):
        return None if np.isnan(x) or np.isinf(x) else x
    
    max_pred_mean     = _check_number(np.quantile(np.concatenate(means), 0.98)) if len(means) > 0 else None
    min_pred_mean     = _check_number(np.quantile(np.concatenate(means), 0.02)) if len(means) > 0 else None
    max_pred_std      = _check_number(np.quantile(np.concatenate(stds), 0.98)) if len(stds) > 0 else None
    min_pred_std      = _check_number(np.quantile(np.concatenate(stds), 0.02)) if len(stds) > 0 else None
    max_pred_residual = _check_number(np.quantile(np.concatenate(residuals), 0.98)) if len(residuals) > 0 else None
    min_pred_residual = _check_number(np.quantile(np.concatenate(residuals), 0.02)) if len(residuals) > 0 else None
    max_pred_residual_normed = _check_number(np.quantile(np.concatenate(normed_residuals), 0.98)) if len(normed_residuals) > 0 else None
    min_pred_residual_normed = _check_number(np.quantile(np.concatenate(normed_residuals), 0.02)) if len(normed_residuals) > 0 else None
    
    # visualize all results

    num_methods  = method_dict.__len__()
    method_names = list(method_dict.keys())
    datasets     = ['train', 'val', 'test'] if with_valset else ['train', 'test']
    colors       = ['b', 'r', 'orange'] if with_valset else ['b', 'orange']

    fig, ax = plt.subplots(15,num_methods,figsize=(35,40), squeeze=False)
    
    for j,method in enumerate(method_dict):
        for i,df in enumerate(method_dict[method]):
        
            df.plot.scatter(x='gt',y='pred_mean',ax=ax[0,j],color=colors[i])
            
            if 'pred_std' in df.columns.values:
                df.plot.scatter(x='gt',y='pred_std',ax=ax[1,j],color=colors[i])
                df.plot.scatter(x='pred_residual',y='pred_std',ax=ax[4,j],color=colors[i])
                df.plot.scatter(x='pca0_projection',y='pred_std',ax=ax[11,j],color=colors[i])        
                     
            if 'total_std' in df.columns.values:
                df.plot.scatter(x='gt',y='total_std',ax=ax[2,j],color=colors[i])    
                df.plot.scatter(x='pred_residual',y='total_std',ax=ax[5,j],color=colors[i])
                
            if 'pred_residual_normed' in df.columns.values:
                ax[13,j].hist(df['pred_residual_normed'],bins=30,density=True,color=colors[i])

            if 'net_gradient_norm' in df.columns.values:
                df.plot.scatter(x='gt',y='net_gradient_norm',ax=ax[14,j],color=colors[i]) 
            
            df.plot.scatter(x='gt',y='pred_residual',ax=ax[3,j],color=colors[i])
            df.plot.scatter(x='pca0_projection',y='pred_mean',ax=ax[10,j],color=colors[i])
            df.plot.scatter(x='pca0_projection',y='pred_residual',ax=ax[12,j],color=colors[i])
            

        if method != 'vanilla':
            try:
                plot_densitymap(method_dict[method][0]['pred_residual'], method_dict[method][0]['pred_std'], ax[6, j])
            except Exception:
                print("Exception caught in plot_densitymap, skipping plot ... ", method_dict[method][0]['pred_residual'],  method_dict[method][0]['pred_std'])

            try:
                plot_densitymap(method_dict[method][0]['pred_residual'], method_dict[method][0]['total_std'], ax[7, j])
            except Exception:
                print("Exception caught in plot_densitymap, skipping plot ... ", method_dict[method][0]['pred_residual'],  method_dict[method][0]['total_std'])

            try:
                plot_densitymap(method_dict[method][test_idx]['pred_residual'], method_dict[method][test_idx]['pred_std'], ax[8, j])
            except Exception:
                print("Exception caught in plot_densitymap, skipping plot ... ", method_dict[method][test_idx]['pred_residual'],  method_dict[method][test_idx]['pred_std'])

            try:
                plot_densitymap(method_dict[method][test_idx]['pred_residual'], method_dict[method][test_idx]['total_std'], ax[9, j])
            except Exception:
                print("Exception caught in plot_densitymap, skipping plot ... ", method_dict[method][test_idx]['pred_residual'],  method_dict[method][test_idx]['total_std'])
        
        if min_pred_residual is not None and max_pred_residual is not None:
            line_sigma_1_data =  pd.DataFrame([[x,np.abs(x)] for x in np.linspace(min_pred_residual-0.2,max_pred_residual+0.2,200)])
            line_sigma_3_data =  pd.DataFrame([[x,np.abs(x/3)] for x in np.linspace(min_pred_residual-0.2,max_pred_residual+0.2,200)])
       
        line_sigma_1_data.plot(kind='line',x=0,y=1, color='k',ax=ax[4,j],alpha=1)
        line_sigma_3_data.plot(kind='line',x=0,y=1,color='r',ax=ax[4,j],alpha=1) 
        line_sigma_1_data.plot(kind='line',x=0,y=1,color='k',ax=ax[5,j],alpha=1)
        line_sigma_3_data.plot(kind='line',x=0,y=1,color='r',ax=ax[5,j],alpha=1)
        line_sigma_1_data.plot(kind='line',x=0,y=1,color='k',ax=ax[6,j],alpha=1)
        line_sigma_3_data.plot(kind='line',x=0,y=1,color='r',ax=ax[6,j],alpha=1) 
        line_sigma_1_data.plot(kind='line',x=0,y=1,color='k',ax=ax[7,j],alpha=1)
        line_sigma_3_data.plot(kind='line',x=0,y=1,color='r',ax=ax[7,j],alpha=1)
        line_sigma_1_data.plot(kind='line',x=0,y=1,color='k',ax=ax[8,j],alpha=1)
        line_sigma_3_data.plot(kind='line',x=0,y=1,color='r',ax=ax[8,j],alpha=1) 
        line_sigma_1_data.plot(kind='line',x=0,y=1,color='k',ax=ax[9,j],alpha=1)
        line_sigma_3_data.plot(kind='line',x=0,y=1,color='r',ax=ax[9,j],alpha=1)
        
        
        if min_pred_mean is not None and max_pred_mean is not None:
            ax[0,j].set_ylim([min_pred_mean,max_pred_mean])
            ax[11,j].set_ylim([min_pred_mean,max_pred_mean])
        
        if min_pred_std is not None and max_pred_std is not None:
            ax[1,j].set_ylim([min_pred_std,max_pred_std])
            ax[2,j].set_ylim([min_pred_std,max_pred_std])
            ax[4,j].set_ylim([min_pred_std,max_pred_std])
            ax[5,j].set_ylim([min_pred_std,max_pred_std])
            ax[6,j].set_ylim([min_pred_std,max_pred_std])
            ax[7,j].set_ylim([min_pred_std,max_pred_std])
            ax[8,j].set_ylim([min_pred_std,max_pred_std])
            ax[9,j].set_ylim([min_pred_std,max_pred_std])
            ax[11,j].set_ylim([min_pred_std,max_pred_std])
        
        if min_pred_residual is not None and max_pred_residual is not None:
            ax[3,j].set_ylim([min_pred_residual,max_pred_residual])
            ax[4,j].set_xlim([min_pred_residual-0.2,max_pred_residual+0.2])
            ax[5,j].set_xlim([min_pred_residual-0.2,max_pred_residual+0.2])
            ax[6,j].set_xlim([min_pred_residual-0.2,max_pred_residual+0.2])
            ax[7,j].set_xlim([min_pred_residual-0.2,max_pred_residual+0.2])
            ax[8,j].set_xlim([min_pred_residual-0.2,max_pred_residual+0.2])
            ax[9,j].set_xlim([min_pred_residual-0.2,max_pred_residual+0.2])
            ax[12,j].set_ylim([min_pred_residual,max_pred_residual])
            
        if min_pred_residual_normed is not None and max_pred_residual_normed is not None:
            ax[13,j].set_xlim([min_pred_residual_normed-5,max_pred_residual_normed+5])
            
        ax[4,j].set_xlabel('pred_residual')
        ax[5,j].set_xlabel('pred_residual')
        ax[6,j].set_xlabel('pred_residual')
        ax[7,j].set_xlabel('pred_residual')
        ax[8,j].set_xlabel('pred_residual')
        ax[9,j].set_xlabel('pred_residual')
        ax[13,j].set_xlabel('pred_residual_normed')
        ax[13,j].set_ylabel('pdf')
        #ax[7,j].set_yscale('log')

        for k in range(6):
            ax[k,j].set_title(method_names[j]+' (train/test data)')
        
        for k in range(6, 8):
            ax[k, j].set_title(method_names[j] + ' (train data)')
            ax[k, j].legend()
        
        for k in range(8, 10):
            ax[k, j].set_title(method_names[j] + ' (test data)')
            ax[k, j].legend()
        
        for k in range(10, 15):
            ax[k,j].set_title(method_names[j]+' (train/test data)')

    plt.tight_layout()
    
    if file is not None:
        plt.savefig(file)
    
    plt.clf()
    plt.close(fig)
    
def plot_aggregates(aggregated_mean, metric, datasets=None, splits=None, methods=None, s=200, ylim=None):
    
    if datasets is None:
        datasets = aggregated_mean.index.get_level_values(0).unique().values
        
    if splits is None:
        splits = aggregated_mean.index.get_level_values(1).unique().values
    
    if methods is None:
        methods = aggregated_mean.columns.get_level_values(0).unique().values
    
    fig, ax = plt.subplots(2, 1, figsize=(32, 18))
    iid_splits = [split for split in splits if split in {'train', 'test'}]
    val_iid = aggregated_mean.loc[(datasets, iid_splits), (methods, metric)].values.reshape((len(datasets), len(iid_splits), len(methods)))
    mean_iid = np.mean(val_iid, axis=0).flatten()
    q25_iid = np.quantile(val_iid, 0.25, axis=0).flatten()
    q75_iid = np.quantile(val_iid, 0.75, axis=0).flatten()
    x = [j*len(methods) + i + j for j in range(len(iid_splits)) for i in range(len(methods))]
    ax[0].scatter(x, mean_iid, marker='x', s=s)
    ax[0].scatter(x, q25_iid, marker='_', s=s, color='grey')
    ax[0].scatter(x, q75_iid, marker='_', s=s, color='grey')
    ax[0].vlines(x, q25_iid, q75_iid, color='grey', linewidth=2)
    
    for i in range(len(iid_splits)-1):
        ax[0].axvline((i+1)*len(methods) + i, color='black')
    
    ax[0].set_ylabel(metric)
    ax[0].set_xticks([j*len(methods) + i + j for j in range(len(iid_splits)) for i in range(len(methods))])
    ax[0].set_xticklabels([s if s != 'mc_mod_sml' else 'ours' for s in np.tile(methods, len(iid_splits))], rotation=45)
    if ylim is not None:
        ax[0].set_ylim(ylim)
        
    ood_splits = [split for split in splits if split in {'label_test_extrapolate', 'label_test_interpolate', 'pca_test_extrapolate', 'pca_test_interpolate'}]
    val_ood = aggregated_mean.loc[(datasets, ood_splits), (methods, metric)].values.reshape((len(datasets), len(ood_splits), len(methods)))
    mean_ood = np.mean(val_ood, axis=0).flatten()
    q25_ood = np.quantile(val_ood, 0.25, axis=0).flatten()
    q75_ood = np.quantile(val_ood, 0.75, axis=0).flatten()
    x = [j*len(methods) + i + j for j in range(len(ood_splits)) for i in range(len(methods))]
    ax[1].scatter(x, mean_ood, marker='x', s=s)
    ax[1].scatter(x, q25_ood, marker='_', s=s, color='grey')
    ax[1].scatter(x, q75_ood, marker='_', s=s, color='grey')
    ax[1].vlines(x, q25_ood, q75_ood, color='grey', linewidth=2)
    
    for i in range(len(ood_splits)-1):
        ax[1].axvline((i+1)*len(methods) + i, color='black')
        
    ax[1].set_ylabel(metric)
    ax[1].set_xticks([j*len(methods) + i + j for j in range(len(ood_splits)) for i in range(len(methods))])
    ax[1].set_xticklabels([s if s != 'mc_mod_sml' else 'ours' for s in np.tile(methods, len(ood_splits))], rotation=45)
    if ylim is not None:
        ax[1].set_ylim(ylim)
        
def plot_aggregates_oneline(aggregated_mean, metric, datasets=None, splits=None, methods=None, s=200, ylim=None, title_fontsize=30, ticklabel_rotation=45, use_boxplot=True, emphasize_method=None):
    
    if datasets is None:
        datasets = aggregated_mean.index.get_level_values(0).unique().values
        
    if splits is None:
        splits = aggregated_mean.index.get_level_values(1).unique().values
    
    if methods is None:
        methods = aggregated_mean.columns.get_level_values(0).unique().values
        
    iid_splits = [split for split in splits if split in {'train', 'test'}]
    ood_splits = [split for split in splits if split in {'label_test_extrapolate', 'label_test_interpolate', 'pca_test_extrapolate', 'pca_test_interpolate'}]
    
    
    fig, ax = plt.subplots(1, 1, figsize=(32, 12))
    val = aggregated_mean.loc[(datasets, splits), (methods, metric)].values.reshape((len(datasets), len(splits), len(methods)))
    
    if use_boxplot:
        capprops = {'linewidth': 5}
        whiskerprops = {'linestyle': 'solid', 'linewidth': 3}
        boxprops = {'linewidth': 3}
        flierprops = {'marker': 'x', 'markersize': 12, 'linestyle': 'none'}
        medianprops = {'linewidth': 6}
        meanprops = {'marker': 'o', 'markersize': 15, 'markerfacecolor': 'blue', 'markeredgecolor': 'blue'}
        for split in range(6):
            val_split = val[:, split, :]
            res = ax.boxplot(val_split, positions=[split*len(methods)+i+split for i in range(len(methods))], showmeans=True, widths=0.6,
                       whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops, boxprops=boxprops, medianprops=medianprops, meanprops=meanprops)
            for cap in res['caps']:
                cap.set_xdata(cap.get_xdata() + np.array([-.1,.1]))
            
    else:
        mean = np.mean(val, axis=0).flatten()
        median = np.median(val, axis=0).flatten()
        q25 = np.quantile(val, 0.25, axis=0).flatten()
        q75 = np.quantile(val, 0.75, axis=0).flatten()
        x = [j*len(methods) + i + j for j in range(len(splits)) for i in range(len(methods))]
        ax.scatter(x, q25, marker='_', s=s, color='grey')
        ax.scatter(x, q75, marker='_', s=s, color='grey')
        ax.vlines(x, q25, q75, color='grey', linewidth=2)
        ax.scatter(x, median, marker='_', s=s, color='orange')
        ax.scatter(x, mean, marker='x', s=s)
    
    if ylim is None:
        ylim = ax.get_ylim()
    
    for i in range(len(splits)):
        if i < len(splits)-1:
            lw = 6 if i in [len(iid_splits)-1, len(iid_splits)-1+len(ood_splits)/2.] else 4
            ax.axvline((i+1)*len(methods) + i, color='black', linewidth=lw)
        ax.text(len(methods)*i + i + len(methods)//2, ylim[1] + ylim[1]*(title_fontsize/500), name_map(splits[i]), horizontalalignment='center', verticalalignment='center', fontsize=title_fontsize)
    
    if emphasize_method is not None:
        emphasize_idx = list(methods).index(emphasize_method)
        for i in range(len(splits)):
            ax.axvspan(i*len(methods) + emphasize_idx - 0.5 + i, i*len(methods) + emphasize_idx + 0.5 + i, color='blue', alpha=0.1) 
    
    ax.margins(0.02)
    
    ax.text(len(methods), ylim[1]+ ylim[1]*(title_fontsize/400 + title_fontsize/400), 'i.i.d. data split', weight='bold', horizontalalignment='center', verticalalignment='center', fontsize=title_fontsize)
    ax.text(3*len(methods) +2, ylim[1] + ylim[1]*(title_fontsize/400 + title_fontsize/400), 'label-based data shift', weight='bold', horizontalalignment='center', verticalalignment='center', fontsize=title_fontsize)
    ax.text(5*len(methods) +4, ylim[1] + ylim[1]*(title_fontsize/400 + title_fontsize/400), 'pca-based data shift', weight='bold',horizontalalignment='center', verticalalignment='center', fontsize=title_fontsize)
   
    ax.set_ylabel(name_map(metric))
    ax.set_xticks([j*len(methods) + i + j for j in range(len(splits)) for i in range(len(methods))])
    
    ax.set_xticklabels([name_map(s) for s in np.tile(methods, len(splits))], rotation=ticklabel_rotation)
    if ylim is not None:
        ax.set_ylim(ylim)
        
def plot_x_vs_preds(method_dicts, methods=None, trte=[0, 1], rows=['gt', 'mean', 'res', 'gt_std', 'std'], fold_idx=0, s=5, interpol_gt=False, from_multiple=False, savefig=None, additional_data_dict=None):
    
    if not from_multiple:
        method_dicts_ = [method_dicts]
    else:
        method_dicts_ = method_dicts
    
    methods_ = {key: method_dict for method_dict in method_dicts_ for key in sorted(method_dict[fold_idx])}
    
    if methods is not None:
        methods_ = OrderedDict([(method, methods_[method]) for method in methods if method in set(methods_.keys()).intersection(set(methods))])
    
    n_methods = len(methods_)
    fig, ax = plt.subplots(len(trte)*len(rows), n_methods, figsize=(n_methods*8, len(rows)*5))
    
    trte_to_color = {0: 'orange', 1: 'blue'}
    ylims = [[([], []) for _ in range(len(rows))] for _ in range(len(trte))]
    
    for i, trte_i in enumerate(trte): # train/test
        for j, method in enumerate(methods_):
                
            gt = methods_[method][fold_idx][method][trte_i]['gt'].values
            x = np.array([val[0] for val in methods_[method][fold_idx][method][trte_i]['x'].values])
            x_unique = np.unique(x)
            grouped_gt = [gt[x == x_val] for x_val in x_unique]
            gt_std = [np.std(group) for group in grouped_gt]
            
            pred_mean = methods_[method][fold_idx][method][trte_i]['pred_mean']
            pred_std = methods_[method][fold_idx][method][trte_i]['total_std']
            residual = pred_mean - gt
            
            metrics_list = []

            k = 0
            if 'gt' in rows:
                if interpol_gt:
                    x_argsort = np.argsort(x)
                    ax[i*len(rows)+k, j].plot(x[x_argsort], gt[x_argsort], color=trte_to_color[trte_i])
                else:
                    ax[i*len(rows)+k, j].scatter(x, gt, s=s, color=trte_to_color[trte_i])
                metrics_list.append(gt)
                k += 1
            
            if 'mean' in rows:
                ax[i*len(rows)+k, j].scatter(x, pred_mean, s=s, color=trte_to_color[trte_i])
                metrics_list.append(pred_mean)
                k += 1
            
            if 'res' in rows:
                ax[i*len(rows)+k, j].scatter(x, residual, s=s, color=trte_to_color[trte_i])
                metrics_list.append(residual)
                k += 1

            if 'gt_std' in rows:
                ax[i*len(rows)+k, j].scatter(x_unique, gt_std, s=s, color=trte_to_color[trte_i])
                metrics_list.append(gt_std)
                k += 1
                
            if 'std' in rows:
                ax[i*len(rows)+k, j].scatter(x, pred_std, s=s, color=trte_to_color[trte_i])

                if j == 0:
                    ax[i*len(rows)+k, j].set_ylabel('std')
                    
                if additional_data_dict is not None and 'std' in additional_data_dict:
                    ad = additional_data_dict['std']
                    ax[i*len(rows)+k, j].scatter(ad['x'], ad['y'], s=s, color=ad['color'])
                
                metrics_list.append(pred_std)
            
            for plot_ident, data in enumerate(metrics_list):
                ylims[i][plot_ident][0].append(np.min(data))
                ylims[i][plot_ident][1].append(np.max(data))
    
    for j, method in enumerate(methods_):
        ax[0, j].set_title(name_map(method))
        ax[len(trte)*len(rows)-1, j].set_xlabel('x')
        
    for j, row in enumerate(rows):
        ax[j, 0].set_ylabel(row)
    
    for i, trte_i in enumerate(trte):
        for plot_ident, ylim_vals in enumerate(ylims[i]):
            ylims[i][plot_ident] = (np.min(ylims[i][plot_ident][0]),
                                    np.max(ylims[i][plot_ident][1]))
            
            ymin, ymax = ylims[i][plot_ident]
            if plot_ident == rows.index('mean'):
                ylims[i][plot_ident] = ylims[i][rows.index('gt')]
            else:
                ylims[i][plot_ident] = (ymin - (ymax - ymin)*0.2,
                                   ymax + (ymax - ymin)*0.2)
            
            for j in range(len(methods_)):
                ax[i*len(rows)+plot_ident, j].set_ylim(*ylims[i][plot_ident])
    
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    
def _75q(x):
    return np.quantile(x, .75)

def _25q(x):
    return np.quantile(x, .25)

def plot_metrics(aggregated_mean, metric, idents=('train', 'test', 'label_test_interpolate', 'label_test_extrapolate', 'pca_test_interpolate', 'pca_test_extrapolate'), 
                 datasets=None, methods=None, ylim=None, yscale=None, figsize=(10, 6), ax=None, 
                 summary_stat_over=None, summary_stat_funcs=None, summary_rank_funcs=None,
                ident_offsets=default_ident_offsets, fs_name='ours', ticklabel_tilt=0, s=10,
                savefig=None, xticklabels=True):
    
    show = False
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        show = True
    
    if datasets is None:
        datasets = aggregated_mean.index.get_level_values(0).unique().values
    else:
        datasets = np.array(datasets)
    
    if methods is None:
        methods = aggregated_mean.columns.get_level_values(0).unique().values
        
    datasets_idx = np.arange(datasets.size)
    max_dataset_idx = datasets_idx.max()
    
    if summary_stat_funcs is None:
        summary_stat_funcs = [np.mean, np.median, np.min, lambda x: np.quantile(x, 0.25), lambda x: np.quantile(x, 0.75), np.max]
    if summary_rank_funcs is None:
        summary_rank_funcs = [np.mean]
    
    aggregated_ranks = None
    if len(summary_rank_funcs) > 0:
        aggregated_ranks = aggregated_mean.loc[(datasets, idents), (methods, metric)].apply(
            get_ranks if metric != 'r2' else lambda x: get_ranks(x, higher_is_better=True), axis=1, result_type='broadcast')
    
    ident_str = {'label_test_interpolate': 'label_test_interp', 'label_test_extrapolate': 'label_test_extrap', 'pca_test_interpolate': 'pca_test_interp', 'pca_test_extrapolate': 'pca_test_extrap'}
    
    for ident in idents:
        for method in sorted(methods):
            
            values_over_datasets = aggregated_mean.loc[(datasets, ident), (method, metric)].values
            ax.scatter(datasets_idx + ident_offsets[ident], 
                        values_over_datasets,
                       s=s,
                       label='%s, %s' % (method, ident_str[ident] if ident in ident_str else ident),
                       marker=method_to_marker[method],
                       color=ident_to_color[ident],
                       alpha=0.5)
            
            if summary_stat_over is None:
                summary_values = values_over_datasets
            else:
                summary_values = aggregated_mean.loc[(summary_stat_over, ident), (method, metric)].values
            
            for summary_stat_count, summary_stat_func in enumerate(summary_stat_funcs):
                if callable(summary_stat_func):
                    ax.scatter(max_dataset_idx + summary_stat_count + 1 + ident_offsets[ident], 
                               summary_stat_func(summary_values),
                               s=s,
                               marker=method_to_marker[method],
                               color=ident_to_color[ident],
                               alpha=0.5)
    
    
    if ylim is None:
        ylim = ax.get_ylim()
        
    if summary_rank_funcs is not None and len(summary_rank_funcs) > 0:
        for ident in idents:
            for method in sorted(methods):
                if summary_stat_over is None:
                    rank_summary_values = aggregated_ranks.loc[(datasets, ident), (method, metric)].values
                else:
                    rank_summary_values = aggregated_ranks.loc[(summary_stat_over, ident), (method, metric)].values
                
                rank_summary_values = ((ylim[1]-ylim[0])/len(methods))*rank_summary_values + ylim[0]
                
                for summary_stat_count, summary_rank_func in enumerate(summary_rank_funcs):
                    if callable(summary_stat_func):
                        ax.scatter(max_dataset_idx + summary_stat_count + len(summary_stat_funcs) + 1 + ident_offsets[ident], 
                                   summary_rank_func(rank_summary_values),
                                   s=s,
                                   marker=method_to_marker[method],
                                   color=ident_to_color[ident],
                                   alpha=0.5)
            
                   
    a = 0.2
    ax.plot([-0.5, -0.5], ylim, '--', color='grey', alpha=a)
    for ds_idx in datasets_idx:
        if ds_idx == max_dataset_idx:
            a = 0.5
        else:
            a = 0.2
        ax.plot([ds_idx+0.5, ds_idx+0.5], ylim, '--', color='grey', alpha=a)
    
    for ds_idx in range(max_dataset_idx + 1, max_dataset_idx + len(summary_stat_funcs) + len(summary_rank_funcs) +1):
        ax.plot([ds_idx+0.5, ds_idx+0.5], ylim, '--', color='grey', alpha=0.2)
    ax.axvspan(max_dataset_idx + 1 - 0.5, max_dataset_idx + len(summary_stat_funcs) + len(summary_rank_funcs) + 1.5, color='grey', alpha=0.05)
            
    ax.legend(prop=fontP, bbox_to_anchor=(1, 1), loc='upper left')
    
    if yscale is None:
        ax.set_yscale('log')
    else:
        ax.set_yscale(yscale)

    reduce_func_to_str = {np.mean: 'mean', np.median: 'median', np.min: 'min', np.max: 'max', _75q: '75q', _25q: '25q'}
        
    ax.set_ylabel(metric if metric != 'ws_dist' else 'Wasserstein distance')
    
    
    ax.set_xticks(np.concatenate((datasets_idx, np.arange(max_dataset_idx + 1, max_dataset_idx + len(summary_stat_funcs) + len(summary_rank_funcs) +1))))
    if xticklabels:
        ax.set_xticklabels(np.concatenate(([dataset + "\n" + ("(%.0fk)"% (float(dataset_to_size[dataset])/1000.) if (float(dataset_to_size[dataset]) >= 1000) else "(%.1fk)"%(float(dataset_to_size[dataset])/1000.) ) for dataset in datasets], 
                                       [reduce_func_to_str[func] if func in reduce_func_to_str else func.__name__ for func in summary_stat_funcs],
                                      [reduce_func_to_str[func] + " rank" if func in reduce_func_to_str else func.__name__ for func in summary_rank_funcs])),
                          rotation=ticklabel_tilt)
    else:
        ax.set_xticklabels([])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(-1.25, max_dataset_idx + len(summary_stat_funcs) + len(summary_rank_funcs) + 1.25)
    
    if savefig is not None:
        plt.savefig(savefig)
    
    if show:
        plt.show()
    

def plot_uncertainty_vs_performance(aggregated_mean, perf, unc, item, second_dim_vals=None, second_dim='datasets',
                                    idents=('train', 'test', 'label_test_interpolate', 'label_test_extrapolate', 'pca_test_interpolate', 'pca_test_extrapolate'), 
                                    xlim=None, ylim=None, xscale=None, figsize=(10, 6)):
    
    plt.figure(figsize=figsize)
    
    
    if second_dim == 'datasets':
        
        if second_dim_vals is None:
            second_dim_vals = aggregated_mean.index.get_level_values(0).unique().values
        
        for dataset_id in second_dim_vals:
            for ident in idents:
                plt.scatter(aggregated_mean.loc[(dataset_id, ident), (item, perf)],
                            aggregated_mean.loc[(dataset_id, ident), (item, unc)],
                           color=ident_to_color[ident],
                           label='%s_%s' % (dataset_id, ident),
                            marker=dataset_to_marker[dataset_id],
                           alpha=0.5) 
                
        plt.title('Method=%s' % item)
    
    elif second_dim == 'methods':
        
        if second_dim_vals is None:
            second_dim_vals = aggregated_mean.columns.get_level_values(0).unique().values
        
        for method in second_dim_vals:
            for ident in idents:
                plt.scatter(aggregated_mean.loc[(item, ident), (method, perf)],
                            aggregated_mean.loc[(item, ident), (method, unc)],
                           color=ident_to_color[ident],
                            marker=method_to_marker[method],
                           label='%s_%s' % (method, ident),
                           alpha=0.5) 
                
        plt.title('Dataset=%s' % item)
    
    plt.xlabel(perf)
    plt.ylabel(unc)
    plt.legend(prop=fontP, bbox_to_anchor=(1, 1), loc='upper left')
    plt.xscale('symlog')
    plt.yscale('linear')
    
    if xlim is not None:
        plt.xlim(*xlim)
    
    if ylim is not None:
        plt.ylim(*ylim)
        
    if xscale is not None:
        plt.xscale(xscale)
    
    plt.show()

def plot_uncertainty_vs_uncertainty(aggregated_mean, metrics, item, second_dim='datasets', second_dim_vals=None,
                                    idents=('train', 'test', 'label_test_interpolate', 'label_test_extrapolate', 'pca_test_interpolate', 'pca_test_extrapolate'), 
                                    ylim=None, xlim=None, xscale=None, figsize=(10, 6), ax=None, title=True, legend=True, s=40):
    
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    
    for metric_pair in itert.combinations(metrics, 2):
        unc1, unc2 = metric_pair
        
        for ident in idents:
            
            if second_dim == 'datasets':
                if second_dim_vals is None:
                    datasets = aggregated_mean.index.get_level_values(0).unique().values
                else:
                    datasets = second_dim_vals
                    
                for dataset_id in datasets:
                    ax.scatter(aggregated_mean.loc[(dataset_id, ident), (item, unc1)],
                                aggregated_mean.loc[(dataset_id, ident), (item, unc2)],
                                label='%s_%s_%s' % (unc1, unc2, ident),
                                color=ident_to_color[ident],
                                marker=dataset_to_marker[dataset_id],
                                alpha=0.5, s=s)
                    
                    plt.title('dataset=%s' % item)
            elif second_dim == 'methods':
                if second_dim_vals is None:
                    methods = aggregated_mean.columns.get_level_values(0).unique().values
                else:
                    methods = second_dim_vals
                    
                for method in methods:
                    ax.scatter(aggregated_mean.loc[(item, ident), (method, unc1)],
                                aggregated_mean.loc[(item, ident), (method, unc2)],
                               label='%s_%s_%s' % (unc1, unc2, ident),
                                color=ident_to_color[ident],
                                marker=method_to_marker[method],
                                alpha=0.5, s=s
                               )
                    
                if title:
                    ax.set_title('dataset=%s' % item)
        
        if legend:
            ax.legend(prop=fontP, bbox_to_anchor=(1, 1), loc='upper left')
        
        if len(metrics) == 2:
            ax.set_xlabel(metrics[0])
            ax.set_ylabel(metrics[1])
        else:
            ax.set_xlabel('uncertainty')
            ax.set_ylabel('uncertainty')
        
        if xscale is not None:
            ax.set_xscale(xscale)
        else:
            ax.set_xscale('log')
        
        if ylim is not None:
            ax.set_ylim(*ylim)
            
        if xlim is not None:
            ax.set_xlim(*xlim)
        
        #plt.show()
        
def plot_ood_behavior_vs_mc(exp_dirs, compare_method, baseline_method, datasets=None, markerstyle='o', legendmarkersize=50, col=4, alpha=0.2, textsize=25, emphasize_method=None):
    
    exp_dir_to_datasets = {exp_dir: [] for exp_dir in exp_dirs} 
    n_datasets = 0
    for exp_dir in exp_dirs:
        datasets_ = os.listdir(exp_dir)
        
        if datasets is not None:
            datasets_ = [ds for ds in datasets if ds in datasets_]
        
        for dataset_id in datasets_:
            if dataset_id in available_datasets:
                exp_dir_to_datasets[exp_dir].append(dataset_id)
                n_datasets += 1

    row = 2*int(np.ceil(n_datasets/col))
    fig, ax = plt.subplots(row, col, figsize=(col*10, row*8))

    count = 0
    for exp_dir in exp_dir_to_datasets:
        for dataset_id in exp_dir_to_datasets[exp_dir]:

            dir_files = get_dir_files(exp_dir, dataset_id)
            method_dict = load_method_dict(dir_files, 'single_label_split')

            gt = method_dict[0][baseline_method][0]['gt'].values
            max_gt_tr = np.max(gt)
            gt = np.concatenate((method_dict[0][baseline_method][1]['gt'].values, gt))
            gt_order = np.argsort(gt)

            binsize = dataset_to_size[dataset_id] // 20
            gt_binned = gt[gt_order][:-(gt.shape[0] % binsize)].reshape((-1, binsize))
            mean_gt_bins = np.concatenate((np.median(gt_binned, axis=1), [np.median(gt[gt_order][-(gt.shape[0] % binsize):])]))

            for method in [baseline_method, compare_method]:
                pred_mean = method_dict[0][method][0]['pred_mean'].values
                pred_std = method_dict[0][method][0]['pred_std'].values

                pred_mean = np.concatenate((method_dict[0][method][1]['pred_mean'].values, pred_mean))
                pred_std = np.concatenate((method_dict[0][method][1]['pred_std'].values, pred_std))

                ax[count // col, count % col].scatter(gt[gt_order], pred_std[gt_order], marker=markerstyle, alpha=alpha)

                pred_std_binned = pred_std[gt_order][:-(gt.shape[0] % binsize)].reshape((-1, binsize))
                mean_pred_std_bins = np.concatenate((np.median(pred_std_binned, axis=1), [np.median(pred_std[gt_order][-(gt.shape[0] % binsize):])]))
                #ax[count // col, count % col].plot(mean_gt_bins, mean_pred_std_bins, linewidth=4, color='k')
                ax[count // col, count % col].plot(mean_gt_bins, mean_pred_std_bins, label=name_map(method))


            ylim = ax[count // col, count % col].get_ylim()
            ax[count // col, count % col].axvline(max_gt_tr, color='k')
            ax[count // col, count % col].text(max_gt_tr - 0.25* (textsize/ 25.), (1-0.15*textsize/25.)*ylim[1], 'in-data', horizontalalignment='center',      verticalalignment='center', fontsize=textsize,fontweight='bold', rotation=90)
            ax[count // col, count % col].text(max_gt_tr + 0.25* (textsize/ 25.), (1-0.15*textsize/25.)*ylim[1], 'ood', horizontalalignment='center', verticalalignment='center', fontsize=textsize,fontweight='bold', rotation=90)
            ax[count // col, count % col].set_title('"%s"'%dataset_id)
            ax[count // col, count % col].set_ylabel('std')
            ax[count // col, count % col].set_xlabel('label')
            lgnd = ax[count // col, count % col].legend(loc='upper left')
            #for handle in lgnd.legendHandles:
            #    handle.set_sizes([legendmarkersize])
            #    handle.set_alpha(1)
            count += 1

    for exp_dir in exp_dir_to_datasets:
        for dataset_id in exp_dir_to_datasets[exp_dir]:

            dir_files = get_dir_files(exp_dir, dataset_id)
            method_dict = load_method_dict(dir_files, 'single_pca_split')

            pca0 = method_dict[0][baseline_method][0]['pca0_projection'].values
            max_pca0_tr = np.max(pca0)
            pca0 = np.concatenate((method_dict[0][baseline_method][1]['gt'].values, pca0))
            pca0_order = np.argsort(pca0)

            binsize = dataset_to_size[dataset_id] // 20
            pca0_binned = pca0[pca0_order][:-(pca0.shape[0] % binsize)].reshape((-1, binsize))
            mean_pca0_bins = np.concatenate((np.median(pca0_binned, axis=1), [np.median(pca0[pca0_order][-(pca0.shape[0] % binsize):])]))

            for method in [baseline_method, compare_method]:
                pred_mean = method_dict[0][method][0]['pred_mean'].values
                pred_std = method_dict[0][method][0]['pred_std'].values

                pred_mean = np.concatenate((method_dict[0][method][1]['pred_mean'].values, pred_mean))
                pred_std = np.concatenate((method_dict[0][method][1]['pred_std'].values, pred_std))

                ax[count // col, count % col].scatter(pca0[pca0_order], pred_std[pca0_order], marker=markerstyle, alpha=alpha)

                pred_std_binned = pred_std[pca0_order][:-(pca0.shape[0] % binsize)].reshape((-1, binsize))
                mean_pred_std_bins = np.concatenate((np.median(pred_std_binned, axis=1), [np.median(pred_std[pca0_order][-(pca0.shape[0] % binsize):])]))
                ax[count // col, count % col].plot(mean_pca0_bins, mean_pred_std_bins, label=name_map(method))

            ylim = ax[count // col, count % col].get_ylim()
            ax[count // col, count % col].axvline(max_pca0_tr, color='k')
            ax[count // col, count % col].text(max_pca0_tr - 0.25 * (textsize/ 25.), (1-0.15*textsize/25.)*ylim[1], 'in-data', horizontalalignment='center',      verticalalignment='center', fontsize=textsize, fontweight='bold', rotation=90)
            ax[count // col, count % col].text(max_pca0_tr + 0.25* (textsize/ 25.), (1-0.15*textsize/25.)*ylim[1], 'ood', horizontalalignment='center', verticalalignment='center', fontsize=textsize, fontweight='bold', rotation=90)
            ax[count // col, count % col].set_title('"%s"'%dataset_id)
            ax[count // col, count % col].set_ylabel('std')
            ax[count // col, count % col].set_xlabel('1. pca comp. (input)')
            lgnd = ax[count // col, count % col].legend(loc='upper left')
            #for handle in lgnd.legendHandles:
            #    handle.set_sizes([legendmarkersize])
            #    handle.set_alpha(1)
            count += 1
        
        plt.tight_layout()
        
def show_sigmaplots(exp_dirs, methods, datasets=None, splitmode = 'single_random_split', use_heat=False, savefig=None):
    
    if datasets is None:
        datasets = [ds for exp_dir in exp_dirs for ds in os.listdir(exp_dir) if ds in available_datasets]
    
    
    plt.clf()
    fig, ax = plt.subplots(len(datasets), len(methods), figsize=(len(methods)*10, len(datasets)*10), squeeze=False)
    
    all_x, all_y = np.empty(0), np.empty(0)
    datasets_used, methods_used = [], []
    for exp_dir in exp_dirs:
        dataset_dirs = os.listdir(exp_dir)
        
        for row_i, dataset_id in enumerate(datasets):
            if dataset_id not in dataset_dirs:
                continue
            
            dir_files = get_dir_files(exp_dir, dataset_id)
            method_dict = load_method_dict(dir_files, splitmode)
            if method_dict is None or len(method_dict) == 0:
                continue
            
            for col_i, method in enumerate(methods): 
                if method not in method_dict[0]: # using first fold
                    continue
                
                test_df = method_dict[0][method][1]
                x, y = test_df['pred_residual'].values, test_df['pred_std'].values
                all_x, all_y = np.concatenate((x, all_x)), np.concatenate((y, all_y))
                if use_heat:
                    plot_densitymap(x, y, ax[row_i, col_i])
                else:
                    ax[row_i, col_i].scatter(x, y)
                
                if dataset_id not in datasets_used:
                    datasets_used.append(dataset_id)
                if method not in methods_used:
                    methods_used.append(method)
    
    xmin, ymin = np.quantile(all_x, 0.03), 0#np.quantile(all_y, 0.02)
    xmax, ymax = np.quantile(all_x, 0.96), np.quantile(all_y, 0.96)
    xmin, xmax = min(xmin, -xmax),  max(-xmin, xmax) # symmetric x
    for i, dataset_id in enumerate(datasets_used):
        for j, method in enumerate(methods_used):
            
            if i == (len(datasets_used) - 1):
                ax[i, j].set_xlabel('pred_residual')
            
            if i == 0:
                ax[i, j].set_title('%s' % name_map(method) )
            
            if j == 0:
                ax[i, j].set_ylabel('pred_std')
                
            ax[i, j].plot([xmin, 0, xmax], [abs(xmin), 0, xmax], color='orange', label=r'$1 \sigma$')
            ax[i, j].plot([xmin, 0, xmax], [(1./3)*abs(xmin), 0, (1./3)*xmax], color='b', label=r'$3 \sigma$')
            ax[i, j].set_xlim(xmin, xmax)
            ax[i, j].set_ylim(0, ymax)
            ax[i, j].legend()
    
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
        
