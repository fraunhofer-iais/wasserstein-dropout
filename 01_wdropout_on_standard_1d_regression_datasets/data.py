import re
import os
import gzip
import json

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import levy, lognorm, pareto, cauchy, t, gumbel_r, invweibull

from multiprocessing.pool import Pool as ProcessPool
from multiprocessing import cpu_count

available_datasets = {'boston', 'concrete', 'energy', 'abalone', 'naval', 
                      'power', 'protein', 'wine_red', 'yacht', 'year', 
                      'california', 'diabetes', 'superconduct', 'kin8nm',
                      'toy_modulated', 'toy_hf'}

available_splits = {'random_folds', 'single_random_split', 'single_label_split', 'label_folds', 'single_pca_split', 'pca_folds'}


# Computes gaussian * sine; Represents the noise/uncertainty of the main polynomial function
def sine_bump(centre, std, amplitude, frequency):
    def sine_bump_instance(x):
        return amplitude * np.exp( -((x-centre)**2) / (2*std**2) ) * np.sin(frequency*x)

    return sine_bump_instance

# third degree polynomial + uncertainty (sine * gaussian)
def poly_fluct(x, centre=-1, std=1, amplitude=4000, frequency=2):
    return 0.01*((5*x)**2-(1*x)**3+sine_bump(centre,std,amplitude,frequency)(x))

# Like poly_fluct but sine has frequency 1 (why does this represent the "mean"?)
def poly_fluct_mean(x, centre=-1, std=10, amplitude=4000, frequency=1):
    return 0.01 * ((5 * x) ** 2 - (1 * x) ** 3 + sine_bump(centre, std, amplitude, frequency)(x))

# Takes the absolute value of the uncertainty curve
def poly_fluct_sigma(x):
    return np.abs(sine_bump(12, 5, 10, 2)(x))

# samples from a gaussian with the third degree polynomial evaluated at x as mean and the absolute uncertainty curve eval. at x as sigma 
def poly_fluct_sigma_fluct_normal(x,sample_size, centre_1=-1, std_1=1, amplitude_1=4000, frequency_1=2, 
                                  centre_2=12, std_2=5, amplitude_2=1200, frequency_2=0.1, added_std=0):
    return 0.01*(np.random.normal(100*poly_fluct(x, centre_1, std_1, amplitude_1, frequency_1),
                                  np.abs(sine_bump(centre_2,std_2,amplitude_2,frequency_2)(x))+added_std,sample_size))

# reads plain text file without header, seperation by arbitrary number of whitespace
# converts to float
def plain_table_reader(file):
    res = []
    with open(file) as f:
        for line in f:
            str_feats = [feat.strip() for feat in re.split(r'\s+', line)]
            float_feats = [float(feat) for feat in str_feats if len(feat) > 0]
            if len(float_feats) > 0:
                res.append(float_feats)
    return np.asarray(res)

def load_dataset(id):
    
    """ Toy data - Fluctation, Gaussian white noise, ..."""
    
    
    if id == 'toy_hf':
        lb, ub, size = -15, 20, 1000 #-20, 30, 1000
        x_range = np.linspace(lb, ub, size)
        X = x_range[:, None]
        y = poly_fluct_mean(x_range, frequency=3)
        return X, y
    
    if id == 'toy_modulated':
        sample_size = 10
        lb, ub, steps = -15, 15, 2000
        data_range = np.linspace(lb, ub, steps)
        X = np.repeat(data_range, sample_size)[:, None]
        y = np.concatenate([np.random.normal(0,np.exp(-0.02*i**2),sample_size) for i in data_range])
        return X, y
    
    """
    UCI datasets , 1D -regression
    """

    # Features: 13, Points: 506
    if id == 'boston':
        boston = load_boston()
        return boston['data'], boston['target']
    
    # features: 8, points: 20640
    if id == 'california':
        california = fetch_california_housing()
        return california['data'], california['target']
    
    # features: 7, points: 442
    if id == 'diabetes':
        diabetes = load_diabetes()
        return diabetes['data'], diabetes['target']
    
    #http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
    # features: 8, points: 1030
    if id == 'concrete':
        concrete = pd.ExcelFile('./data/Concrete_Data.xls').parse()
        concrete = concrete.to_numpy()
        return concrete[:, :-1], concrete[:, -1]
        
    #https://archive.ics.uci.edu/ml/datasets/Energy+efficiency
    # features: 8, points: 768; 2 gt labels (using latter one)
    if id == 'energy':
        energy_n_feat = 8
        energy_n_gt = 2
        energy = pd.ExcelFile('./data/ENB2012_data.xlsx').parse()
        energy = energy.to_numpy()
        assert(energy.shape[1] == (energy_n_feat + energy_n_gt))
        return energy[:, :-energy_n_gt], energy[:, -1] # note: using cooling load gt only #energy[:, -energy_n_gt:]
    
    #https://archive.ics.uci.edu/ml/datasets/abalone
    # features: 8 (using only 7, first feature is ignored), points: 4176
    if id == 'abalone':
        abalone = pd.read_csv('./data/abalone.data')
        abalone = abalone.to_numpy()[:, 1:].astype(np.float64) # ignoring first feature which is categorical
        return abalone[:, :-1], abalone[:, -1]
    
    #https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants
    #features: 16, points: 11934, has 2 gt labels, using the latter one
    if id == 'naval':
        naval_n_feat = 16
        naval_n_gt = 2
        naval = plain_table_reader('./data/UCI CBM Dataset/data.txt')
        return naval[:, :-naval_n_gt], naval[:, -1] # note: using turbine gt only #naval[:, -naval_n_gt:]
    
    #https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
    if id == 'power':
        power = pd.ExcelFile('./data/CCPP/Folds5x2_pp.xlsx').parse()
        power = power.to_numpy()
        return power[:, :-1], power[:, -1]
    
    #https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure
    if id == 'protein':
        protein = pd.read_csv('./data/CASP.csv')
        protein = protein.to_numpy()
        return protein[:, 1:], protein[:, 0]
    
    #https://archive.ics.uci.edu/ml/datasets/wine+quality
    # features: 11, points: 1599
    if id == 'wine_red':
        wine_red = pd.read_csv('./data/winequality-red.csv', sep=';')
        wine_red = wine_red.to_numpy()
        return wine_red[:, :-1], wine_red[:, -1]
    
    #http://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics
    # features: 6, points: 308
    if id == 'yacht':
        yacht = plain_table_reader('./data/yacht_hydrodynamics.data')
        return yacht[:, :-1], yacht[:, -1]
    
    #https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
    # features: 90, points: 515345
    if id == 'year':
        year = pd.read_csv('./data/YearPredictionMSD.txt', header=None)
        year = year.to_numpy()
        return year[:, 1:], year[:, 0]
    
    # features: 81, points: 21263
    if id == 'superconduct':
        superconduct = pd.read_csv('./data/superconduct/train.csv')
        superconduct = superconduct.to_numpy()
        return superconduct[:, :-1], superconduct[:, -1]

    # features: 9, points: 8192 # https://www.openml.org/d/189
    if id == 'kin8nm':
        kin8nm = pd.read_csv('./data/dataset_2175_kin8nm.arff')
        kin8nm = kin8nm.to_numpy()
        return kin8nm[:, :-1], kin8nm[:, -1]

def compute_pca_projections(X):
    pca_scaler = StandardScaler()  # each feature centered around mean with std = 1
    X_scaled = pca_scaler.fit_transform(X)
    pca = PCA(n_components=min(X_scaled.shape[1], 5))
    pca.fit(X_scaled)
    projections = np.matmul(X_scaled, pca.components_[0])
    return projections
  
def compute_idx_splits(X, y, fold_idxs=None, train_perc=0.5, val_perc=0., splits=None, folds_with_val=False, val_split_kind='random_from_train'):
    
    n_folds = 10
    if fold_idxs is None:
        fold_idxs = list(range(n_folds))
    else:
        fold_idxs = np.array(fold_idxs)
        n_folds = len(fold_idxs)
        if np.any((fold_idxs < 0) | (fold_idxs > 9) ):
            raise Exception("Given fold_idxs have to lie in [0, 9]")
            
    if train_perc + val_perc > 1.:
        raise Exception("train_perc + val_perc > 1")
    
    res = dict()
    
    n_data = X.shape[0]
    assert(n_data == len(y))
    
    n_test = n_data // 10
    n_train = n_data - n_test
    
    if folds_with_val:
        n_val = n_data // 5
        n_train = n_train - n_val
    
    idxs_random = np.random.choice(n_data, size=n_data, replace=False)
    
    if 'random_folds' in splits:
        folds = []
        for i in fold_idxs:
            
            start_test = i*n_test
            end_test = start_test + n_test
            
            test = idxs_random[start_test:end_test]
            if folds_with_val:
                start_val = (start_test - n_val) % n_data 
                end_val = start_test
                
                if start_val > end_val: # val part splits in two sections
                    train = idxs_random[end_test:start_val]
                    val = np.concatenate((idxs_random[0:start_test], idxs_random[start_val:]))
                else:
                    train = np.concatenate((idxs_random[0:start_val], idxs_random[end_test:]))
                    val = idxs_random[start_val:end_val]
            
                folds.append((train, val, test))
                
            else:
                train = np.concatenate((idxs_random[0:start_test], idxs_random[end_test:]))
                folds.append((train, test))
           
        res['random_folds'] = folds
    
    if 'single_random_split' in splits:
        n_single_train = int(train_perc*n_data)
        
        train = idxs_random[:n_single_train]
        if val_perc > 0:
            n_single_val = int(val_perc*n_data)
            val = idxs_random[n_single_train:n_single_train+n_single_val]
            test = idxs_random[n_single_train+n_single_val:]
            res['single_random_split'] = (train, val, test)
        else:
            test = idxs_random[n_single_train:]
            res['single_random_split'] = (train, test)
    
    if 'single_label_split' in splits:
        y_train_quantile = np.quantile(y, train_perc)
        idxs_y_train = np.where(y <= y_train_quantile)[0]
        
        if val_perc > 0:
            if val_split_kind == 'split_all':
                y_val_quantile = np.quantile(y, train_perc + val_perc)
                idxs_y_val = np.where((y <= y_val_quantile) & (y > y_train_quantile))[0]
                idxs_y_test = np.where(y > y_val_quantile)[0]
            elif val_split_kind == 'random_from_train':
                y_train_val_quantile = np.quantile(y, train_perc + val_perc)
                idxs_y_train_val = np.where(y <= y_train_val_quantile)[0]
                train_val_split_idxs = np.random.choice(len(idxs_y_train_val), size=len(idxs_y_train_val), replace=False)
                idxs_y_train = idxs_y_train_val[train_val_split_idxs[:int(n_data*train_perc)]]
                idxs_y_val = idxs_y_train_val[train_val_split_idxs[int(n_data*train_perc):]]
                idxs_y_test = np.where(y > y_train_val_quantile)[0]
            res['single_label_split'] = (idxs_y_train, idxs_y_val, idxs_y_test)
        else:
            idxs_y_test = np.where(y > y_train_quantile)[0]
            res['single_label_split'] = (idxs_y_train, idxs_y_test)
    
    def compute_val_ranges(lower_test, quant_range=0.2):
        if lower_test >= quant_range:
            upper = lower_test
            lower = upper - quant_range
        else:
            lower = 1 + lower_test - quant_range
            upper = quant_range - (1. - lower)
            if np.isclose(upper, 0.):
                upper = 1.

        return np.round(lower, 4), np.round(upper, 4)
    
    if 'label_folds' in splits:
        test_quantile_fold_range = 1. / 10.
        fold_label = []
        for i in fold_idxs:

            lower_quantile_test_ = i * test_quantile_fold_range
            upper_quantile_test_ = lower_quantile_test_ + test_quantile_fold_range
            lower_quantile_test = np.quantile(y, lower_quantile_test_)
            upper_quantile_test = np.quantile(y, upper_quantile_test_)
            
            test = np.where((lower_quantile_test <= y) & (y <= upper_quantile_test))[0]

            if folds_with_val and val_split_kind == 'split_all':
                val_quantile_fold_range = 0.2
                lower_quantile_val_, upper_quantile_val_ = compute_val_ranges(lower_quantile_test_, val_quantile_fold_range)

                lower_quantile_val = np.quantile(y, lower_quantile_val_)
                upper_quantile_val = np.quantile(y, upper_quantile_val_)
                if lower_quantile_val_ > upper_quantile_val_:
                    val = np.where((y < upper_quantile_val) | (y >= lower_quantile_val))[0]
                    train = np.where((y > upper_quantile_test) & (y < lower_quantile_val))[0]
                elif upper_quantile_test_ < lower_quantile_val_:
                    val = np.where((lower_quantile_val <= y) & (y <= upper_quantile_val))[0]
                    train = np.where((upper_quantile_test < y) & (y < lower_quantile_val))[0]       
                else:
                    val = np.where((lower_quantile_val <= y) & (y < upper_quantile_val))[0]
                    train = np.where((y < lower_quantile_val) | (y > upper_quantile_test))[0]
                
                fold_label.append((train, val, test))
            else:
                train = np.where((y < lower_quantile_test) | (y > upper_quantile_test))[0]
                if folds_with_val and val_split_kind == 'random_from_train':
                    train_ = train.copy()
                    train_val_idxs = np.random.choice(len(train), size=len(train), replace=False)
                    train = train_[train_val_idxs[:int(len(train)*(7./9.))]] # train makes out 0.9 of all data, but needs to be 0.7
                    val = train_[train_val_idxs[int(len(train)*(7./9.)):]]
                    fold_label.append((train, val, test))
                else:
                    fold_label.append((train, test))
                
        res['label_folds'] = fold_label
    
    projections = compute_pca_projections(X)
    res['projections'] = projections
    
    if 'single_pca_split' in splits:
        projections_train_quantile = np.quantile(projections, train_perc)
        idxs_train_pca0 = np.where(projections <= projections_train_quantile)[0]
        
        if val_perc > 0:
            if val_split_kind == 'split_all':
                projections_val_quantile = np.quantile(projections, train_perc + val_perc)
                idxs_val_pca0 = np.where((projections <= projections_val_quantile) & (projections > projections_train_quantile))[0]
                idxs_test_pca0 = np.where(projections > projections_val_quantile)[0]
            
            elif val_split_kind == 'random_from_train':
                projections_train_val_quantile = np.quantile(projections, train_perc + val_perc)
                idxs_projections_train_val = np.where(projections <= projections_train_val_quantile)[0]
                train_val_split_idxs = np.random.choice(len(idxs_projections_train_val), size=len(idxs_projections_train_val), replace=False)
                idxs_train_pca0 = idxs_projections_train_val[train_val_split_idxs[:int(n_data*train_perc)]]
                idxs_val_pca0 = idxs_projections_train_val[train_val_split_idxs[int(n_data*train_perc):]]
                idxs_test_pca0 = np.where(projections > projections_train_val_quantile)[0]
            res['single_pca_split'] = (idxs_train_pca0, idxs_val_pca0, idxs_test_pca0)
        else:
            idxs_test_pca0 = np.where(projections > projections_train_quantile)[0]
            res['single_pca_split'] = (idxs_train_pca0, idxs_test_pca0)
    
    if 'pca_folds' in splits:
        test_quantile_fold_range = 1. / 10.
        fold_pca0 = []
        for i in fold_idxs:
            lower_quantile_test_ = i * test_quantile_fold_range
            upper_quantile_test_ = lower_quantile_test_ + test_quantile_fold_range
            lower_quantile_test = np.quantile(projections, lower_quantile_test_)
            upper_quantile_test = np.quantile(projections, upper_quantile_test_)
            
            test = np.where((lower_quantile_test <= projections) & (projections <= upper_quantile_test))[0]

            if folds_with_val and val_split_kind == 'split_all':
                val_quantile_fold_range = 0.2
                lower_quantile_val_, upper_quantile_val_ = compute_val_ranges(lower_quantile_test_, val_quantile_fold_range)

                lower_quantile_val = np.quantile(projections, lower_quantile_val_)
                upper_quantile_val = np.quantile(projections, upper_quantile_val_)
                if lower_quantile_val_ > upper_quantile_val_:
                    val = np.where((projections < upper_quantile_val) | (projections >= lower_quantile_val))[0]
                    train = np.where((projections > upper_quantile_test) & (projections < lower_quantile_val))[0]
                elif upper_quantile_test_ < lower_quantile_val_:
                    val = np.where((lower_quantile_val <= projections) & (projections <= upper_quantile_val))[0]
                    train = np.where((upper_quantile_test < projections) & (projections < lower_quantile_val))[0]       
                else:
                    val = np.where((lower_quantile_val <= projections) & (projections < upper_quantile_val))[0]
                    train = np.where((projections < lower_quantile_val) | (projections > upper_quantile_test))[0]
                
                fold_pca0.append((train, val, test))
            else:
                train = np.where((projections < lower_quantile_test) | (projections > upper_quantile_test))[0]
                
                if folds_with_val and val_split_kind == 'random_from_train':
                    train_ = train.copy()
                    train_val_idxs = np.random.choice(len(train), size=len(train), replace=False)
                    train = train_[train_val_idxs[:int(len(train)*(7./9.))]] # train makes out 0.9 of all data, but needs to be 0.7
                    val = train_[train_val_idxs[int(len(train)*(7./9.)):]]
                    fold_pca0.append((train, val, test))
                else:
                    fold_pca0.append((train, test))
            
        res['pca_folds'] = fold_pca0
    
    return res

def scale_to_standard(X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(X_train)
    
    if len(y_train.shape) == 1:
        y_train = y_scaler.fit_transform(y_train.reshape(-1,1))
    else:
        y_train = y_scaler.fit_transform(y_train)
        
    if X_val is not None:
        X_val = X_scaler.transform(X_val)
    if y_val is not None:
        if len(y_test.shape) == 1:
            y_val = y_scaler.transform(y_val.reshape(-1,1))
        else:
            y_val = y_scaler.transform(y_val)
    
    X_test = X_scaler.transform(X_test)
    if len(y_test.shape) == 1:
        y_test = y_scaler.transform(y_test.reshape(-1,1))
    else:
        y_test = y_scaler.transform(y_test)
    
    return X_train, y_train, X_test, y_test, X_val, y_val, X_scaler, y_scaler


def get_dir_files(exp_dir, dataset_id):
    dir_files_ = os.listdir('%s/%s' % (exp_dir, dataset_id))
    dir_files = {'plots': {split: {} for split in available_splits},
                 'method_dict': {split: {} for split in available_splits},
                 'global_stats': {split: {} for split in available_splits},
                  'global_stats_epochs': {split: {} for split in available_splits},
                 'model': {split: {} for split in available_splits},
                  'data_dict': {split: {} for split in available_splits}}

    file_pattern = r'((\w+)_|)dataset=(\w+)_splitmode=(\w+)_foldidx=(\d+)(_dist=(\d+))*'
    file_matcher = re.compile(file_pattern)
    for dir_file in dir_files_:
        matches = file_matcher.match(dir_file)
        if matches is not None:
            matches = matches.groups()
            if matches[2] == dataset_id and matches[3] in available_splits:
                
                split = matches[3]
                fold_idx = int(matches[4])
                kind = matches[1] if matches[1] is not None else 'plots'
                
                if matches[5] is not None and matches[6] is not None:
                    dist_i = matches[6]
                    if fold_idx not in dir_files[kind][split]:
                        dir_files[kind][split][fold_idx] = {}
                    dir_files[kind][split][fold_idx][dist_i] = '%s/%s/%s' % (exp_dir, dataset_id, dir_file)
                else:
                    dir_files[kind][split][fold_idx] = '%s/%s/%s' % (exp_dir, dataset_id, dir_file)
                
                
            else:
                print("Warning. File %s has unexpected form" % dir_file)
            
    return dir_files
    

def load_global_stats(dir_files, splitmode):
    
    res = []
    global_stats = None
    for fold_idx in sorted(dir_files['global_stats'][splitmode]):
        
        file = dir_files['global_stats'][splitmode][fold_idx]
        if file.endswith('.json'):
            with open(file) as f:
                global_stats = json.load(f)
        elif file.endswith('.json.zip'):
            with gzip.open(file) as f:
                global_stats = json.load(f)
        else:
            raise Exception("File has to be .json or .json.zip, but is %s" % file)
        
        res.append(global_stats)
            
    return res

def _load_method_dict_fold(inp):
    
    file, epochs = inp
    
    if file.endswith('.json'):
        with open(file) as f:
            method_dict_json = json.load(f)
    elif file.endswith('.json.zip'):
        with gzip.open(file) as f:
            method_dict_json = json.load(f)
    else:
        raise Exception("File has to be .json or .json.zip, but is %s" % file)

    def _from_json(method_dict):
        for key in method_dict_json:

            df_train = pd.read_json(method_dict_json[key][0])
            df_test = pd.read_json(method_dict_json[key][1])
            method_dict[key] = [df_train, df_test]
    
    if epochs:
        method_dict = {}
        for epoch in method_dict_json:
            method_dict[epoch] = {}
            _from_json(method_dict[epoch])
    else:
        method_dict = {}
        _from_json(method_dict)
    
    return method_dict

def load_method_dict(dir_files, splitmode, folds=None, epochs=False):
    
    key = 'method_dict_epochs' if epochs else 'method_dict'
    if folds is None:
        folds = sorted(dir_files[key][splitmode])
    
    res = []
    for fold_idx in folds:
        
        file = dir_files[key][splitmode][fold_idx]
        method_dict = _load_method_dict_fold((file, epochs))
        res.append(method_dict)
            
    return res

def load_method_dict_multiprocessing(dir_files, splitmode):
    res = []
    method_dict_json, method_dict = None, None
    
    inps = []
    for fold_idx in sorted(dir_files['method_dict'][splitmode]):
        file = dir_files['method_dict'][splitmode][fold_idx]
        inps.append((file, self_distill))
    
    n_processes = cpu_count()
    pool = ProcessPool(processes=n_processes)
    res = pool.map(_load_method_dict_fold, inps)
    pool.close()
    pool.join()
            
    return res

def aggregate_over_folds(exp_dirs, dataset_blacklist=tuple(), exclude_ood=False):

    def _pair_to_string(a, b):
        return str(a) + " " + str(b)
    
    aggregated = pd.DataFrame(dtype=object)
    
    for exp_dir in exp_dirs:
        
        datasets = os.listdir(exp_dir)
        print(datasets)
        
        for dataset_id in datasets:
            
            if dataset_id in dataset_blacklist:
                continue
            
            if dataset_id in available_datasets:
                
                print(dataset_id)

                dir_files = get_dir_files(exp_dir, dataset_id)

                splitmode = 'random_folds'
                global_stats_folds = load_global_stats(dir_files, splitmode)

                for fold_idx, fold in enumerate(global_stats_folds):
                    for method in sorted(fold):
                        
                        trte_idents = ['train', 'val', 'test'] if len(fold[method]) == 3 else ['train', 'test']
                        for i, trte in enumerate(trte_idents):
                            dataset_trte = _pair_to_string(dataset_id, trte)

                            for metric in fold[method][i]:
                                method_metric = _pair_to_string(method, metric)
                                if dataset_trte not in aggregated.index \
                                or method_metric not in aggregated.columns \
                                or not isinstance(aggregated.loc[dataset_trte, method_metric], np.ndarray):
                                    aggregated.loc[dataset_trte, method_metric] = 0.
                                    aggregated[method_metric] = aggregated[method_metric].astype('object')
                                    aggregated.at[dataset_trte, method_metric] = np.zeros(len(global_stats_folds))

                                aggregated.loc[dataset_trte, method_metric][fold_idx] = fold[method][i][metric]

                
                if exclude_ood:
                    continue
                
                splitmode_to_ident = {'label_folds': 'label_test', 'pca_folds': 'pca_test'}
                for splitmode in sorted(splitmode_to_ident):
                    global_stats_folds = load_global_stats(dir_files, splitmode)

                    fold_mode_to_fold_idxs  = {'extrapolate': [0, len(global_stats_folds) -1], 'interpolate': np.arange(1, len(global_stats_folds)-1)}
                    for fold_mode in ['extrapolate', 'interpolate']:
                        dataset_ident = _pair_to_string(dataset_id, '%s_%s' % (splitmode_to_ident[splitmode], fold_mode))
                        
                        fold_idxs = fold_mode_to_fold_idxs[fold_mode]
                        for i, fold_idx in enumerate(fold_idxs):
                            
                            fold = global_stats_folds[fold_idx]
                            test_idx = 2 if len(fold) == 3 else 1
                            for method in sorted(fold):

                                for metric in fold[method][test_idx]:

                                    method_metric = _pair_to_string(method, metric)
                                    if dataset_ident not in aggregated.index \
                                    or method_metric not in aggregated.columns \
                                    or not isinstance(aggregated.loc[dataset_ident, method_metric], np.ndarray):
                                        aggregated.loc[dataset_ident, method_metric] = 0.
                                        aggregated[method_metric] = aggregated[method_metric].astype('object')
                                        aggregated.at[dataset_ident, method_metric] = np.zeros(len(fold_idxs))

                                    aggregated.loc[dataset_ident, method_metric][i] = fold[method][test_idx][metric]
                            
    aggregated.columns = aggregated.columns.str.split(expand=True)
    aggregated = aggregated.set_index(aggregated.index.str.split(expand=True))
    return aggregated