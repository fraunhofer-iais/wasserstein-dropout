# Wasserstein Dropout on Standard 1D Regression Datasets

Wasserstein dropout (W-dropout) is a novel training objective for dropout-based regression networks that yields improved uncertainty estimates. The code accompanies our paper ["Wasserstein dropout"](https://arxiv.org/abs/2012.12687) and allows for reproducing the results therein.

Please note that ["Wasserstein dropout"](https://arxiv.org/abs/2012.12687) is a successor of the ["second-moment loss"](https://arxiv.org/abs/2101.02726), a previously published uncertainty estimation technique. This code repository partially overlaps with [the repository accompanying the "second-moment loss" paper](https://github.com/fraunhofer-iais/second-moment-loss), particularly w.r.t. the implementations of (some) benchmark methods and the evaluation parts.

## Overview

* Wasserstein dropout networks capture [heteroscedastic](https://en.wikipedia.org/wiki/Heteroscedasticity), i.e., input-dependent, data noise by means of their sub-network distributions.
* Technically, this is achieved by matching dropout-induced output distributions to the (factual or implicit) data distributions via minimization of the [Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric).
* Wasserstein dropout networks outperform state-of-the-art uncertainty techniques w.r.t. various benchmark metrics, not only in-data but also under data shifts.
* Wasserstein dropout can serve as a drop-in replacement for MC dropout on regression tasks without increasing computational requirements (at inference).

#### Baselines 

Wasserstein dropout models are compared to various other approaches for uncertainty estimation:

* parametric uncertainty (i.e., NLL-optimized networks that explicitly parameterize output variance; using a [Gaussian likelihood](https://ieeexplore.ieee.org/document/374138) and a [Student-t likelihood](https://arxiv.org/abs/1910.02600), respectively),
* MC dropout (["standard"](https://arxiv.org/pdf/1506.02142.pdf), [Concrete](https://arxiv.org/pdf/1705.07832.pdf) variant and [combined with parametric uncertainty](https://arxiv.org/pdf/1703.04977.pdf)),
* [deep ensembles](https://arxiv.org/abs/1612.01474) (variant with/without parametric uncertainty) and
* ["SWAG"](https://arxiv.org/pdf/1902.02476.pdf).


#### Conda Virtual Environment

We used a conda environment on Linux Debian Version 9. Use the provided `environment.yml` file to create this environment as follows:

`conda env create --name torch_env --file=environment.yml`.

In addition, we make use of [the ConcreteDropout implementation by Daniel Kelshaw](https://github.com/danielkelshaw/ConcreteDropout) (see respective classes in `models.py`).


#### Datasets

In order to rerun our training and evaluation, you need to add the following standard 1D datasets to the `data` folder that is located in the same directory as the Jupyter notebooks.

- ["concrete"](http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength):
  Put `Concrete_Data.xls` into the data folder.

- ["energy"](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency):
  Put `ENB2012_data.xlsx` into the data folder.

- ["abalone"](https://archive.ics.uci.edu/ml/datasets/abalone):
  Put `abalone.data` and `abalone.names` into the data folder.

- ["naval"](https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants):
  Extract the `UCI CBM Dataset.zip` file into the data folder such that the data folder has a subdirectory `UCI CBM Dataset`.

- ["power"](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant):
  Extract the `CCPP.zip` file into the data folder such that the data folder has a subdirectory CCPP.

- ["protein"](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure):
  Put `CASP.csv` into the data folder.

- ["wine quality"](https://archive.ics.uci.edu/ml/datasets/wine+quality):
  Put `winequality-red.csv` into the data folder.

- ["yacht"](http://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics):
  Put `yacht_hydrodynamics.data` into the data folder.

- ["year"](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD):
  Extract `YearPredictionMSD.txt.zip` into the data folder such that the data folder has the file `YearPredictionMSD.txt`.

- ["superconduct"](https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data):
  Extract `superconduct.zip` into the data folder such that the data folder has a subdirectory `superconduct`.

- ["kin8nm"](https://www.openml.org/d/189):
  Download the "dataset_2175_kin8nm.arff" file and put it into the data folder.

- "boston", "diabetes", "california":
  These are obtained using functions from sklearn, namely `load_boston`, `load_diabetes` and `fetch_california_housing` from the `sklearn.datasets` package.

The toy datasets are generated using functions in the `data.py` file. 
Note that the "toy_noise" dataset in the paper is referred to as "toy_modulated" in the code.

## Quick Start

#### Model Training

The code for the experiments on toy and standard 1D datasets is provided in the jupyter notebook `train_and_eval.ipynb`. Make sure to have the conda environment activated. Then execute all cells except for the last one.

In the last cell:
Insert an identifier string in line 5 (`exp_ident = ...`). 
This identifier is a part of the path of the directory 
`exp_dir = './experiment_results/*exp_ident*_*CURRENT_TIME*'`
in which experiment log files are stored.
If the directory does not exist it will be created.

To run the training/evaluation for the hyperparameter study (L and p parameter), replace the line
`methods = ['swag', 'de', 'pu', 'evidential', 'pu_de', 'mc_pu', 'mc_wd=0.000001', 'concrete_dropout', 'new_wdrop_exact_l=5']`
with
`methods = ['new_wdrop_exact_l=4', 'new_wdrop_exact_l=5', 'new_wdrop_exact_l=8', 'new_wdrop_exact_l=10', 'new_wdrop_exact_l=20', 'new_wdrop_exact_l=5_p=0.05', 'new_wdrop_exact_l=5_p=0.2', 'new_wdrop_exact_l=5_p=0.3', 'new_wdrop_exact_l=5_p=0.4', 'new_wdrop_exact_l=5_p=0.5']`
This will start the training with Wasserstein dropout loss only and different values for L and p.

To also include runs on the toy datasets "toy_hf" and "toy_modulated", add the strings 'toy_hf' and 'toy_modulated' to the "datasets" list in line 31 (of the last cell).

Run the last cell to start the experiment. After the training finishes, `exp_dir` contains several subdirectories for each dataset.
Those subdirectories contain zipped dictionaries that contain the performance/uncertainty metrics on train/test data.

#### Model Evaluation

To visualize the evaluation, another notebook is provided: `plot_experiment_results.ipynb`.

- Before creating any plots, first run cells 1 and 2.

- To create the plots in Figure 13, set the paths in the `exp_dir`'s list of cell 3. This should point to the `exp_dir` used when running the training/evaluation script.
- Finally, run cell 3. The plots will be stored to the `./plots` directory.

- To create the plots in Figure 4:
	- In cell 4, change the `exp_dir` path accordingly. It should point to `exp_dir` that contains a subdirectory with the "toy_modulated" dataset.
	- Run cell 4 to create the plots for "toy_modulated". It will be stored as `./plots/wdrop_toy_modulated.png`.
	- In cell 5, change the `exp_dir` path accordingly. It should point to `exp_dir` that contains a subdirectory with the "toy_hf" dataset.
	- Run cell 5 to create the plot for "toy_hf". It will be stored as `./plots/wdrop_toy_hf.png`.

- To create the plots in Figure 8:
   - In cell 6, change the `exp_dir`'s list accordingly. It should contain paths to `exp_dir`'s that have subdirectories for the "concrete" and the "diabetes" dataset.
   - Run cell 6 afterwards to create the out-of data behavior plot. It will be stored as `./plots/wdrop_ood_behavior.png`.

- To create the plots in Figures 6 and 7:
   - In cell 8, change the `exp_dir`'s list accordingly. It should contain paths to `exp_dir`'s that have subdirectories for all standard 1D datasets.
   - Run cell 8 afterwards.
   - Run cell 9 to create a plot showing RMSE performance on all datasets.
   - Run cell 10 to create a plot showing NLL performance on all datasets. 
   - Run cell 11 to create a plot showing ECE uncertainty on all datasets.
   - Run cell 12 to create a plot showing Wasserstein distance on all datasets.
   - All plots will be stored in the `./plots` directory.

- To create the plots in Figures 17 and 18:
   - In cell 8, change the `exp_dir`'s list accordingly. It should contain paths to `exp_dir`'s that have subdirectories for all standard 1D datasets for the hyperparameter evaluation.
   - Run cell 8 afterwards.
   - Run cell 13 to create a plot showing RMSE performance on all datasets for Wasserstein dropout with different values of L.
   - Run cell 14 to create a plot showing ECE uncertainty on all datasets for Wasserstein dropout with different values of L.
   - Run cell 15 to create a plot showing RMSE performance on all datasets for Wasserstein dropout with different values of p.
   - Run cell 16 to create a plot showing ECE uncertainty on all datasets for Wasserstein dropout with different values of p.
   - All plots will be stored in the `./plots` directory.


- To create the plot in Figure 19:
   - Change the `exp_dir`'s list in cell 17 such that it points to `exp_dir`'s of a standard training/evaluation run on 1D datasets.
   - The plot will be stored in the `./plots` directory.

#### Used Hardware and Execution Time

All experiments are conducted on a `Intel(R) Xeon(R) Gold 6126 CPU @ 2.60GHz`. 

Running the described experiments with cross validation takes 10h for toy data and 122h for the standard 1D regression tasks.


