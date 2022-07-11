# Wasserstein Dropout for Object Detection

#### Conda Virtual Environment

We used a conda environment on Linux Debian Version 9. Use the provided `environment_object_detection.yml` file to create this environment as follows:

`conda env create --name tf-py2-gpu --file=environment_object_detection.yml`.

Our code is based on [Bichen Wu's implementation](https://github.com/BichenWuUCB/squeezeDet) of the object detection network ["SqueezeDet"](https://arxiv.org/abs/1612.01051).

#### Datasets

In order to rerun our training and evaluation, add the following datasets to the `data` folder that is located in the same directory  (the root folder) as the conda environment yml file.

##### For training and evaluation:

[*KITTI*](http://www.cvlibs.net/datasets/kitti/eval_object.php)

- Download and extract ''left color'' images and their object-level annotations ("annotations of object dataset").
- Put the image and annotation data in the folder named `data/KITTI`. The folder should have the following structure:

```
data
├── KITTI  
│   ├── training
│   │   ├── image_2 // .png images
│   │   │   ├── 000000.png
│   │   │   └── ...
│   │   └── label_2 // .txt annotations
│   │       ├── 000000.txt
│   │       └── ...
│   └── ImageSets   
│       ├── train.txt
│       ├── val.txt
│       └── trainval.txt
│       
└── ... // Further datasets / pre-trained models
```

- The `data/KITTI/ImageSets` folder contains the files `train.txt` and `val.txt`, which specify the split used for training and evaluation. For further instructions, visit the ["SqueezeDet"](https://github.com/BichenWuUCB/squeezeDet) repository.

##### For evaluations under data shift:

The network is trained on KITTI, all other datasets (BDD100k, Nightowls, SynScapes, A2D2, NuImages) are solely used for evaluations under data shift.



[*BDD100k*](https://bdd-data.berkeley.edu/)

- Extract the images and annotations into the folder `data/BDD`. The structure should be as follows:

```
data
├── BDD  
│   ├── images 
│   │   └── 100k 
│   │       ├── train // .png images
│   │       ├── val
│   │       └── test
│   └─ labels 
│        └── detection20
│             └── det_v2_val_release.json
│
└── ... // Further datasets / pre-trained models
```
- Note that the `det_v2_val_release.json` corresponds to the `labels/det_20/det_val.json` in the "Detection 2020 Labels" of the BDD dataset.



[*Nightowls*](https://www.nightowls-dataset.org/)

- Extract the "validation images" and "validation annotations" into the folder `data/Nightowls`. The structure should be as follows:
```
  data
  ├── Nightowls  
  │   ├── nightowls_validation 
  │   │   └── ... // .png images
  │   └─ nightowls_validation.json // annotations  
  │
  └── ... // Further datasets / pre-trained models
```



[*Synscapes*](https://7dlabs.com/synscapes-overview)

  - Extract the images and annotations into the folder `data/Synscapes`. The structure should be as follows:
```
data
├── Synscapes  
│   ├── img 
│   │    └── rgb
│   │         └── ... // .png images
│   ├── meta
│   │    └── ... // .json annotations 
│   │
│   ├── train_idxs.txt // train split
│   └── val_idxs.txt // val split
│
└── ... // Further datasets / pre-trained models
```



[*A2D2*](https://www.a2d2.audi/a2d2/en/dataset.html)

- Extract the image sequences captured by the center front-facing camera ("cam_front_center") together with their corresponding annotations "label2D" into the folder `data/A2D2` as follows:

```
data
├── A2D2  
│   ├── 20180807_145028 // "sequence" folder
│   │   ├── camera 
│   │   │   └── cam_front_center // .png images 
│   │   │       ├── 20180807145028_camera_frontcenter_000000091.png
│   │   │       └── ...
│   │   └── label2D 
│   │       └── cam_front_center // .txt annotations 
│   │             ├── 20180807145028_label2D_frontcenter_000000091.json
│   │             └── ...
│   │
│   └── ... // Further sequences
└── ... // Further datasets / pre-trained models
```
- The following train/val/test split of sequences has been used (hardcoded into our provided data loader):
```
"train": ["20180807_145028", "20180925_135056", "20181108_091945", 
          "20181204_154421", "20181204_135952", "20181108_084007",
          "20181107_132730", "20181108_103155", "20181204_170238",
          "20181016_082154", "20181107_133258", "20181108_123750",
          "20180925_112730", "20181016_095036", "20181107_133445",
          "20181108_141609", "20180925_124435", "20181016_125231"],
"val":  ["20181008_095521", "20181107_132300", "20180925_101535", "20181204_191844"],
"test": ["20180810_142822"],
```



[*NuImages*](https://www.nuscenes.org/nuimages)

- Extract the images and annotations into the folder `data/Synscapes`. The structure should be as follows:

````
data
├── NuImages  
│   ├── samples
│   │    ├── CAM_BACK
│   │    │     └── ... // .jpg images
│   │    ├── CAM_FRONT
│   │    │     └── ... // .jpg images
│   │    └── ... 
│   ├── v1.0-train // annotations for train split
│   │    └── ... // .json annotations 
│   └── v1.0-val // annotations for val split
│        └── ...
└── ... // Further datasets / pre-trained models
```
````

##### For evaluations on corrupted datasets:

For the evaluation on corrupted input data, please duplicate the dataset folders created above (e.g., `data/KITTI`) and apply a "corruption" (e.g., Gaussian noise) to the corresponding images using a standard image manipulation library. 

Considering, for instance, the Synscapes dataset, create a folder `data/Synscapes_corrupted` and copy the contents of the `data/Synscapes` folder into it. Replace the images stored in `data/Synscapes/img/rgb` with their corrupted versions, without changing their file names. 

## Quick Start

#### Model Training (on KITTI)

Make sure to have the conda environment activated. To start a model training on the KITTI dataset, run the following command in the root folder:

`sh scripts/train.sh -dataset kitti_3cls -net squeezeDet -gpu {GPU} -train_dir {TRAIN_DIR} -data_path ./data/KITTI -uncertainty_method {UNC_METHOD}`

where 

- {GPU} is the number of the selected GPU (default is `0`),
- {TRAIN_DIR} a directory with a `train` subfolder in which model checkpoints will be stored,
- {UNC_METHOD} is either "mc" for standard training with activated dropout (MC dropout) or "exact_wdrop" (Wasserstein dropout).

#### Model Evaluation (on Various Datasets)

Make sure to have the conda environment activated. First, the trained model is applied to a given dataset. For this, run the following command in the root folder:

`sh scripts/eval.sh -dataset {DATASET} -net squeezeDet -gpu {GPU} -eval_dir {EVAL_DIR} -data_path {DATA_DIR} -uncertainty_method mc`

where 

- {DATASET} is the dataset identifier, which must be one of the following: `kitti_3cls`, `bdd_3cls`, `nightowls_3cls`, `synscapes_3cls`, `a2d2_3cls`, `nuimages_3cls`,
- {GPU} is the number of the selected GPU (default is `0`),
- {EVAL_DIR} a directory with a sub-directory `train` that contains the model checkpoints generated during training ({EVAL_DIR} should be equal to the {TRAIN_DIR} from model training),
- {DATA_DIR} is the directory of the images and annotations (e.g., `./data/KITTI` for the KITTI dataset or `./data/Synscapes` for the Synscapes dataset).
  To evaluate the model on corrupted data, {DATA_DIR} should point to a dataset folder which contains corrupted images, e.g., `./data/Synscapes_corrupted`.

The script stores inference data in the folder `{EVAL_DIR}/val/detection_files_{STEP}`, where {STEP} is the number of batches processed during training. 

In a second step, another python script computes performance and uncertainty metrics (e.g., RMSE, NLL, ECE) based on the generated inference data. To do this, please run the following command in the root folder: `python src/uncertainty_eval.py --eval_dir {EVAL_DIR} --uncertainty_method mc`.

Note: The script makes heavy use of the CPU via multiprocessing. It is advised to close other programs before launch. The script prints the main metrics to the console. Additionally, they are stored as a python dictionary in a pickle file `aggregated_metrics.pkl` in the directory `{EVAL_DIR}/val/detection_files_{STEP}`.

#### Used Hardware and Execution Time

All experiments are conducted using a `Intel(R) Xeon(R) Gold 6126 CPU @ 2.60GHz` and a `Tesla V100 GPU`.

Running the described experiments with cross validation takes approximately 18h (on GPU).