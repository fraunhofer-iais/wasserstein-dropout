#!/bin/bash

export GPUID=0
export DATASET="KITTI"
export NET="squeezeDet"
export EVAL_DIR="/tmp/bichen/logs/SqueezeDet/"
export IMAGE_SET="val"
export EVAL_ONCE="False"
export MC_KEEP_PROBS="[1.0,0.5]"
export UNCERTAINTY_METHOD="mc"
export DATA_PATH="./data/KITTI"
export PRED_FILTERING_METHOD="nms_orig"
export GLOBAL_STEP="None"
export SUBSAMPLE_FRAC="-1"

if [ $# -eq 0 ]
then
  echo "Usage: ./scripts/train.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help"
  echo "-net                      (squeezeDet)"
  echo "-dataset                  (kitti|bdd|nightowls|synscapes|a2d2|nuimages|bdd_3cls|nightowls_3cls|synscapes_3cls|a2d2_3cls|nuimages_3cls)"
  echo "-gpu                      gpu id"
  echo "-eval_dir                 directory to save logs"
  echo "-image_set                (train|val)"
  echo "-eval_once                (True|False)"
  echo "-mc_keep_probs            [1.0,0.5]"
  echo "-uncertainty_method       mc"
  echo "-data_path                ./data/KITTI"
  echo "-pred_filtering_method     nms_orig"
  echo "-global_step               None"
  echo "-sample_frac            -1"
  echo "-checkpoint_path         defaults to eval_dir/train"
  exit 0
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ./scripts/train.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-net                      (squeezeDet)"
      echo "-dataset                  (kitti|bdd|nightowls|synscapes|a2d2|nuimages|bdd_3cls|nightowls_3cls|synscapes_3cls|a2d2_3cls|nuimages_3cls)"
      echo "-gpu                      gpu id"
      echo "-eval_dir                 directory to save logs"
      echo "-image_set                (train|val)"
      echo "-eval_once                (True|False)"
      echo "-mc_keep_probs            [1.0,0.5]"
      echo "-uncertainty_method       mc"
      echo "-data_path                ./data/KITTI"
      echo "-pred_filtering_method     nms_orig"
      echo "-global_step               None"
      echo "-sample_frac            -1"
      echo "-checkpoint_path         defaults to eval_dir/train"
      exit 0
      ;;
    -net)
      export NET="$2"
      shift
      shift
      ;;
    -dataset)
      export DATASET="$2"
      shift
      shift
      ;;
    -gpu)
      export GPUID="$2"
      shift
      shift
      ;;
    -eval_dir)
      export EVAL_DIR="$2"
      shift
      shift
      ;;
    -image_set)
      export IMAGE_SET="$2"
      shift
      shift
      ;;
     -eval_once)
      export EVAL_ONCE="$2"
      shift
      shift
      ;;
     -mc_keep_probs)
       export MC_KEEP_PROBS="$2"
       shift
       shift
       ;;
     -uncertainty_method)
       export UNCERTAINTY_METHOD="$2"
       shift
       shift
       ;;
      -data_path)
       export DATA_PATH="$2"
       shift
       shift
       ;;
      -pred_filtering_method)
        export PRED_FILTERING_METHOD="$2"
        shift
        shift
        ;;
      -global_step)
        export GLOBAL_STEP="$2"
        shift
        shift
        ;;
      -sample_frac)
        export SUBSAMPLE_FRAC="$2"
        shift
        shift
        ;;
      -checkpoint_path)
        export CHECKPOINT_PATH="$2"
        shift
        shift
        ;;
    *)
      break
      ;;
  esac
done

# =========================================================================== #
# command for squeezeDet:
# =========================================================================== #
python ./src/eval.py \
  --dataset=$DATASET \
  --data_path=$DATA_PATH \
  --image_set=$IMAGE_SET \
  --eval_dir="$EVAL_DIR/$IMAGE_SET" \
  --checkpoint_path="${CHECKPOINT_PATH:=$EVAL_DIR}/train" \
  --net=$NET \
  --gpu=$GPUID \
  --run_once=$EVAL_ONCE \
  --mc_keep_probs=$MC_KEEP_PROBS \
  --uncertainty_method=$UNCERTAINTY_METHOD \
  --pred_filtering_method=$PRED_FILTERING_METHOD \
  --global_step=$GLOBAL_STEP \
  --sample_frac=$SUBSAMPLE_FRAC
