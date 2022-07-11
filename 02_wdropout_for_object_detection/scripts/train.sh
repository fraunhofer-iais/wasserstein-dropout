#!/bin/bash

export GPUID=0
export NET="squeezeDet"
export DATASET="KITTI"
export TRAIN_DIR="/tmp/bichen/logs/SqueezeDet/"
export SUMMARY_STEPS=100
export CHECKPOINT_STEPS=500
export N_SAMPLES="None"
export MAX_STEPS=150000
export MC_KEEP_PROBS="[1.0,0.5]"
export DATA_PATH="./data/KITTI"
export UNCERTAINTY_METHOD="mc"
export PRED_FILTERING_METHOD="nms_orig"

if [ $# -eq 0 ]
then
  echo "Usage: ./scripts/train.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help"
  echo "-dataset                  (kitti|bdd|nightowls|synscapes|a2d2|nuimages|bdd_3cls|nightowls_3cls|synscapes_3cls|a2d2_3cls|nuimages_3cls)"
  echo "-net                      (squeezeDet)"
  echo "-gpu                      gpu id"
  echo "-train_dir                directory for training logs"
  echo "-n_samples                number of samples"
  echo "-mc_keep_probs            [1.0,0.5]"
  echo "-data_path                './data/KITTI'"
  echo "-uncertainty_method       mc or exact_wdrop"
  echo "-pred_filtering_method    one out of nms_orig or nms"
  echo "-max_steps"
  echo "-summary_steps             100"
  echo "-checkpoint_steps          500"
  exit 0
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ./scripts/train.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-dataset                  (kitti|bdd|nightowls|synscapes|a2d2|nuimages|bdd_3cls|nightowls_3cls|synscapes_3cls|a2d2_3cls|nuimages_3cls)"
      echo "-net                      (squeezeDet)"
      echo "-gpu                      gpu id"
      echo "-train_dir                directory for training logs"
      echo "-n_samples                number of samples"
      echo "-mc_keep_probs            [1.0,0.5]"
      echo "-data_path                './data/KITTI'"
      echo "-uncertainty_method       mc or exact_wdrop"
      echo "-pred_filtering_method    one out of nms_orig, nms"
      echo "-max_steps"
      echo "-summary_steps             100"
      echo "-checkpoint_steps          500"
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
    -train_dir)
      export TRAIN_DIR="$2"
      shift
      shift
      ;;
    -checkpoint_steps)
      export CHECKPOINT_STEPS="$2"
      shift
      shift
      ;;
    -summary_steps)
      export SUMMARY_STEPS="$2"
      shift
      shift
      ;;
    -n_samples)
      export N_SAMPLES="$2"
      shift
      shift
      ;;
    -max_steps)
      export MAX_STEPS="$2"
      shift
      shift
      ;;
    -mc_keep_probs)
      export MC_KEEP_PROBS="$2"
      shift
      shift
      ;;
    -data_path)
      export DATA_PATH="$2"
      shift
      shift
      ;;
    -uncertainty_method)
      export UNCERTAINTY_METHOD="$2"
      shift
      shift
      ;;
    -pred_filtering_method)
      export  PRED_FILTERING_METHOD="$2"
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done

case "$NET" in 
  "squeezeDet")
    export PRETRAINED_MODEL_PATH="./data/SqueezeNet/squeezenet_v1.1.pkl"
    ;;
  *)
    echo "net architecture not supported."
    exit 0
    ;;
esac


python ./src/train.py \
  --dataset=$DATASET \
  --data_path=$DATA_PATH \
  --pretrained_model_path=$PRETRAINED_MODEL_PATH \
  --image_set=train \
  --train_dir="$TRAIN_DIR/train" \
  --net=$NET \
  --summary_step=$SUMMARY_STEPS \
  --checkpoint_step=$CHECKPOINT_STEPS \
  --gpu=$GPUID \
  --n_samples=$N_SAMPLES \
  --mc_keep_probs=$MC_KEEP_PROBS \
  --max_steps=$MAX_STEPS \
  --uncertainty_method=$UNCERTAINTY_METHOD \
  --pred_filtering_method=$PRED_FILTERING_METHOD