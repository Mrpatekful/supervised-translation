#!/bin/bash

DATA_DIR=${1:-"/home/patrik/Data/nlp/nmt"}

MODEL_DIR=$(dirname "$0")/model/$MODEL
MODEL_FILE=${3:-$MODEL_DIR/"model"}

mkdir -p $(dirname "$0")/checkpoints
mkdir -p $MODEL_DIR

python $(dirname "$0")/nmt/train.py --model_dir $MODEL_DIR \
                                    --data_dir $DATA_DIR
