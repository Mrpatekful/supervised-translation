#!/bin/bash

MODEL=${1:-seq2seq}
DATA_DIR=${2:-"/home/patrik/Data/nlp/nmt"}

MODEL_DIR=$(dirname "$0")/model/$MODEL
MODEL_FILE=${3:-$MODEL_DIR/"model"}

mkdir -p $(dirname "$0")/../checkpoints
mkdir -p $MODEL_DIR

python $(dirname "$0")/nmt/train.py --model $MODEL \
                                    --data_dir $DATA_DIR
