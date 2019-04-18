#!/bin/bash

RUN_MODE=${1:-"train"}
DATA_DIR="/home/patrik/Data/nlp/nmt"
MODEL_DIR=$(dirname "$0")/model/

mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR


if [ $RUN_MODE == "train" ]
then
    python $(dirname "$0")/nmt/train.py --model_dir $MODEL_DIR \
                                        --data_dir $DATA_DIR
elif [ $RUN_MODE == "eval" ]
then
    python $(dirname "$0")/nmt/eval.py --model_dir $MODEL_DIR \
                                       --data_dir $DATA_DIR
else
    echo "Invalid run mode command $RUN_MODE."
fi
