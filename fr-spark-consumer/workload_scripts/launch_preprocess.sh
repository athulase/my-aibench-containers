#!/bin/bash

source ~/.bashrc
LOG_DIR=$SPARK_LOG_DIR_INFERENCE
export QUEUE_SERVER=$SPARK_QUEUE_SERVER
export QUEUE_PORT=50000
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=1
mkdir -p ${LOG_DIR}

python preprocess.py > ${LOG_DIR}/preprocessing_log.txt