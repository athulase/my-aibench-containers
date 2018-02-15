#!/bin/bash

source ~/.bashrc
LOG_DIR=$CNN_LOG_DIR
ITER=$CNN_ITERATIONS
HOME_DIR=/tf_cnn_scripts
DATA_DIR=$CNN_DATA_DIR
DATA_NAME='flowers'
DATA_FORMAT=$CNN_DATA_FORMAT
MODEL=$CNN_MODEL
BATCH_SIZE=$CNN_BATCH_SIZE
CPU=$CNN_CPU_FAMILY
CORES=$CNN_CORES
NUM_INTRA=$CNN_INTRA
NUM_INTER=$CNN_INTER
SYNTHETIC=$CNN_SYNTHETIC

TRAINED_DIR="$CNN_TRAIN_DIR/$MODEL/"
mkdir -p $TRAINED_DIR
rm -rf $TRAINED_DIR/*
mkdir -p $LOG_DIR/tf-batch-training-${MODEL}/
cd $HOME_DIR

if [[ $CPU == knm || $CPU == knl ]]; then
    export MKL_ENABLE_INSTRUCTIONS=AVX512_MIC_E1
    if [[ $SYNTHETIC == "synthetic" ]]; then
        numactl -m 1 python run_single_node_benchmark.py --model ${MODEL} --batch_size ${BATCH_SIZE} --data_format $DATA_FORMAT --num_intra_threads $NUM_INTRA --num_inter_threads $NUM_INTER > ${LOG_DIR}/tf-batch-training-${MODEL}/$HOSTNAME.log 2>&1
    else
        numactl -m 1 python run_single_node_benchmark.py --model ${MODEL} --batch_size ${BATCH_SIZE} --data_format $DATA_FORMAT --data_dir ${DATA_DIR} --data_name $DATA_NAME --cpu ${CPU} --train_dir ${TRAINED_DIR} --num_intra_threads $NUM_INTRA --num_inter_threads $NUM_INTER > ${LOG_DIR}/tf-batch-training-${MODEL}/$HOSTNAME.log 2>&1
    fi
else
    if [[ $SYNTHETIC == "synthetic" ]]; then
        python run_single_node_benchmark.py --model ${MODEL} --batch_size ${BATCH_SIZE} --data_format $DATA_FORMAT --num_intra_threads $NUM_INTRA --num_inter_threads $NUM_INTER > ${LOG_DIR}/tf-batch-training-${MODEL}/$HOSTNAME.log 2>&1
    else
        python run_single_node_benchmark.py --model ${MODEL} --batch_size ${BATCH_SIZE} --data_format $DATA_FORMAT  --data_dir ${DATA_DIR} --data_name $DATA_NAME --cpu ${CPU} --train_dir ${TRAINED_DIR} --num_intra_threads $NUM_INTRA --num_inter_threads $NUM_INTER > ${LOG_DIR}/tf-batch-training-${MODEL}/$HOSTNAME.log 2>&1
    fi
fi
