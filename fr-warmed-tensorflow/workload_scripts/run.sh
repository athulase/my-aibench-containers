#!/usr/bin/env bash

source ~/.bashrc
CLASSIFIER_FILENAME=$WARMTENSOR_CLASSIFIER
MODEL=$WARMTENSOR_MODEL
LOG_DIR=$WARMTENSOR_LOGDIR
NUM_PROCESSES=$WARMTENSOR_PROCESSES
mkdir -p $LOG_DIR


NO_OF_PROCESSORS=`cat /proc/cpuinfo | grep processor | wc -l`
LIST_OF_PROCESSORS=[`seq -s "," 0 $((NO_OF_PROCESSORS - 1))`]


if [ -z "$WARMTENSOR_INTRA_OP" ]; then export WARMTENSOR_INTRA_OP=$NO_OF_PROCESSORS;  export WARMTENSOR_INTER_OP=1; fi
export KMP_BLOCKTIME=1
#export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export KMP_SETTINGS=1
export OMP_NUM_THREADS=$NO_OF_PROCESSORS
python separate_classify.py 2>&1 | tee ${LOG_DIR}/Classifier.log
cat output_inference_* > /output_inference.txt
cp /output_inference.txt ${LOG_DIR}/output_inference.txt
cp /relevant_metrics ${LOG_DIR}/relevant_metrics
echo "After Classify"



