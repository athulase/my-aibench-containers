#!/bin/bash

sync

source ~/.bashrc
FLOW="Batch_Inference"
LOG_DIR=$FACENET_LOG_DIR
WORKLOAD_LOG="${LOG_DIR}/facenet-logs/facenet_classify_*.log"
WORKLOAD_DIR="${LOG_DIR}/facenet-logs/"
CSV_DIR=${LOG_DIR}/${NAMESPACE}

mkdir -p ${WORKLOAD_DIR}
chmod -R 777 ${WORKLOAD_DIR}
mkdir -p ${CSV_DIR}

IMAGE_COUNT=`grep "Number of images:" $WORKLOAD_LOG | head -n 1 | cut -d':' -f3 | xargs`
echo "Field,Value"> ${CSV_DIR}/flow_definition.csv
echo "FLOW_TYPE,$FLOW">> ${CSV_DIR}/flow_definition.csv
echo "DOMAIN,image">> ${CSV_DIR}/flow_definition.csv
echo "USECASE,face recognition">> ${CSV_DIR}/flow_definition.csv
echo "CONFIGURATION,facenet">> ${CSV_DIR}/flow_definition.csv
echo "DL_MODEL_TYPE,20170512-110547">> ${CSV_DIR}/flow_definition.csv
echo "FRAMEWORK,tensorflow">> ${CSV_DIR}/flow_definition.csv
echo "DATASET,lfw">> ${CSV_DIR}/flow_definition.csv
echo "IMAGE_COUNT,$IMAGE_COUNT">> ${CSV_DIR}/flow_definition.csv

sync