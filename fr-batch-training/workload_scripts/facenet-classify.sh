##################################################################################
#/*
#* "INTEL CONFIDENTIAL" Copyright 2017 Intel Corporation All Rights
#* Reserved.
#*
#* The source code contained or described herein and all documents related
#* to the source code ("Material") are owned by Intel Corporation or its
#* suppliers or licensors. Title to the Material remains with Intel
#* Corporation or its suppliers and licensors. The Material contains trade
#* secrets and proprietary and confidential information of Intel or its
#* suppliers and licensors. The Material is protected by worldwide copyright
#* and trade secret laws and treaty provisions. No part of the Material may
#* be used, copied, reproduced, modified, published, uploaded, posted,
#* transmitted, distributed, or disclosed in any way without Intel's prior
#* express written permission.
#*
#* No license under any patent, copyright, trade secret or other
#* intellectual property right is granted to or conferred upon you by
#* disclosure or delivery of the Materials, either expressly, by
#* implication, inducement, estoppel or otherwise. Any license under such
#* intellectual property rights must be express and approved by Intel in
#* writing.
#*/
####################################################################################
#!/bin/bash

source ~/.bashrc


FACENET_HOME=$FACENET_HOME_DIR
LOG_DIR=$FACENET_LOG_DIR
DATA_DIR=$FACENET_HOME_DIR/datasets/lfw
MODEL_DIR=$FACENET_HOME_DIR/models
ALLIGNED_DATA_DIR=$FACENET_HOME_DIR/datasets/lfw/lfw_mtcnnpy_160
batch_size=$FACENET_BATCH_SIZE
min_nrof_images_per_class=$FACENET_MIN_NR_OF_IMAGES_PER_CLASS
nrof_train_images_per_class=$FACENET_NR_OF_TRAIN_IMAGES_PER_CLASS
PKL_FILE_PATH=$MODEL_DIR/20170512-110547/lfw_classifier.pkl

mkdir -p $LOG_DIR/facenet-logs
chmod 777 -R $LOG_DIR/facenet-logs

CLASSIFY_IMAGE_DIR=$ALLIGNED_DATA_DIR

python $FACENET_HOME/src/classifier.py CLASSIFY ${CLASSIFY_IMAGE_DIR} $MODEL_DIR/20170512-110547/20170512-110547.pb ${PKL_FILE_PATH} --batch_size ${batch_size} --min_nrof_images_per_class ${min_nrof_images_per_class} --nrof_train_images_per_class ${nrof_train_images_per_class} --use_split_dataset > "${LOG_DIR}"/facenet-logs/facenet_classify_$HOSTNAME.log 2>"${LOG_DIR}"/facenet-logs/facenet_classify_$HOSTNAME.err
ret=$?
echo "return code:$ret"
if [ $ret -ne 0 ]; then
     exit 0
fi
