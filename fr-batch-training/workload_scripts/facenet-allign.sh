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

NO_OF_PROCESSORS=`cat /proc/cpuinfo | grep processor | wc -l`
LIST_OF_PROCESSORS=[`seq -s "," 0 $((NO_OF_PROCESSORS - 1))`]

echo "NO_of_processors is: $NO_OF_PROCESSORS"
echo "LIST_OF_PROCESSORS is: $LIST_OF_PROCESSORS"

KMP_AFFINITY=granularity=fine,explicit,proclist=$LIST_OF_PROCESSORS
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=$NO_OF_PROCESSORS

FACENET_HOME=$FACENET_HOME_DIR
DATA_DIR=$FACENET_HOME_DIR/datasets/lfw
ALLIGNED_DATA_DIR=$FACENET_HOME_DIR/datasets/lfw/lfw_mtcnnpy_160
RAW_DIR=$FACENET_HOME_DIR/datasets/lfw/raw

python $FACENET_HOME/src/align/align_dataset_mtcnn.py ${RAW_DIR} ${ALLIGNED_DATA_DIR} --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25
