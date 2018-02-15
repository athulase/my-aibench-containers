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
FLOW=$1
WORKLOAD_LOG=$2
chmod 777 $WORKLOAD_LOG
LOG_DIR="${LOG_DIR:-/mnt/nvme_shared_x01/facenet_batch_inference/logs}"
DATASET_SIZE=`grep "Number of images:" $WORKLOAD_LOG | head -n 1 | cut -d':' -f2 | xargs`
echo "Field,Value"> ${LOG_DIR}/flow_definition.csv
echo "FLOW_TYPE,$FLOW">> ${LOG_DIR}/flow_definition.csv
echo "DOMAIN,image">> ${LOG_DIR}/flow_definition.csv
echo "USECASE,face recognition">> ${LOG_DIR}/flow_definition.csv
echo "CONFIGURATION,facenet">> ${LOG_DIR}/flow_definition.csv
echo "DL_MODEL_TYPE,20170512-110547">> ${LOG_DIR}/flow_definition.csv
echo "FRAMEWORK,tensorflow">> ${LOG_DIR}/flow_definition.csv
echo "DATASET,lfw">> ${LOG_DIR}/flow_definition.csv
echo "DATASET_SIZE,$DATASET_SIZE">> ${LOG_DIR}/flow_definition.csv
