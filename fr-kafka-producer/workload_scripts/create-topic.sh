###################################################################################
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
###################################################################################

#!/bin/bash
source ~/.bashrc
KAFKA_PRODUCER=fr-kafka-producer-0
DATA_DIR=$KAFKA_DATA_DIR
TOPIC_NAME=$KAFKA_TOPIC_NAME
LOG_DIR=$KAFKA_LOG_DIR
NUM_SLAVES=$KAFKA_NUM_SLAVES
LAST_TOPIC=$((KAFKA_TOPIC_NUMBER-1))

mkdir -p ${LOG_DIR}/kafka-logs/
chmod 777 -R ${LOG_DIR}/kafka-logs/

for i in `seq 0 9`;
do
    ${KAFKA_HOME}/bin/kafka-topics.sh --create --zookeeper ${KAFKA_IP}:2080 --replication-factor 1 --partitions $NUM_SLAVES --topic $TOPIC_NAME$i > ${LOG_DIR}/kafka-logs/kafka_create_topic_$HOSTNAME.log
done