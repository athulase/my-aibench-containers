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

LOG_DIR=$KAFKA_LOG_DIR

mkdir -p /tmp/kafka-logs
chmod -R 777 /tmp/kafka-logs

mkdir -p ${LOG_DIR}/kafka-logs
chmod -R 777 ${LOG_DIR}/kafka-logs

nohup ${KAFKA_HOME}/bin/zookeeper-server-start.sh ${KAFKA_HOME}/config/zookeeper.properties > ${LOG_DIR}/kafka-logs/zookeeper_service_start_$HOSTNAME.log 2> ${LOG_DIR}/kafka-logs/zookeeper_service_start_$HOSTNAME.err &
sleep 15
nohup ${KAFKA_HOME}/bin/kafka-server-start.sh ${KAFKA_HOME}/config/server.properties  > ${LOG_DIR}/kafka-logs/kafka_service_start_$HOSTNAME.log 2> ${LOG_DIR}/kafka-logs/kafka_service_start_$HOSTNAME.err &

