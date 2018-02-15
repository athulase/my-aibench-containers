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

#SPARK_HOST=$SPK_HOST
LOG_DIR=$SPARK_LOG_DIR_INFERENCE
MODEL=$SPARK_MODEL
PKL_FILE=$SPARK_LFW
BATCH_SIZE=$SPARK_BATCH_SIZE
KAFKA_PRODUCER=$SPARK_KAFKA_PRODUCER-0
QUEUE_SERVER=$SPARK_QUEUE_SERVER
QUEUE_PORT=50000
KAFKA_HOSTNAME=$KAFKA_PRODUCER
#echo $KAFKA_PRODUCER >> "${LOG_DIR}"/spark-consumer-logs/spark_consumer_$HOSTNAME_rinf.log
TOPIC_NAME=$SPARK_TOPIC_NAME
NUM_SLAVES=$(($SPARK_NODE_COUNT - 1))
NO_OF_PROCESSORS=`cat /proc/cpuinfo | grep processor | wc -l`
SPARK_EXECS=$NUM_SLAVES
SPARK_EXEC_CORES=$NO_OF_PROCESSORS

mkdir -p "${LOG_DIR}"/spark-consumer-logs
chmod 777 -R "${LOG_DIR}"/spark-consumer-logs

TAG_DIR=${LOG_DIR}/tags_fr
mkdir -p "${TAG_DIR}"
chmod 777 -R "${TAG_DIR}"

#mv spark*.jar "${SPARK_HOME}"/jars/

spark-submit --num-executors ${SPARK_EXECS} \
             --executor-cores ${SPARK_EXEC_CORES} \
             --driver-java-options "-Dlog4j.configuration=file:/spark-log.properties" \
             --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:/spark-log.properties" \
             --master spark://${SPARK_IP}:7077 \
             --conf spark.default.parallelism=$NUM_SLAVES \
             --jars "${SPARK_HOME}"/jars/spark-streaming-kafka-0-8-assembly_2.11-2.2.0.jar,"${SPARK_HOME}"/jars/scala-library-2.11.8.jar \
              /decoupled_spark.py --zkQuorum "${KAFKA_HOSTNAME}":2080 \
                                  --kafka_producer "${KAFKA_HOSTNAME}":9092 \
                                  --topic "${TOPIC_NAME}" --wl_label "fr-spark-batch-inference"
                                  --kafka_topic_number "${SPARK_KAFKA_TOPIC_NUMBER}" > "${LOG_DIR}"/spark-consumer-logs/spark_consumer_$HOSTNAME.log 2>"${LOG_DIR}"/spark-consumer-logs/spark_consumer_$HOSTNAME.err

#BG_PROCESS=$!
#echo "export BACK_PROCESS=$BG_PROCESS" >> ~/.bashrc

