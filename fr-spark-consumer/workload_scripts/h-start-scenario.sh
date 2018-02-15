#!/bin/bash

if [ -z "$OVERRIDE_START_SCENARIO" ]
then
    source ~/.bashrc

    LOG_DIR=$SPARK_LOG_DIR_INFERENCE
    FLOW_NAME=$SPARK_WORKLOAD_NAME-0
    if [[ $HOSTNAME == $FLOW_NAME* ]]; then
      /start-spark-consumer.sh
      cp /spark-master.log "${LOG_DIR}"/spark-consumer-logs/spark-master-${HOSTNAME}.log
      #wait $BACK_PROCESS
    fi
else
    /bin/bash $OVERRIDE_START_SCENARIO
fi