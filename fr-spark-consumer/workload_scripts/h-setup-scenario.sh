#!/bin/bash

if [ -z "$OVERRIDE_SETUP_SCENARIO" ]
then
    source ~/.bashrc
    python /enable-ssh.py
    sleep 5
    LOG_DIR=$SPARK_LOG_DIR_INFERENCE
    export FLOW_NAME=$SPARK_WORKLOAD_NAME-0
    if [[ $HOSTNAME == $FLOW_NAME* ]]; then
      /start-spark-services.sh
    else
      screen -S preprocess -dm /launch_preprocess.sh
    fi

else
     /bin/bash $OVERRIDE_SETUP_SCENARIO
fi