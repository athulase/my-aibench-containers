#!/bin/bash

if [ -z "$OVERRIDE_START_CONTAINER" ]
then
     # Locate the Discovery Service
    python3.6 /discovery-service.py client
    sleep 1

    # Workload
    echo "unset JAVA_HOME" >> ~/.bashrc; \
    echo 'export JAVA_HOME='"/usr/lib/jvm/java-1.8.0-openjdk" >> ~/.bashrc; \
    echo "export SPARK_IP=`hostname -i`" >> ~/.bashrc; \
    source ~/.bashrc
    LOG_DIR=$SPARK_LOG_DIR_INFERENCE
    cp /spark-streaming-kafka-0-8-assembly_2.11-2.2.0.jar "${SPARK_HOME}"/jars; \
    chmod 777 "${SPARK_HOME}"/jars/spark-streaming-kafka-0-8-assembly_2.11-2.2.0.jar
    for i in $(seq 0 $(($SPARK_NODE_COUNT - 1)) ); do
        NODE=$SPARK_WORKLOAD_NAME-$i
        echo $NODE >> /hostnames.txt
    done

    /fetch_dependencies.sh

    ssh-keygen -A
    mkdir -p ~/.ssh
    /usr/sbin/sshd &
    ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa
    echo 'export SPARK_WORKER_DIR='"${LOG_DIR}"/spark-consumer-logs >> $SPARK_HOME/conf/spark-env.sh

    export FLOW_NAME=$SPARK_WORKLOAD_NAME-0
    if [[ $HOSTNAME == $FLOW_NAME* ]]; then
        echo 'export SPARK_MASTER_HOST='$SPARK_IP >> $SPARK_HOME/conf/spark-env.sh;\
        echo 'export SPARK_EXECUTOR_MEMORY='2000m >> $SPARK_HOME/conf/spark-env.sh
    fi
    source $SPARK_HOME/conf/spark-env.sh
else
    /bin/bash $OVERRIDE_START_CONTAINER
fi