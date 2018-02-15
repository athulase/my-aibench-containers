#!/bin/bash

if [ -z "$OVERRIDE_START_CONTAINER" ]
then
     # Locate the Discovery Service
    OWN_IP=`hostname -i`
    python3.6 /discovery-service.py client
    sleep 1

#     SPARK_MASTER=""
#     echo $SPARK_MASTER >> /hostnames.txt
    echo "export KAFKA_IP=`hostname -i`" >> ~/.bashrc; \
    source ~/.bashrc

    sed -ie "s/fr-kafka-producer-0/$KAFKA_IP" server.properties


#    for i in $(seq 0 $(($SPARK_NODE_COUNT - 1)) ); do
#        NODE=$SPARK_WORKLOAD_NAME-$i
#        echo $NODE >> /hostnames.txt
#    done
#    ssh-keygen -A
#    mkdir -p ~/.ssh
#    /usr/sbin/sshd &
#    ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa

    /start-services.sh

else
    /bin/bash $OVERRIDE_START_CONTAINER
fi