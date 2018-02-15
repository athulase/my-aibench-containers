#!/bin/bash

source ~/.bashrc

nr_of_producers=$SIMPLE_QUEUE_NR_OF_PRODUCERS
nr_of_consumers=$SIMPLE_QUEUE_NR_OF_CONSUMERS
recv_bufsize=$SIMPLE_QUEUE_RECV_BUFSIZE

LOGDIR=$SIMPLE_QUEUE_LOG_DIR
QUEUE_BACKEND=$QUEUE_BACKEND
OWN_IP=`hostname -i`

# Locate the Discovery Service and register services
python3.6 /discovery-service.py client
sleep 1
sync
python3.6 /discovery-service.py register $QUEUE_BACKEND $OWN_IP "50000:50001"

mkdir -p $LOGDIR

export SIMPLE_QUEUE_PUSH_PORT=50000
export SIMPLE_QUEUE_GET_PORT=50001

python /simple-queue-server.py > $LOGDIR/simple-queue-server.log 2>&1
echo "After queue"
