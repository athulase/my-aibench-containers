#!/bin/bash

if [ -z "$OVERRIDE_START_CONTAINER" ]
then
    source ~/.bashrc

     # Locate the Discovery Service
    python3.6 /discovery-service.py client
    sleep 1

    /fetch_dependencies.sh

#   echo "unset JAVA_HOME" >> ~/.bashrc; \
#   echo 'export JAVA_HOME='"/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.144-0.b01.el7_4.x86_64" >> ~/.bashrc; \
    echo "export SPARK_IP=`hostname -i`" >> ~/.bashrc; \
else
    /bin/bash $OVERRIDE_START_CONTAINER
fi