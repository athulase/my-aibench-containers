#!/bin/bash

if [ -z "$OVERRIDE_START_CONTAINER" ]
then
    # Locate the Discovery Service
    OWN_IP=`hostname -i`
    python3.6 /discovery-service.py client
    sleep 1


else
    /bin/bash $OVERRIDE_START_CONTAINER
fi