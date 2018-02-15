#!/bin/bash

if [ -z "$OVERRIDE_STOP_SCENARIO" ]
then
    source ~/.bashrc

    # Locate reportip server
    addr=`python3.6 /discovery-service.py query reportip`

    # Log the data in reporting DB
    end_time=$(date +'%Y-%m-%d %H:%M:%S')
    if [ "$DRY_RUN" = False ]; then
        curl -X PUT -H "Content-Type: application/json" -d '{"end_time":"'"$end_time"'"}' http://$addr/flow/runid=$RUNID
    fi

    # Stop the performance measurement tools
    /performance-stop.sh
else
    /bin/bash $OVERRIDE_STOP_SCENARIO
fi