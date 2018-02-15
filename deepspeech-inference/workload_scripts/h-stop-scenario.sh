#!/bin/bash

if [ -z "$OVERRIDE_STOP_SCENARIO" ]
then
   source ~/.bashrc

    # Locate reportip server
    addr=`python3.6 /discovery-service.py query reportip`

    # Log the data in reporting DB
    end_time=$(date +'%Y-%m-%d %H:%M:%S')
    throughput_value=`head -n 1 relevant_metrics | awk '{print $1}'`
    throughput_meaning=`head -n 1 relevant_metrics | awk '{print $2}'`
    if [ "$DRY_RUN" = False ]; then
        curl -X PUT -H "Content-Type: application/json" -d '{ "end_time":"'"$end_time"'", "throughput_value":"'"$throughput_value"'", "throughput_meaning":"'"$throughput_meaning"'"  }' http://$addr/flow/runid=$RUNID
    fi

    # Stop the performance measurement tools
    /performance-stop.sh
else
    /bin/bash $OVERRIDE_STOP_SCENARIO
fi