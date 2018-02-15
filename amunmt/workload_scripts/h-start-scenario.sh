#!/bin/bash

if [ -z "$OVERRIDE_START_SCENARIO" ]
then
    source ~/.bashrc

    # Start the performance measurement tools
    /performance-start.sh

    # Locate reportip server
    addr=`python3.6 /discovery-service.py query reportip`

    # Log the data in reporting DB
    if [ "$DRY_RUN" = False ]; then
        start_time=$(date +'%Y-%m-%d %H:%M:%S')
        dataset="EN-DE"
        model="EN-DE"
        ml_framework="Marian"
        ml_framework_version="1.4.0"
        ml_framework_compiler=`gcc -v 2>&1 >/dev/null | grep 'version'`
        cpu=`cat /proc/cpuinfo | grep 'model name' | uniq | awk -F": " '{print $2}'`
        curl -X PUT -H "Content-Type: application/json" -d '{"dataset":"'"$dataset"'", "model":"'"$model"'", "ml_framework":"'"$ml_framework"'", "ml_framework_version":"'"$ml_framework_version"'", "ml_framework_compiler":"'"$ml_framework_compiler"'", "cpu":"'"$cpu"'", "start_time":"'"$start_time"'"}' http://$addr/flow/runid=$RUNID
    fi

    # Start the scenario
    /startamun.sh
else
    /bin/bash $OVERRIDE_START_SCENARIO
fi