#!/bin/bash

if [ -z "$OVERRIDE_START_SCENARIO" ]
then
    source ~/.bashrc

    /facenet-init.sh
    sleep 10

    /facenet-allign.sh
    sleep 10

    /facenet-training.sh
    sleep 10

    /facenet-classify.sh
    sleep 10
else
    /bin/bash $OVERRIDE_START_SCENARIO
fi