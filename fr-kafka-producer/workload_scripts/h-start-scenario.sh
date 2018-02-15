#!/bin/bash

if [ -z "$OVERRIDE_START_SCENARIO" ]
then
    source ~/.bashrc
    sleep 40

    /start-producer-tf.sh
else
    /bin/bash $OVERRIDE_START_SCENARIO
fi