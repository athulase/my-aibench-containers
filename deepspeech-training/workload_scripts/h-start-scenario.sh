#!/bin/bash

if [ -z "$OVERRIDE_START_SCENARIO" ]
then
    /start_deep_speech_training.sh
else
    /bin/bash $OVERRIDE_START_SCENARIO
fi