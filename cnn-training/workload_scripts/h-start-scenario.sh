#!/bin/bash

if [ -z "$OVERRIDE_START_SCENARIO" ]
then
    /start-tf-batch-training.sh
else
    /bin/bash $OVERRIDE_START_SCENARIO
fi