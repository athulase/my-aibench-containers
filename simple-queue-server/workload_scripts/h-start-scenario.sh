#!/bin/bash

if [ -z "$OVERRIDE_START_SCENARIO" ]
then
    echo "not empty"
else
    /bin/bash $OVERRIDE_START_SCENARIO
fi