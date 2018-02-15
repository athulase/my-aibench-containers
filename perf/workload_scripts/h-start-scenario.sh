#!/bin/bash

if [ -z "$OVERRIDE_START_SCENARIO" ]
then
    sleep 5
else
    /bin/bash $OVERRIDE_START_SCENARIO
fi