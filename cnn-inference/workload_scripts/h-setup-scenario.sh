#!/bin/bash

if [ -z "$OVERRIDE_SETUP_SCENARIO" ]
then
    echo "nothing to be done in $0"
else
    /bin/bash $OVERRIDE_SETUP_SCENARIO
fi