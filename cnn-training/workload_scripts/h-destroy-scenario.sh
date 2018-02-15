#!/bin/bash

if [ -z "$OVERRIDE_DESTROY_SCENARIO" ]
then
    echo "nothing to be done in $0"
else
    /bin/bash $OVERRIDE_DESTROY_SCENARIO
fi