#!/bin/bash

if [ -z "$OVERRIDE_STOP_CONTAINER" ]
then
    echo "nothing to be done in $0"
else
    /bin/bash $OVERRIDE_STOP_CONTAINER
fi