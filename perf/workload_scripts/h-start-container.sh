#!/bin/bash

if [ -z "$OVERRIDE_START_CONTAINER" ]
then
    echo "nothing to be done in $0"
else
    /bin/bash $OVERRIDE_START_CONTAINER
fi