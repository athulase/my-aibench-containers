#!/bin/bash

if [ -z "$OVERRIDE_STOP_CONTAINER" ]
then
    pkill python
else
    /bin/bash $OVERRIDE_STOP_CONTAINER
fi