#!/bin/bash

if [ -z "$OVERRIDE_SETUP_SCENARIO" ]
then
    source ~/.bashrc

else
     /bin/bash $OVERRIDE_SETUP_SCENARIO
fi