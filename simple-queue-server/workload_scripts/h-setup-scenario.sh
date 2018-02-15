#!/bin/bash

if [ -z "$OVERRIDE_SETUP_SCENARIO" ]
then
     source ~/.bashrc
     screen -S simple-queue -dm ./start.sh
else
      /bin/bash $OVERRIDE_SETUP_SCENARIO
fi