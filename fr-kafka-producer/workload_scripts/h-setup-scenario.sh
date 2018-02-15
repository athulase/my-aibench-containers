#!/bin/bash

if [ -z "$OVERRIDE_SETUP_SCENARIO" ]
then
      source ~/.bashrc
#      python /enable-ssh.py
#      sleep 5
      /create-topic.sh
else
      /bin/bash $OVERRIDE_SETUP_SCENARIO
fi