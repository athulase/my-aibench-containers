#!/bin/bash

if [ -z "$OVERRIDE_SETUP_SCENARIO" ]
then
      echo "not empty $0"
else
      /bin/bash $OVERRIDE_SETUP_SCENARIO
fi