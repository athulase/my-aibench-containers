#!/bin/bash

python3.6 /discovery-service.py server &
/usr/bin/etcd --config-file /etcd.json &