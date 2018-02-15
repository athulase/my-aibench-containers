#!/usr/bin/env bash

source ~/.bashrc
cd /; \
rm -f 20170512-110547.pb; \
cp $FACENET_HOME_DIR/src/align/det*.npy . ; \
cp $FACENET_HOME_DIR/models/20170512-110547/20170512-110547.pb . ; \
chmod 777 20170512-110547.pb ; \
cp $FACENET_HOME_DIR/models/20170512-110547/lfw_classifier.pkl . ; \
chmod 777 lfw_classifier.pkl .