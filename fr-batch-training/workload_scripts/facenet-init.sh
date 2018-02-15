#!/bin/bash

source ~/.bashrc

FACENET_HOME=$FACENET_HOME_DIR
LOG_DIR=$FACENET_LOG_DIR
DATA_DIR=$FACENET_HOME_DIR/datasets/lfw
MODEL_DIR=$FACENET_HOME_DIR/models
RAW_DIR=$FACENET_HOME_DIR/datasets/lfw/raw

# Create the dirs
mkdir -p $FACENET_HOME

# Clone the Facenet repo
cd $FACENET_HOME
cd ..
rm -rf facenet
git clone https://github.com/davidsandberg/facenet.git facenet
cd facenet
git checkout 4faf590600f122c3cd2ab3ab3c85bd3bd2d00822
git apply /facenet-patch.diff

# create the rest of the dirs
mkdir -p $LOG_DIR
mkdir -p $DATA_DIR
mkdir -p $MODEL_DIR
mkdir -p $RAW_DIR

sync

# Get dependencies
cd $MODEL_DIR
python /gdown.py 0B5MzpY9kBtDVZ2RpVDYwWmxoSUk 20170512-110547.zip
unzip 20170512-110547.zip

cd $DATA_DIR
wget -q http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar xf lfw.tgz -C $RAW_DIR --strip-components=1

# set the extra env value
echo 'export PYTHONPATH=$FACENET_HOME_DIR/src' >> ~/.bashrc
