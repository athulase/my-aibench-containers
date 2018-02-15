#!/bin/bash

source ~/.bashrc
#PS1="[\u@\h \W] \$"
LOG_DIR=$DS_LOG_DIR
DATA_DIR=$DS_DATA_DIR
FILE_NAME=$DS_FILE_NAME
BATCH_SIZE=$DS_BATCH_SIZE
STEPS=$DS_STEPS
CPU=$DS_CPU
NCHW=$DS_NCHW
DUMMY=$DS_DUMMY
ENGINE=$DS_ENGINE
SOURCES=/root/deepSpeech/src
mkdir -p "${LOG_DIR}/ds_train_log"
chmod 777 -R "${LOG_DIR}/ds_train_log"
cd ${SOURCES}
echo ${FILE_NAME}

python deepSpeech_train.py --platform ${CPU} --batch_size ${BATCH_SIZE} --no-shuffle --max_steps ${STEPS} --num_rnn_layers 7 --num_hidden 1760 --num_filters 32 --initial_lr 1e-4 --temporal_stride 4 --train_dir ${FILE_NAME} --data_dir ${DATA_DIR} --debug false --dummy ${DUMMY} --nchw ${NCHW} --engine ${ENGINE} 2>&1
ret=$?
echo "return code:$ret"
if [ $ret -ne 0 ]; then
     exit $ret
fi 

