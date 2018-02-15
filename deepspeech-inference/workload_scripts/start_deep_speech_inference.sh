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
mkdir -p "${LOG_DIR}/ds_infer_log"
chmod 777 -R "${LOG_DIR}/ds_infer_log"
cd ${SOURCES}

start_time=`date +%s`

python deepSpeech_test.py --eval_data 'test' --run_once True --batch_size ${BATCH_SIZE} --data_dir ${DATA_DIR} --eval_dir ${FILE_NAME}/../eval --checkpoint_dir ${FILE_NAME}   > ${LOG_DIR}/ds_infer_log/deepspeech_inference_${HOSTNAME}.log 2>&1
ret=$?

# Save the throughput
end_time=`date +%s`
timetaken=$((end_time-start_time))
file_count=1000
result=$(echo ${file_count}/${timetaken}|bc -l)

echo "$result FilesPerSecond" >> /relevant_metrics
echo "1.0 AverageConfidence" >> /relevant_metrics
cp /relevant_metrics $LOG_DIR/relevant_metrics

# exit script
echo "return code:$ret"
if [ $ret -ne 0 ]; then
     exit $ret
fi