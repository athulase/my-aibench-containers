##################################################################################
#/*
#* "INTEL CONFIDENTIAL" Copyright 2017 Intel Corporation All Rights
#* Reserved.
#*
#* The source code contained or described herein and all documents related
#* to the source code ("Material") are owned by Intel Corporation or its
#* suppliers or licensors. Title to the Material remains with Intel
#* Corporation or its suppliers and licensors. The Material contains trade
#* secrets and proprietary and confidential information of Intel or its
#* suppliers and licensors. The Material is protected by worldwide copyright
#* and trade secret laws and treaty provisions. No part of the Material may
#* be used, copied, reproduced, modified, published, uploaded, posted,
#* transmitted, distributed, or disclosed in any way without Intel's prior
#* express written permission.
#*
#* No license under any patent, copyright, trade secret or other
#* intellectual property right is granted to or conferred upon you by
#* disclosure or delivery of the Materials, either expressly, by
#* implication, inducement, estoppel or otherwise. Any license under such
#* intellectual property rights must be express and approved by Intel in
#* writing.
#*/

####################################################################################
#!/bin/bash
source ~/.bashrc
LOG_DIR=$SPARK_LOG_DIR_INFERENCE
unset SPARK_MASTER_PORT
mkdir -p "${LOG_DIR}"/spark-consumer-logs
chmod 777 "${LOG_DIR}"/spark-consumer-logs

SPARK_HOST=$SPK_HOST
#sed '/\<spark\>/!d' /etc/hosts > "${SPARK_HOME}"/conf/slaves

for i in $(seq 1 $(($SPARK_NODE_COUNT - 1)) ); do
    SLAVE=$SPARK_WORKLOAD_NAME-$i
    echo $SLAVE >> $SPARK_HOME/conf/slaves
done

#awk '{print $3}' $SPARK_HOME/conf/slaves > test.tmp && mv test.tmp $SPARK_HOME/conf/slaves
#sort -u $SPARK_HOME/conf/slaves > test.tmp && mv -f test.tmp $SPARK_HOME/conf/slaves
"${SPARK_HOME}"/sbin/start-master.sh >> "${LOG_DIR}"/spark-consumer-logs/spark_services_start_$HOSTNAME.log
"${SPARK_HOME}"/sbin/start-slaves.sh "${SPARK_IP}":7077 >>"${LOG_DIR}"/spark-consumer-logs/spark_services_start_$HOSTNAME.log

