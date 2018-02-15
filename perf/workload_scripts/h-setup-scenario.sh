#!/bin/bash

if [ -z "$OVERRIDE_SETUP_SCENARIO" ]
then
    source ~/.bashrc

    LOG_DIR=$PERFORMANCE_LOG_DIR
    mkdir -p $LOG_DIR

    TOOLS=$PERFORMANCE_TOOLS
    BACKEND=(${TOOLS//,/ })
    SLEEP_INTERVAL=99999
    for index in ${!BACKEND[@]}; do
        TOOL=${BACKEND[index]};
        if [ $TOOL = "perf" ]; then
            # perf
            echo 0 > /proc/sys/kernel/perf_event_paranoid
            nohup perf record -o $LOG_DIR/perf.data -a /bin/sleep $SLEEP_INTERVAL > $LOG_DIR/perf.txt 2>&1 &
        fi
        if [ $TOOL = "sar" ]; then
            #CPU
            nohup sar -P ALL 1 > $LOG_DIR/cpu.txt 2>&1 &
            #network
            nohup sar -n DEV 1 > $LOG_DIR/network.txt 2>&1 &
            #memory
            nohup sar -rS 1 > $LOG_DIR/mem.txt 2>&1 &
            #c/s
            nohup sar -w 1 > $LOG_DIR/cs.txt 2>&1 &
            #I/O
            nohup iostat -x -k > $LOG_DIR/io.txt 2>&1 &
        fi
        if [ $TOOL = "emon" ]; then
            EMON_DIR=$PERFORMANCE_EMON_DIR
            EMON_EVENTS=$PERFORMANCE_EMON_EVENTS

            ## Build EMON drivers
            #cd $EMON_DIR/sepdk/src
            #KERNEL=`uname -r`
            #rpm -ivh ftp://mirror.switch.ch/pool/4/mirror/scientificlinux/7.1/x86_64/updates/security/kernel-devel-$KERNEL.rpm
            #echo -e "\n\n/usr/src/kernels/$KERNEL\n" | ./build-driver
            $EMON_DIR/sepdk/src/insmod-sep

            cd /
            source $EMON_DIR/sep_vars.sh
            #https://software.intel.com/sites/default/files/managed/67/19/emon_user_guide_0.pdf
            nohup emon -t0 -C $EMON_EVENTS /bin/sleep $SLEEP_INTERVAL > $LOG_DIR/emon.log 2>&1 &
        fi
        if [ $TOOL = "vtune" ]; then
            VTUNE_DIR=$PERFORMANCE_VTUNE_DIR
            VTUNE_ANALYSIS=$PERFORMANCE_VTUNE_ANALYSIS
            export INTEL_LICENSE_FILE=$VTUNE_DIR/../licenses
            source $VTUNE_DIR/amplxe-vars.sh
            source $VTUNE_DIR/apsvars.sh
            nohup amplxe-cl -collect $VTUNE_ANALYSIS --duration $SLEEP_INTERVAL -r $LOG_DIR/r001cc > $LOG_DIR/vtune.log 2>&1 &
        fi
    done
else
    /bin/bash $OVERRIDE_SETUP_SCENARIO
fi