#!/bin/bash

source ~/.bashrc

LOG_DIR=$PERFORMANCE_LOG_DIR
mkdir -p $LOG_DIR

TOOLS=$PERFORMANCE_TOOLS
BACKEND=(${TOOLS//,/ })
for index in ${!BACKEND[@]}; do
    TOOL=${BACKEND[index]};
    if [ $TOOL = "perf" ]; then
        pkill perf
        sync
        perf report -f -i $LOG_DIR/perf.data > $LOG_DIR/perf.data.report.txt
    fi
    if [ $TOOL = "sar" ]; then
        pkill sar
    fi
    if [ $TOOL = "emon" ]; then
        pkill emon
    fi
    if [ $TOOL = "vtune" ]; then
        pkill amplxe-cl
        sync
        VTUNE_DIR=$PERFORMANCE_VTUNE_DIR
        export INTEL_LICENSE_FILE=$VTUNE_DIR/../licenses
        source $VTUNE_DIR/amplxe-vars.sh
        source $VTUNE_DIR/apsvars.sh
        amplxe-cl -report hotspots -r $LOG_DIR/r001cc > $LOG_DIR/r001cc.csv
    fi
done
