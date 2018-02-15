FROM 10.105.15.44:5000/cnn-flower-workload:15022018154325

###############################################################################
# Copy everything to /
###############################################################################
COPY workload_scripts/* /
RUN mkdir -p /tf_cnn_scripts
COPY workload_scripts/tf_cnn_scripts/* /tf_cnn_scripts/
RUN chmod 777 /*.sh; \
    chmod 777 /*.py; \
    chmod 777 /tf_cnn_scripts/*; \
    sync

