FROM 10.105.15.44:5000/centos-base:15022018154805

###############################################################################
# Others
###############################################################################
COPY workload_scripts/* /
RUN chmod 777 /*.sh; \
    chmod 777 /*.py; \
    sync

###############################################################################
# System dependencies
###############################################################################
RUN yum groupinstall "Development tools" -y && yum install -y \
    gcc-c++ \
    gcc \
    perf \
    sysstat

###############################################################################
# Always change to /
###############################################################################
WORKDIR /

###############################################################################
# Make sure that everything is written to disk
###############################################################################
RUN sync
