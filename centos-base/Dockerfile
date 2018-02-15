FROM centos:centos7.4.1708

ENV http_proxy "http://proxy-chain.intel.com:911"
ENV https_proxy "http://proxy-chain.intel.com:912"

###############################################################################
# Proxy and other scripts
###############################################################################
COPY workload_scripts/* /
RUN chmod 777 /*.sh; \
    chmod 777 /*.py; \
    sync; \
    /proxy-enable.sh

###############################################################################
# Timezone
###############################################################################
RUN mv /etc/localtime /etc/localtime.bak && ln -s /usr/share/zoneinfo/America/Los_Angeles /etc/localtime

###############################################################################
# System dependencies
###############################################################################
RUN yum --enablerepo=extras install epel-release -y
RUN yum -y install https://centos7.iuscommunity.org/ius-release.rpm
RUN yum update -y && yum install -y \
    bc \
    epel-release \
    etcd \
    java-1.8.0-openjdk-devel \
    net-tools \
    python-devel \
    python-pip \
    python36u \
    openssh-clients \
    openssh-server \
    vim \
    wget \
    which

###############################################################################
# Others
###############################################################################
RUN echo 'export JAVA_HOME=/usr/lib/jvm/jre-1.8.0-openjdk' >> ~/.bashrc; \
    echo 'export JRE_HOME=/usr/lib/jvm/jre' >> ~/.bashrc; \
    echo 'export PATH='"${PATH}":'$JAVA_HOME/bin' >> ~/.bashrc

###############################################################################
# Install Python3 dependencies
###############################################################################
RUN python3.6 -m ensurepip; \
    python3.6 -m pip install etcd3==0.7.0; \
    python3.6 -m pip install PyYAML==3.12
