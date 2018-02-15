FROM 10.105.15.44:5000/template-gcc:15022018154805

###############################################################################
# Copy everything to /
###############################################################################
COPY workload_scripts/* /
RUN chmod 777 /*.sh; \
    chmod 777 /*.py; \
    sync


###############################################################################
# System dependencies
###############################################################################
RUN yum update -y && yum install -y \
    openssh-clients \
    openssh-server

RUN pip install paramiko


###############################################################################
# Permit SSH login as root
###############################################################################
RUN mkdir /var/run/sshd; \
    echo 'root:passwordai' | chpasswd; \
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config;\
    echo "UseDNS no" >> /etc/ssh/sshd_config;\
    echo "export VISIBLE=now" >> /etc/profile

###############################################################################
# Install Apache Kafka
###############################################################################
RUN mkdir /opt/kafka; \
    wget -q -O /tmp/kafka.tgz http://www.us.apache.org/dist/kafka/0.11.0.2/kafka_2.11-0.11.0.2.tgz; \
    tar -xf /tmp/kafka.tgz --directory /opt/kafka --strip 1; \
    rm -f /tmp/kafka.tgz; \
    chmod 777 -R /opt/kafka; \
    sync; \
    echo 'export KAFKA_HOME=/opt/kafka' >> ~/.bashrc; \
    source ~/.bashrc; \
    mkdir -p /tmp/zookeeper; \
    echo 1 > /tmp/zookeeper/myid
COPY [ "workload_scripts/server.properties", "workload_scripts/zookeeper.properties", "/opt/kafka/config/"]

###############################################################################
# Always change to /
###############################################################################
WORKDIR /

###############################################################################
# Make sure that everything is written to disk
###############################################################################
RUN sync
