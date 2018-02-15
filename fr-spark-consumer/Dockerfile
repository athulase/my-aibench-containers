FROM 10.105.15.44:5000/tensorflow-gcc:15022018154805

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
# Setup Google drive LFW
###############################################################################
WORKDIR /
RUN python /gdown.py 0B5MzpY9kBtDVZ2RpVDYwWmxoSUk /tmp/20170512-110547.zip; \
    cd /tmp ;\
    unzip 20170512-110547.zip; \
    cp /tmp/20170512-110547/20170512-110547.pb /20170512-110547.pb; \
    rm -f /tmp/20170512-110547.zip; \
    rm -r /tmp/20170512-110547

###############################################################################
# Install Scala & Spark
###############################################################################
RUN mkdir -p /opt/spark; \
    wget -q -O /tmp/scala.rpm http://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.rpm; \
    wget -q -O /tmp/spark.tar.gz http://www.us.apache.org/dist/spark/spark-2.2.0/spark-2.2.0-bin-hadoop2.7.tgz; \
    yum install -y /tmp/scala.rpm; \
    tar xzf /tmp/spark.tar.gz --directory /opt/spark --strip 1; \
    chmod 777 -R /opt/spark; \
    sync; \
    cp /opt/spark/conf/spark-env.sh.template /opt/spark/conf/spark-env.sh; \
    rm -f /tmp/spark.tar.gz; \
    rm -f /tmp/scala.rpm

##############################################################################
# Setup large dependencies. Temporary solution.
##############################################################################
RUN wget -q http://central.maven.org/maven2/org/apache/spark/spark-streaming-kafka-0-8-assembly_2.11/2.2.0/spark-streaming-kafka-0-8-assembly_2.11-2.2.0.jar

##############################################################################
# Set SPARK_HOME
##############################################################################
RUN	echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc; \
	echo 'export PATH=$PATH:$SPARK_HOME/bin' >> ~/.bashrc; \
	yum -y install screen; \
	source ~/.bashrc

###############################################################################
# Always change to /
###############################################################################
WORKDIR /

###############################################################################
# Make sure that everything is written to disk
###############################################################################
RUN sync
