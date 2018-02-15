FROM 10.105.15.44:5000/template-gcc:15022018154325

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
RUN yum install -y \
    bc

###############################################################################
# Get and compile Marian
###############################################################################
# cmake
RUN wget -q -O /tmp/cmake.tar.gz https://cmake.org/files/v3.9/cmake-3.9.3-Linux-x86_64.tar.gz; \
    mkdir -p /cmake; \
    tar -xf /tmp/cmake.tar.gz --directory /cmake --strip 1; \
    chmod 777 -R /cmake; \
    sync; \
    cd /; \
    rm -f /tmp/cmake.tar.gz

# Boost
RUN wget -q -O /tmp/boost.tar.gz https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.gz; \
    mkdir -p /tmp/boost; \
    tar -xf /tmp/boost.tar.gz --directory /tmp/boost --strip 1; \
    cd /tmp/boost; \
    ./bootstrap.sh; \
    ./b2 -j `cat /proc/cpuinfo | grep processor | wc -l`; \
    ./b2 install; \
    cd /; \
    rm -f /tmp/boost.tar.gz

# Marian
RUN git clone https://github.com/marian-nmt/marian.git /marian; \
    cd /marian; \
    git checkout 5c9b850161b022df3424dbdb9e8ac26156c73502; \
    mkdir build; \
    cd build; \
    /cmake/bin/cmake .. -DCUDA=off; \
    make -j `cat /proc/cpuinfo | grep processor | wc -l`; \
    rm -rf /cmake; \
    git clone https://github.com/marian-nmt/moses-scripts /marian/examples/translate/moses-scripts; \
    for i in truecase-model.en model.npz vocab.en.json vocab.de.json ende.bpe;do wget -q -P /marian/examples/translate/en-de http://data.statmt.org/rsennrich/wmt16_systems/en-de/"${i}";done

###############################################################################
# Always change to /
###############################################################################
WORKDIR /

###############################################################################
# Make sure that everything is written to disk
###############################################################################
RUN sync
