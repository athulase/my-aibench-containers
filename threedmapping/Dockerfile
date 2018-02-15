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
# Get and compile everything
###############################################################################
RUN export CXXFLAGS=-std=c++11

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

# Glog
RUN git clone https://github.com/google/glog /tmp/glog; \
    cd /tmp/glog; \
    git reset --hard 2a6df66252d266080489c310b8146e63b66b2add; \
    chmod 777 -R *; \
    sync; \
    ./autogen.sh; \
    ./configure; \
    make -j `cat /proc/cpuinfo | grep processor | wc -l`; \
    make install; \
    cd /; \
    rm -rf /tmp/glog

# Gflags
RUN git clone https://github.com/gflags/gflags /tmp/gflags; \
    cd /tmp/gflags; \
    git reset --hard 60784b53e364c2e2594916bc84af075c4f679fa8; \
    chmod 777 -R *; \
    sync; \
    /cmake/bin/cmake /tmp/gflags; \
    make -j `cat /proc/cpuinfo | grep processor | wc -l`; \
    make install; \
    cd /; \
    rm -rf /tmp/gflags

# Eigen
RUN wget -q -O /tmp/eigen.tar.bz2 http://bitbucket.org/eigen/eigen/get/3.3.3.tar.bz2; \
    mkdir -p /tmp/eigen; \
    tar -xf /tmp/eigen.tar.bz2 --directory /tmp/eigen --strip 1; \
    chmod 777 -R /tmp/eigen/*; \
    sync; \
    cd /tmp/eigen; \
    mkdir build; \
    cd build; \
    /cmake/bin/cmake /tmp/eigen; \
    make -j `cat /proc/cpuinfo | grep processor | wc -l`; \
    make install; \
    cd /; \
    rm -f  /tmp/eigen.tar.bz2; \
    rm -rf /tmp/eigen

# Ceres
RUN wget -q -O /tmp/ceres.tar.gz http://ceres-solver.org/ceres-solver-1.13.0.tar.gz; \
    mkdir -p /tmp/ceres; \
    tar -xf /tmp/ceres.tar.gz --directory /tmp/ceres --strip 1; \
    chmod 777 -R /tmp/ceres/*; \
    sync; \
    mkdir /tmp/ceres-bin; \
    cd /tmp/ceres-bin; \
    /cmake/bin/cmake /tmp/ceres; \
    make -j `cat /proc/cpuinfo | grep processor | wc -l`; \
    make install; \
    cd /; \
    rm -rf /tmp/ceres; \
    rm -f  /tmp/ceres.tar.gz

# Ilmbase
RUN wget -q -O /tmp/ilmbase.tar.gz http://download.savannah.nongnu.org/releases/openexr/ilmbase-2.2.0.tar.gz; \
    mkdir -p /tmp/ilmbase; \
    tar -xf /tmp/ilmbase.tar.gz --directory /tmp/ilmbase --strip 1; \
    chmod 777 -R /tmp/ilmbase; \
    sync; \
    cd /tmp/ilmbase; \
    ./bootstrap; \
    ./configure ; \
    make -j `cat /proc/cpuinfo | grep processor | wc -l`; \
    make install; \
    cd /; \
    rm -rf /tmp/ilmbase; \
    rm -f  /tmp/ilmbase.tar.gz

# Openexr
RUN wget -q -O /tmp/openexr.tar.gz http://download.savannah.nongnu.org/releases/openexr/openexr-2.2.0.tar.gz; \
    mkdir -p /tmp/openexr; \
    tar -xf /tmp/openexr.tar.gz --directory /tmp/openexr --strip 1; \
    chmod 777 -R /tmp/openexr; \
    sync; \
    cd /tmp/openexr; \
    export LD_LIBRARY_PATH=/usr/local/lib; \
    ./configure; \
    make -j `cat /proc/cpuinfo | grep processor | wc -l`; \
    make install; \
    cd /; \
    rm -rf /tmp/openexr; \
    rm -rf /tmp/openexr.tar.gz

# OpenimageIO
RUN git clone https://github.com/OpenImageIO/oiio /tmp/oiio; \
    cd /tmp/oiio; \
    git checkout remotes/origin/RB-1.7; \
    chmod 777 -R /tmp/oiio; \
    sync; \
    mkdir build; \
    cd build; \
    /cmake/bin/cmake /tmp/oiio; \
    make -j `cat /proc/cpuinfo | grep processor | wc -l`; \
    make install; \
    cd /; \
    rm -rf /tmp/oiio

# Theia
RUN git clone https://github.com/sweeneychris/TheiaSfM /tmp/theia; \
    cd /tmp/theia; \
    git reset --hard 56a4ef7b35ebcc10668a923139acb2081a2bc4fa; \
    chmod 777 -R /tmp/theia; \
    sync; \
    mkdir -p /theia-build; \
    cd /theia-build; \
    /cmake/bin/cmake /tmp/theia; \
    make -j `cat /proc/cpuinfo | grep processor | wc -l`; \
    rm -rf /tmp/theia

###############################################################################
# Always change to /
###############################################################################
WORKDIR /

###############################################################################
# Make sure that everything is written to disk
###############################################################################
RUN sync
