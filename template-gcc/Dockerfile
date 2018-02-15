FROM 10.105.15.44:5000/centos-gcc620:15022018154325

###############################################################################
# System dependencies
###############################################################################
RUN yum update -y && yum install -y \
    autoconf \
    automake \
    atlas-devel \
    binutils \
    blas-devel \
    boost \
    boost-devel \
    flex \
    freeglut-devel \
    freetype \
    freetype-devel \
    future \
    gettext \
    git \
    hdf5-devel \
    make \
    mlocate \
    mpfr-devel \
    net-tools \
    numactl \
    numpy  \
    opencv* \
    openssl-devel \
    openssl-libs \
    patch \
    perf \
    pkgconfig \
    python-devel \
    python-pip \
    suitesparse-devel \
    sysstat \
    tcl-devel \
    tk-devel \
    tkinter \
    which \
    zlib-devel

###############################################################################
# Install Python2 dependencies
###############################################################################
RUN pip install --upgrade pip; \
    pip install setuptools; \
    pip install enum34; \
    pip install futures; \
    pip install 'grpcio>=1.1.3'; \
    pip install numpy; \
    pip install h5py; \
    pip install kafka; \
    pip install mock; \
    pip install pexpect; \
    pip install --pre 'protobuf>=3.0.0a3'; \
    pip install pyspark; \
    pip install python-Levenshtein; \
    pip install python_speech_features; \
    pip install PySoundFile; \
    pip install psutil; \
    pip install requests; \
    pip install scipy; \
    pip install scikit-learn; \
    pip install six; \
    pip install tqdm; \
    pip install wheel; \
    pip install matplotlib; \
    pip install Pillow; \
    pip install scikit-image; \
    pip install opencv-python

###############################################################################
# Other scripts
###############################################################################
COPY workload_scripts/* /
RUN chmod 777 /*.sh; \
    chmod 777 /*.py; \
    sync
