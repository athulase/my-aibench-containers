FROM 10.105.15.44:5000/centos-base:15022018154325

###############################################################################
# System dependencies
###############################################################################
RUN yum update -y && yum groupinstall "Development tools" -y && yum install -y \
    expect \
    gcc-c++ \
    gcc \
    gettext \
    glibc-devel \
    glibc-devel.i686 \
    gmp-devel \
    mpfr-devel \
    libmpc-devel \
    lapack-devel \
    lcms2-devel \
    libffi-devel \
    libgcc.i686 \
    libgfortran \
    libjpeg-devel \
    libmpc-devel \
    libpng-devel \
    libraqm-devel \
    libsndfile \
    libSM \
    libtiff-devel \
    libtool \
    libwebp-devel \
    libXext \
    libXi-devel \
    libXmu-devel \
    texinfo \
    wget

###############################################################################
# Build and install GCC system wide
###############################################################################
RUN wget -q -O /tmp/gcc.tar.gz https://ftp.gnu.org/gnu/gcc/gcc-6.2.0/gcc-6.2.0.tar.gz; \
    mkdir -p /opt/gcc; \
    tar -xf /tmp/gcc.tar.gz --directory /opt/gcc --strip 1; \
    cd /opt/gcc; \
    ./configure --prefix=/usr/; \
    make -j `cat /proc/cpuinfo | grep processor | wc -l`; \
    make install; \
    sync; \
    rm -rf /tmp/gcc.tar.gz; \
    rm -rf /opt/gcc; \
    sync
