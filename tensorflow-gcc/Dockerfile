FROM 10.105.15.44:5000/template-gcc:15022018154325

###############################################################################
# Copy everything to /
###############################################################################
COPY workload_scripts/* /
RUN chmod 777 /*.sh; \
    chmod 777 /*.py; \
    sync

###############################################################################
# Install Bazel
###############################################################################
RUN wget -q -O /tmp/bazel.sh https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh; \
    chmod 777 /tmp/bazel.sh; \
    sync; \
    /tmp/bazel.sh; \
    /bin/bash -c "source /usr/local/lib/bazel/bin/bazel-complete.bash";

###############################################################################
# Build Tensorflow
###############################################################################
RUN mkdir -p /root/TF/wheels
RUN git clone https://github.com/tensorflow/tensorflow /tensorflow; \
    cd /tensorflow; \
    git checkout 48c54eee17b35ebc15e274e36632eccb76072f7d
WORKDIR /tensorflow
RUN chmod 777 -R .; \
    sync
RUN updatedb; \
    echo -e "\n\ny\n\n\nn\n\n\n\n\n\n\n\n" | ./configure; \
    bazel build --config=mkl --copt="-DEIGEN_USE_VML" --copt="-mavx2" --copt="-O3" --copt="-mfma" -s -c opt //tensorflow/tools/pip_package:build_pip_package --verbose_failures; \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /root/TF/wheels/; \
    pip install --upgrade /root/TF/wheels/tensorflow-*.whl; \
    cd ..; \
    python -c 'import tensorflow as tf; print(tf.__version__)';

###############################################################################
# Always change to /
###############################################################################
WORKDIR /

###############################################################################
# Make sure that everything is written to disk
###############################################################################
RUN sync