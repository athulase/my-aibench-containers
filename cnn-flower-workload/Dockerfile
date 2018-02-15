FROM 10.105.15.44:5000/tensorflow-gcc:15022018154805

RUN git clone https://github.com/tensorflow/models.git; \
   cd models; \
   git reset --hard 1e2ada24c6734b3f6f4e09cb98f66f3aad68de76; \
   cd research/inception/; \
   bazel build //inception:download_and_preprocess_flowers; \
   echo "unset JAVA_HOME" >> ~/.bashrc; \
   echo "export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.144-0.b01.el7_4.x86_64" >> ~/.bashrc; \
   source ~/.bashrc; \
   bazel build //inception:download_and_preprocess_flowers; \
   bazel-bin/inception/download_and_preprocess_flowers /flowers_data_dir;

###############################################################################
# Always change to /
###############################################################################
WORKDIR /

###############################################################################
# Make sure that everything is written to disk
###############################################################################
RUN sync