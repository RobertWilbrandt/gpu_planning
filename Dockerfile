FROM wilbrandt/cuda_dev_docker:9.1
LABEL maintainer="Robert Wilbrandt <robert@stamm-wilbrandt.de>"
LABEL description="Build environment for the gpu_planning project"

# Install boost libraries
RUN apt install -y libboost-log-dev libboost-regex-dev libboost-program-options-dev

# Install recent gtest
RUN apt install -y git \
  && mkdir -p /system \
  && git clone --branch release-1.10.0 --depth 1 https://github.com/google/googletest.git /system/googletest \
  && mkdir /system/googletest/build \
  && cd /system/googletest/build \
  && cmake .. -GNinja \
  && cmake --build . --target install

# Install doxygen
RUN apt install -y doxygen

# Set up custom entry point
COPY ./docker_entrypoint.sh /gpu_planning_entrypoint.sh
ENTRYPOINT ["/gpu_planning_entrypoint.sh"]
CMD ["bash"]
