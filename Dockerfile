FROM wilbrandt/cuda_dev_docker:9.1
LABEL maintainer="Robert Wilbrandt <robert@stamm-wilbrandt.de>"
LABEL description="Build environment for the gpu_planning project"

RUN apt install -y libboost-log-dev libboost-regex-dev libboost-program-options-dev

COPY ./docker_entrypoint.sh /gpu_planning_entrypoint.sh
ENTRYPOINT ["/gpu_planning_entrypoint.sh"]
CMD ["bash"]
