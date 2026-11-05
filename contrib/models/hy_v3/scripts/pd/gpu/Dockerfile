# GPU reference image for the Hy3 1P1D comparison (scripts/pd/gpu/gpu_serve.sh):
# vllm/vllm-openai:v0.29.0 (Ubuntu 24.04, CUDA 13, native hy_v3) + the EFA userspace
# (aws-efa-installer 1.51.0, libfabric 2.7.0, as on the p5 host) + GDRCopy 2.6 userspace (the host
# runs gdrdrv 2.6) + Mooncake 0.3.14 (the cuda13 wheel, which includes the EFA transport).
#     docker build -t hy3-efa-029 -f Dockerfile .
FROM vllm/vllm-openai:v0.29.0
ARG EFA_VERSION=1.51.0
ARG GDRCOPY_VERSION=2.6
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl ca-certificates pciutils environment-modules tcl libhwloc-dev build-essential git && \
    cd /tmp && curl -sSLO https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_VERSION}.tar.gz && \
    tar xf aws-efa-installer-${EFA_VERSION}.tar.gz && cd aws-efa-installer && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    cd /tmp && git clone -q --depth 1 -b v${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git && \
    cd gdrcopy && make -j prefix=/usr/local CUDA=/usr/local/cuda lib lib_install && ldconfig && \
    rm -rf /tmp/aws-efa-installer* /tmp/gdrcopy && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip uninstall -y mooncake-transfer-engine && \
    pip install --no-cache-dir "mooncake-transfer-engine-cuda13==0.3.14" && \
    python3 -c "import sysconfig,pathlib; so=pathlib.Path(sysconfig.get_paths()['purelib'])/'mooncake'/'engine.so'; print('EfaTransport in engine.so:', b'EfaTransport' in so.read_bytes())"
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/usr/local/lib:${LD_LIBRARY_PATH}
RUN cat /opt/amazon/efa_installed_packages | head -2 && /opt/amazon/efa/bin/fi_info --version | head -2 && ls /usr/local/lib/libgdrapi*
