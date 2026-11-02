# syntax=docker/dockerfile:1
ARG STREET_CUDA_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04
ARG DRIVEGS_TRAIN_IMAGE

FROM ${STREET_CUDA_IMAGE} AS streetcrafter
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl ffmpeg git \
        libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglib2.0-0 ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL --retry 3 \
        https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-25.3.1-0-Linux-x86_64.sh \
        -o /tmp/miniforge.sh \
    && echo '376b160ed8130820db0ab0f3826ac1fc85923647f75c1b8231166e3d559ab768  /tmp/miniforge.sh' | sha256sum -c - \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda create -y --override-channels -c conda-forge \
        -p /opt/conda/envs/streetcrafter python=3.9.23 pip \
    && /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/envs/streetcrafter/bin:/usr/local/cuda-12.1/bin:/usr/bin:/bin \
    CUDA_HOME=/usr/local/cuda-12.1 \
    LD_LIBRARY_PATH=/opt/conda/envs/streetcrafter/lib:/usr/local/cuda-12.1/lib64 \
    LIBRARY_PATH=/usr/local/cuda-12.1/lib64/stubs \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    CUDAHOSTCXX=/usr/bin/g++ \
    TORCH_CUDA_ARCH_LIST=8.0 \
    TORCH_EXTENSIONS_DIR=/opt/streetcrafter-kernels \
    FORCE_CUDA=1 \
    MAX_JOBS=4 \
    PIP_CONFIG_FILE=/dev/null \
    PIP_CONSTRAINT=/opt/streetcrafter/constraints.txt \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG PYPI_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PYPI_INDEX_URL}
COPY docker/joint/constraints.streetcrafter.txt /opt/streetcrafter/constraints.txt
COPY docker/joint/requirements.streetcrafter.txt docker/joint/requirements.git.txt /opt/streetcrafter/
RUN python -m pip install setuptools==75.8.2 wheel ninja \
    && python -m pip install --no-deps \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && python -m pip install \
        torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 \
        -r /opt/streetcrafter/requirements.streetcrafter.txt \
    && python -m pip install --no-build-isolation -r /opt/streetcrafter/requirements.git.txt

COPY submodules/sdata /tmp/submodules/sdata
COPY submodules/simple-knn /tmp/submodules/simple-knn
RUN python -m pip install --no-build-isolation /tmp/submodules/sdata /tmp/submodules/simple-knn \
    && rm -rf /tmp/submodules \
    && python -m pip check

COPY docker/joint/precompile_street.py /opt/streetcrafter/precompile_street.py
RUN ln -sf libcuda.so /usr/local/cuda-12.1/lib64/stubs/libcuda.so.1 \
    && LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64/stubs:${LD_LIBRARY_PATH}" \
       python /opt/streetcrafter/precompile_street.py

WORKDIR /workspace/street_crafter
COPY . .
COPY --chmod=755 docker/joint/streetcrafter-python /usr/local/bin/streetcrafter-python
RUN mkdir -p .venv/bin \
    && ln -s /usr/local/bin/streetcrafter-python .venv/bin/python \
    && /usr/local/bin/streetcrafter-python docker/joint/validate_joint.py --street-only

FROM ${DRIVEGS_TRAIN_IMAGE} AS joint
RUN export PATH=/usr/sbin:/usr/bin:/sbin:/bin \
    && unset LD_LIBRARY_PATH \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=streetcrafter /opt/conda/envs/streetcrafter /opt/conda/envs/streetcrafter
COPY --from=streetcrafter /usr/local/cuda-12.1 /usr/local/cuda-12.1
COPY --from=streetcrafter /opt/streetcrafter /opt/streetcrafter
COPY --from=streetcrafter /opt/streetcrafter-kernels /opt/streetcrafter-kernels
COPY --from=streetcrafter /workspace/street_crafter /workspace/street_crafter
COPY --chmod=755 docker/joint/streetcrafter-python docker/joint/drivegs-python docker/joint/joint-distill /usr/local/bin/

COPY source-metadata.json /opt/joint/source-metadata.json
RUN python /workspace/street_crafter/docker/joint/configure_joint.py \
    && python /workspace/street_crafter/docker/joint/validate_joint.py --joint

ARG STREETCRAFTER_COMMIT=unknown
ARG DRIVEGS_COMMIT=unknown
ARG SOURCE_FINGERPRINT=unknown
LABEL org.opencontainers.image.title="DriveGS + StreetCrafter A100 (Miniforge)" \
      io.joint.streetcrafter.commit=${STREETCRAFTER_COMMIT} \
      io.joint.drivegs.commit=${DRIVEGS_COMMIT} \
      io.joint.source.fingerprint=${SOURCE_FINGERPRINT}
WORKDIR /workspace/drivegs
CMD ["bash"]
