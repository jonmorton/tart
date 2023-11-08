ARG BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
FROM ${BASE_IMAGE} as dev-base

# Install some basic utilities
RUN rm -f /etc/apt/sources.list.d/*.list \
    && apt-get update -y && apt-get upgrade -y && apt-get install -y \
    curl \
    ca-certificates \
    build-essential \
    git \
    bzip2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt autoremove -y \
    && apt-get clean -y

FROM dev-base as conda

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"
ENV PYTHONPYCACHEPREFIX /root/.pycache

COPY requirements.txt .
RUN curl -fsSLo ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda update -y conda \
    && /opt/conda/bin/conda install -y python=3.11 numpy conda-build ipython cmake ninja pyyaml \
    && /opt/conda/bin/conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia \
    && /opt/conda/bin/python -mpip install -r requirements.txt \
    && MAX_JOBS=4 /opt/conda/bin/python -mpip install flash-attn --no-build-isolation \
    && /opt/conda/bin/python -mpip install -U xformers \
    #&& /opt/conda/bin/python -mpip install -v -U git+https://github.com/NVIDIA/apex.git@master#egg=apex --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
    #&& /opt/conda/bin/python -mpip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers \
    && /opt/conda/bin/conda clean -ya \
    && /opt/conda/bin/python -mpip cache purge

FROM ${BASE_IMAGE} as dev
COPY --from=conda /opt/conda /opt/conda

RUN apt-get update -y && \apt-get upgrade -y && apt-get install -y \
    curl \
    ca-certificates \
    build-essential \
    git \
    bzip2 \
    zstd \
    libzstd-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt autoremove -y \
    && apt-get clean -y

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH /opt/conda/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"
ENV LIBRARY_PATH /usr/local/cuda/lib64
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64
ENV PYTHONPYCACHEPREFIX /root/.pycache

WORKDIR /workspace

# Set the default command to python3
CMD ["python3"]
