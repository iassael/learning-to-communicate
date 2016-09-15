FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
# FROM ubuntu:16.04
MAINTAINER Yannis Assael, Jakob Foerster

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler && \
    curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    rm -rf /var/lib/apt/lists/*

# Clone torch (and package) repos:
RUN mkdir -p /opt && git clone https://github.com/torch/distro.git /opt/torch --recursive

# Run installation script
RUN cd /opt/torch && ./install.sh -b

# Export environment variables manually
ENV TORCH_DIR /opt/torch/pkg/torch/build/cmake-exports/
ENV LUA_PATH '/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/opt/torch/install/share/lua/5.1/?.lua;/opt/torch/install/share/lua/5.1/?/init.lua;./?.lua;/opt/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH '/root/.luarocks/lib/lua/5.1/?.so;/opt/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH /opt/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH /opt/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH /opt/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH '/opt/torch/install/lib/?.so;'$LUA_CPATH

# Install torch packages
RUN luarocks install totem && \
    luarocks install https://raw.githubusercontent.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec && \
    luarocks install unsup && \
    luarocks install csvigo && \
    luarocks install loadcaffe && \
    luarocks install classic && \
    luarocks install pprint && \
    luarocks install class && \
    luarocks install image && \
    luarocks install mnist && \
    luarocks install https://raw.githubusercontent.com/deepmind/torch-distributions/master/distributions-0-0.rockspec

# Cleanup
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /project
