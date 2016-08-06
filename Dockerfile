FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
# FROM ubuntu:14.04
MAINTAINER Yannis Assael, Jakob Foerster

# CUDA includes
ENV CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda
ENV CPPFLAGS "-I/usr/local/cuda/include $CPPFLAGS"
ENV CFLAGS "-I/usr/local/cuda/include $CFLAGS"
ENV CMAKE_INCLUDE_PATH /usr/local/cuda/include:$CMAKE_INCLUDE_PATH

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y nano vim man build-essential

# Install curl and dependencies for torch
RUN apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler nano

# Clone torch (and package) repos:
RUN curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash && \
    mkdir -p /opt && git clone https://github.com/torch/distro.git /opt/torch --recursive

# Patch: generate code for 750 Ti (sm_50), Titan Black / 780 Ti (sm_35), Titan X (sm_52), and ONLY these:
# This makes sure we generate GPU-specific code to prevent the 60-second JIT compile for new containers.
# RUN cd /opt/torch && \
#     sed -i 's/"-arch=sm_20"/"-gencode arch=compute_50,code=sm_50 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52"/' \
#         extra/cutorch/lib/THC/CMakeLists.txt \
#         extra/cunn/CMakeLists.txt \
#         extra/cunnx/CMakeLists.txt

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
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    rm -rf /tmp/var/*

WORKDIR /project
