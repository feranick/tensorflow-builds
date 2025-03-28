# tensorflow-builds
Tensorflow building information and binaries for selected platforms. 

**Note**: These binaries are provided as is. They may not work on your systems as they are highly customized during compilation.

**Use at your own risk**. No warranty or guarantee is provided.

# Compilation

## Pre-requisites: Docker builds
Linux distributions may come with outdated version of compilers and libraries that are not compatible with the most recent version of TF. The easiest way to compile it is to use docker. Run this command to get a docker image on a bash, from where you can then proceed with the regular compilation:

```
docker run --rm -it --entrypoint bash tensorflow/build:2.17-python3.10
```

Choose the image you need from here: https://hub.docker.com/r/tensorflow/build/tags

You can transfer the binaries from the docker image from your host using:
```
docker cp <containerId>:/tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house /host/path/target`
```

where the `<containerID>` may be found, while the container is running, with:
```
docker ps -a
```

Once completed, you can purge all docker images with:
```
docker system prune -a
```

Alternatively, you can install `openssh-client` to enable `scp` and therefore transfer from the docker image to the host:
```
sudo apt update; sudo apt install openssh-client
cp bazel-bin/tensorflow/tools/pip_package/wheel_house/* user@host.com
```

## Pre-requisites: Native builds

These packages were compiled using standard tensorflow [compilation                                                        guidelines](https://www.tensorflow.org/install/install_sources). 

Prerequiresites and installed packages:
```
sudo apt install libstdc++-12-dev:amd64 python3-pip python3-wheel python3-requests patchelf python-is-python3 lld
sudo pip3 install pip numpy packaging opt_einsum grpcio flatbuf
sudo pip3 install keras_preprocessing --no-deps
```
Get `tensorflow` source and desired version:
```
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout v2.17.0
```

## Configure

```
./configure
```

If GPU support is enabled (via CUDA), do not specify any local or remote path/version, just press ENTER. 

### Compilation Flags and optimization
During compilations, additional can be added when asked, and will be included through the flag `--config=opt`. These are recommended:

```
-Wno-sign-compare -Wno-error=unused-command-line-argument -Wno-gnu-offsetof-extensions -O3 -march=native
```

## Compilation
CPU-based system, Python 3.12, TF 2.18.0 or newer:
```
export TF_PYTHON_VERSION=3.12; bazel build --config=opt //tensorflow/tools/pip_package:wheel --repo_env=WHEEL_NAME=tensorflow
```    
CPU-based system, TF < 2.17.0: 
```
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
```
GPU-based system, Python 3.12, TF 2.18.0 or newer
```
export TF_PYTHON_VERSION=3.12; bazel build --config=opt --config=cuda_wheel //tensorflow/tools/pip_package:wheel --repo_env=WHEEL_NAME=tensorflow
```    
GPU-based system, Python 3.12, TF 2.17.0
```
export TF_PYTHON_VERSION=3.12; bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:wheel --repo_env=WHEEL_NAME=tensorflow
```    
GPU-based system, TF < 2.17.0
```
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
```

Wheel packages will be located here:
```
/tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house
```

ONLY for versions of TF < 2.17.0, to create a wheel package, issue the following command. 
```
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```
# GPU Support
Binaries are build using:
- CUDA Toolkit 12.5
- cuDNN 9
- TensorRT 8.6.1
 
[You may need to activate NVidia develoepr repos](https://developer.nvidia.com/cuda-toolkit)
To compile with the correct libraries, you need binaries from NVidia and these following commands:
```
sudo apt-get install tensorrt libnccl2 libnccl-dev, tensorrt
cd /usr/local/cuda-12.5
sudo ln -s /usr/include/nccl.h include
sudo mkdir lib
sudo ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 lib
```

# Additional Optimization flags
You can check the CPU optimization flags supported by your CP by running:
```
# Linux
cat /proc/cpuinfo | grep flags

# macOS
sysctl -a | grep "machdep.cpu.*features:"
```
You can selectively target some of the flags that may be available:
```
-mavx -mavx2 -mfma -msse4.2
-msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma
-mavx512f -mavx512vnni
```

# Binaries:
https://www.dropbox.com/sh/f40eb6xsioj74il/AADHVj0hDxxo0yyv43Myvg65a?dl=0

# Supported platforms:
 
Currently supported platforms
- MacOS - 15.0, no GPU, Python 3.12
- Linux x86-64:
  - Ubuntu 24.04, no GPU, Python 3.12
  - Ubuntu 24.04, CUDA 12.5, Python3.12
  
## Benchmarking:

There are several benchmarking options. One derived from [here](https://github.com/tobigithub/tensorflow-deep-learning/wiki/tf-benchmarks) is provided. To run:

##
    python3 benchmark.py  

## Known Issues:
1. If compiling on older NVidia hardware, when running TensorFlow v2.18.0 or higher, a crash may occur becuase the system may not be able to find `libdevice`. Add the following to your `.bashrc` or `.profile`

    ```
    export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-12.5
    ```
