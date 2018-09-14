# tensorflow-builds
Tensorflow binary builds for some platforms

Compilation
===========

These packages were compiled using standard tensorflow [compilation
                                                        guidelines](https://www.tensorflow.org/install/install_sources). 
Compilation was carried out using -mnative flags.

`git clone https://github.com/tensorflow/tensorflow`

`cd tensorflow`

`git checkout r1.10`

`./configure`

If compiling for a CPU-based system:

`bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`

If compiling for a GPU-CUDA-based system:

`bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package`

`bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg`

**Configuration**:
./configure was run with support for:
- Google Cloud Compute, Amazon AWS, Hadoop File System support.

GPU Support
===========
Binaries are build using CUDA Toolkit 9.2
Binaries with NVidia GPU support are built with NCCL (version 2.2)  and TensorRT support.

To compile with both, you need binaries from NVidia, installed using apt-get. From there:

`sudo apt-get install tensorrt libnccl2 libnccl-dev`

`cd /usr/local/cuda-9.2`

`sudo ln -s /usr/include/nccl.h include`

`sudo mkdir lib`

`sudo ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 lib`

Binaries:
=========
https://www.dropbox.com/sh/f40eb6xsioj74il/AADHVj0hDxxo0yyv43Myvg65a?dl=0

Supported platforms:
====================

- MacOS - 10.12, no GPU, Python 3.6
- Linux x86-64:
  - Ubuntu 16.04, no GPU, Python 3.5 
  - Ubuntu 16.04, CUDA 9.2, Python3.5
  - CentOS 7.3, no GPU, Python3.6
  
Benchmarking:
==============

There are several benchmarking options. One derived from [here](https://github.com/tobigithub/tensorflow-deep-learning/wiki/tf-benchmarks) is provided. To run:

`python3 benchmark.py`  
