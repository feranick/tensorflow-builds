# tensorflow-builds
Tensorflow binary builds for some platforms

Compilation
===========

These packages were compiled using standard tensorflow [compilation
                                                        guidelines](https://www.tensorflow.org/install/install_sources). 
Compilation was carried out using -mnative flags.

`git clone https://github.com/tensorflow/tensorflow`

`cd tensorflow`

`git checkout r1.3`

`./configure`

If compiling for a CPU-based system:

`bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`

If compiling for a GPU-CUDA-based system:

`bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package`

`bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg`

**Configuration**:
./configure was run with support for:
- Google Cloud Compute support (TensorFlow 1.2+).

Binaries:
=========
https://www.dropbox.com/sh/f40eb6xsioj74il/AADHVj0hDxxo0yyv43Myvg65a?dl=0

Supported platforms:
====================

- MacOS - 10.12, no GPU, Python 3.6
- Linux x86-64:
  - Ubuntu 16.04, no GPU, Python 3.5 
  - Ubuntu 16.04, CUDA 8.0, Python3.5
  - CentOS 7.3, no GPU, Python3.6
  
Benchmarking:
==============

There are several benchmarking options. One derived from [here](https://github.com/tobigithub/tensorflow-deep-learning/wiki/tf-benchmarks) is provided. To run:

`python3 benchmark.py`  
  
Known issues:
=============
Compilation is broken for tensorflow-gpu using bazel 0.5.3 (fixed in tensorflow 1.4+). To fix it, either use bazel 0.5.2
or change in: 

`tensorflow/third_party/gpus/cuda_configure.bzl`

line 109 from:

`return [repository_ctx.path(_cxx_inc_convert(p))`

to:
`return [str(repository_ctx.path(_cxx_inc_convert(p)))`
