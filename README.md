# tensorflow-builds
Tensorflow binary builds for some platforms

Compilation
===========

These packages were compiled using standard tensorflow [compilation
                                                        guidelines](https://www.tensorflow.org/install/install_sources). 
Compilation was carried out using -mnative flags.

git clone https://github.com/tensorflow/tensorflow

cd tensorflow

git checkout r1.2

./configure

bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

**Configuration**:
./configure was run with support for:
- Google Cloud Compute support (TensorFlow 1.2+).

Binaries:
=========
https://www.dropbox.com/sh/f40eb6xsioj74il/AADHVj0hDxxo0yyv43Myvg65a?dl=0

Supported platforms:
====================

- MacOS - Sierra, no GPU, Python 3.6
- Linux x86-64 - Ubuntu 16.04, no GPU, Python 3.5 
