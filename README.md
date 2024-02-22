# tensorflow-builds
Tensorflow building information and binaries for selected platforms. 

**Note**: These binaries are provided as is. They may not work on your systems as they are highly customized during compilation.

**Use at your own risk**. No warranty or guarantee is provided.

## Compilation

These packages were compiled using standard tensorflow [compilation                                                        guidelines](https://www.tensorflow.org/install/install_sources). 

Prerequiresites and installed packages:
```
sudo apt install libstdc++-12-dev:amd64 python3-pip python3-wheel python3-requests patchelf python-is-python3
sudo pip3 install pip numpy packaging opt_einsum grpcio flatbuf
sudo pip3 install keras_preprocessing --no-deps
```

Compilation was carried out using -mnative flags.

`git clone https://github.com/tensorflow/tensorflow`

`cd tensorflow`

`git checkout v2.15.0`

`./configure`

If compiling for a CPU-based system:

`bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`

If compiling for a GPU-CUDA-based system:

`bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package`

`bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg`

**Configuration**:
./configure was run with support for:
- Google Cloud Compute, Amazon AWS, Hadoop File System support.

### Optimizations
For additional optimizations, build with `native` support can be enabled by adding the folllowing to `.bazelrc` after running the above `./configure` step

```
build:linux --copt=-march=native
```
or (for MacOS):
```
build:macos --copt=-march=native
```

## GPU Support
Binaries are build using CUDA Toolkit 12.2. [You may need to activate NVidia develoepr repos](https://developer.nvidia.com/cuda-toolkit)
Binaries with NVidia GPU support TensorRT support (v.8.9) and libnccl2.

To compile with both, you need binaries from NVidia, installed using apt-get. From there:

`sudo apt-get install tensorrt libnccl2 libnccl-dev, tensorrt`

`cd /usr/local/cuda-12.2`

`sudo ln -s /usr/include/nccl.h include`

`sudo mkdir lib`

`sudo ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 lib`

## Binaries:
https://www.dropbox.com/sh/f40eb6xsioj74il/AADHVj0hDxxo0yyv43Myvg65a?dl=0

## Supported platforms:
 
Currently supported platforms
- MacOS - 10.12, no GPU, Python 3.7
- Linux x86-64:
  - Ubuntu 18.04, no GPU, Python 3.6 
  - Ubuntu 18.04, CUDA 10.1, Python3.6
  - CentOS 7.3, no GPU, Python3.6
  
## Benchmarking:

There are several benchmarking options. One derived from [here](https://github.com/tobigithub/tensorflow-deep-learning/wiki/tf-benchmarks) is provided. To run:

`python3 benchmark.py`  

## Known Issues:
1. As of version `2.15.0`, Tensorflow as moved to `clang` as the main compiler. Unfortunately compilation is broken in Ubuntu with GPU support, so the use of `nvcc` is recommended. 
2. The linker provided with XCode may be lead to a crash when compiling for MacOS. In case of a compilation crash, a "classic" linker can be invoked by adding the following line to `tensorflow/tensorflow.bzl`:
```

    def pywrap_tensorflow_macro_opensource(
         clean_dep("//tensorflow:macos"): [
             # TODO: the -w suppresses a wall of harmless warnings about hidden typeinfo symbols
             # not being exported.  There should be a better way to deal with this.
+            "-ld_classic",
             "-Wl,-w",
             "-Wl,-exported_symbols_list,$(location %s.lds)" % vscriptname,
         ], 
```
