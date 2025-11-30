+++
title = 'Error when building TensorRT - cuda_runtime.h: no such file or directory'
date = 2019-11-22T14:17:46-03:00
draft = false
tags = ['cuda', 'c++', 'deep-learning', 'cmake']
+++

> This post was originally submitted to [StackOverflow](https://stackoverflow.com/a/58996096/957997) and this [GitHub issue](https://github.com/NVIDIA/nccl/issues/131#issuecomment-557546609).

We were trying to build the [TensorRT SDK](https://developer.nvidia.com/tensorrt) using CMake, but we were receiving the following error:

> error: cuda_runtime.h: No such file or directory

It still wasn't able to find the header files (maybe it is the CMake version that couldn't find the directory `./targets/x86_64-linux/include` or because we have multiple CUDA versions installed in our cluster). Setting `CPATH` and `LD_LIBRARY_PATH` manually to the following fixed the issue for us:

```bash
export CPATH=/usr/local/cuda-10.1/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.1/bin:$PATH
```
