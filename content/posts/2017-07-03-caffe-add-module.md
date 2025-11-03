+++
title = 'How to add a new module in Caffe'
date = 2017-07-03T08:52:43-03:00
draft = false
+++

# yjxiong/caffe - How to Use 

These are some notes on how to use the yjxiong/caffe fork for video recognition and how to include your own custom module.

## Compiling for training with support to OpenMPI

- Update cuDNN to version 5 because of BatchNorm and winograd (speeds up training)
- Download OpenMPI. Use either version 1.8.7 or 1.8.10 because it [was the reported one to work](https://github.com/yjxiong/caffe/issues/68#issuecomment-209371404). I'm using 1.8.7 just in case.
- Compile OpenMPI:
    ```
    cd openmpi-1.8.7/
    ./configure --with-cuda  # by default, --prefix=/usr/local, therefore, does not need to set it
    make all install
    ldconfig  # maybe you'll have to set a /etc/ld.so.conf.d/<file>, I'm not sure with /usr/local/{lib,bin} are in ld library path by default
    ldconfig -p | grep mpi  # sanity check
    ``` 
- Compile yjxiong-caffe with support to OpenMPI:

    ```
    mkdir build  # It was reported that maybe this folder should be [cmake_build](https://github.com/yjxiong/caffe/issues/42#issuecomment-180465148) because it is hardcoded somewhere
    cd build/
    cmake .. -DUSE_MPI=ON -DCUDNN_INCLUDE=/usr/local/cuda/include
    ```
- Do not set `-DMPI_CXX_COMPILER=<the path to mpicxx binary>` as suggested by yjxiong. When I did, some OMPI libs/bins were not found by cmake
- I had to set `-DCUDNN_INCLUDE` because the default was pointing to `/usr/local/include`, instead to my CUDA/include folder.

Note: This excerpt was extracted from my other repo accio/Troubleshooting.md

## Steps for training

1. Define the solver `visual_rhythm_solver.prototxt`. Some parameters:

    - `test_iter`: initial iteration to test on validation
    - `test_interval`: perform tests every `x` iterations
    - `stepsize`: step to update the learning rate

2. Define training file `visual_rhythm_train_val_fast.prototxt`. You'll need to update `video_data_param`:

    - `source`: txt file comprising path to each folder with the extracted frames in the following format `'{folder} {frames} {label}'`. `{label}` starts from 0.
    - `batch_size`: size of the batch
    - `new_width`/`new_height`: width/height to rescale the input. I'm setting it to the image shape even though they are correct because if I don't set it, it tries to rescale the images to 226x226

And also update `transform_param`:
    - disable `crop_size`: crops a patch from the image
    - disable `fix_crop`: crops a patch on a fixed corner
    - disable `multi_scale`: uses multiple scales (requires to define `max_distort` to avoid to much image distortion; and `scale_ratios`, a scaling factors list)
    - enable `mirror`: d'oh
    - enable `mean_value`: `[104, 117, 123]` this creates an image with three channels matching the input shape

And finally change the other data layer for the validation partition. Much the same as before, but without some of the transformations.

3. Reuse weights from other models as preinitialization.

If you want to use weights from another network, you will have to reshape the number of units in each conv layer to match the input (due to the fully connected layer) and also reshape the input tensors. Script `FIXME` demonstrates how to do that.

4. Run `examples/action_visual_rhythm/train_action_recognition_rgb.sh`.


## Image format

You can see their definition on `video_data_layer.cpp`:

* RGB: `image_%04d.jpg`
* OF: `flow_%c_%04d.jpg`

If you have any questions, these are a few useful files to check:

```
src/caffe/layers/video_data_layers.cpp
src/caffe/util/io.cpp
src/caffe/data_transformer.cpp
```
