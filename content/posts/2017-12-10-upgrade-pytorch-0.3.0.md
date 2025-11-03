+++
title = 'Error when ugprading Pytorch to 0.3.0'
date = 2017-12-10T10:01:53-03:00
draft = false
+++

> This answer was originally posted on [StackOverflow](https://stackoverflow.com/q/47734702/957997).

I tried to update PyTorch to the recently released 0.3.0, 

```bash
conda install pytorch=0.3.0 torchvision -c pytorch
```

But I was getting the following error:

>     NoPackagesFoundError: Dependency missing in current linux-64 channels:
>      - pytorch 0.3.0* -> mkl >=2018

The reason was because `mkl` was coming from the `defaults` channel. To update it to `2018`, I installed it from the `anaconda` channel:

```bash
    conda install mkl -c anaconda
```
