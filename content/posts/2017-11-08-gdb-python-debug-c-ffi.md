+++
title = 'Debugging a C / C++ module in Pytorch with GDB'
date = 2017-11-08T22:51:39-03:00
draft = false
+++

I'm testing with a modified `conv2d` with deformable convolutions implemented as
a C extension and I was having trouble debugging it. I found one approach which
is helpful if you are debugging a module built with `torch.utils.ffi` (or
`cffi`). 

First, enable debugging symbols (`-g`) and set the `-O0` flag, as it will
override the default `-O3` from `distutils.unixccompiler.py`. In my case, it was:

```bash
CC=g++ CFLAGS="-O0 -g" python build.py
```

Then, execute:

```bash
gdb python
```

At the `gdb` prompt, run your script:

```
(gdb) run test_module.py
```

I’m not using python-dbg, so I interrupt the execution with `Ctrl+c`, then I set a breakpoint at the C++ function I want to debug:

```
(gdb) b my_conv2d
```

Then, hit `c` to (c)ontinue.


# Improvements

I posted this answer on [Pytorch Discuss](https://discuss.pytorch.org/t/how-to-use-pdb-or-gdb-debug-from-python-into-c-c-code/8152/6) and [Derek Kim](https://discuss.pytorch.org/u/derek_kim/summary) suggested a more [graceful approach](https://discuss.pytorch.org/t/how-to-use-pdb-or-gdb-debug-from-python-into-c-c-code/8152/10) using `os.kill` and `SIGTRAP`:

```python
# a.py

def breakpoint():
    import os, signal
    os.kill(os.getpid(), signal.SIGTRAP)

# your code...
breakpoint()  # set a breakpoint
# your code...
```

Then, run the usual `gdb python` and hit `(c)ontinue`.
