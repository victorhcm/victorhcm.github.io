+++
title = 'RuntimeError with copy_() and expand() - Unsupported operation: more than one element of the written-to tensor refers to a single memory location'
date = 2020-01-25T00:18:15-03:00
draft = false
+++
This answer was originally posted on [StackOverflow](https://stackoverflow.com/q/59905234/957997).

I'm migrating a repository from Pytorch Nightly 1.0.0 to 1.3.1. Stripping the unnecessary details, it is basically performing the following sequence of operations:

```python
mu = torch.tensor(0.005)
bar = torch.eye(5, 5)
foo = torch.eye(5).expand(5, 5, 5)

# update
bar.copy_(mu * bar)  # ok!
foo.copy_(mu * foo)  # error
```

`bar.copy_(mu * bar)` works, while when I try to `foo.copy_()` the result, it gives the following error:

> RuntimeError: unsupported operation: more than one element of the written-to tensor refers to a single memory location. Please clone() the tensor before performing the operation.

This is happening because `expand()` only creates a new view on the existing tensor, thus it doesn't allocate the full memory necessary to receive all the elements from the operation `mu * foo`, which has more elements than the original tensor `foo`. You can fix it by either using `expand().clone()` or `repeat()`, which will give you the full tensor.

```python
foo = torch.eye(5).expand(5, 5, 5).clone()  # clone gives the full tensor
foo.copy_(mu * foo)  # ok!
```

[albanD][1] suggests that doing `expand().clone()` might still be faster than `repeat()`.

See [here][2] and [here][3] for more details about `expand()` and `repeat()`.

[1]: https://discuss.pytorch.org/t/torch-repeat-and-torch-expand-which-to-use/27969/4
[2]: https://discuss.pytorch.org/t/expand-vs-repeat-semantic-difference/59789
[3]: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.expand
