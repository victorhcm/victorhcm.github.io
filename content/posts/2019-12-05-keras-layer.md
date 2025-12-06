+++
title = 'How to add/remove layers in Keras as of TensorFlow 2.0'
date = 2019-12-12T12:50:29-03:00
draft = false
tags = ["keras", "tensorflow"]
+++

> This was originally posted as a [StackOverflow answer][2].

We're working with a pre-trained Keras model and trying to modify the
architecture. As of Keras 2.3.1 and TensorFlow 2.0, `model.layers.pop()` is not
working as intended anymore (see issue [here][1]). They suggested two options
to do this.

## Recreate and copy

One option is to recreate the model and copy the layers. For instance, if you
want to remove the last layer and add another one, you can do:

```python
model = Sequential()
for layer in source_model.layers[:-1]: # go through until last layer
    model.add(layer)
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

## Functional model

Another option is to use the functional model:

```python
    predictions = Dense(3, activation='softmax')(source_model.layers[-2].output)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
```

`model.layers[-1].output` means the last layer's output which is the final
output, so in your code, you actually didn't remove any layers, you added
another head/path.

[1]: https://github.com/tensorflow/tensorflow/issues/22479#issuecomment-424577448
[2]: https://stackoverflow.com/a/59304656/957997
