import tensorflow as tf
import numpy as np
# todo: variable scope and reuse


def conv2d(x, filter_shape, name, kernel_initializer=tf.initializers.truncated_normal, **kwargs):
    # todo: bias
    w = tf.get_variable(name, shape=filter_shape, initializer=kernel_initializer)
    return tf.nn.conv2d(x, w, **kwargs)


def flatten(x):
    return tf.reshape(x, tf.stack([-1, np.prod([x.value for x in x.shape[1:]])]))


def fully_connected(x, name, output_size, use_bias=True, kernel_initializer=tf.initializers.glorot_uniform,
                    bias_initializer=tf.initializers.zeros):
    input_size = x.shape[1].value

    kernel_shape = None
    if callable(kernel_initializer):
        kernel_shape = (input_size, output_size)
    w = tf.get_variable(name=name+"/w", shape=kernel_shape, initializer=kernel_initializer)

    if use_bias:
        bias_shape = None
        if callable(bias_initializer):
            bias_shape = (output_size, )
        b = tf.get_variable(name=name+"/b", shape=bias_shape, initializer=bias_initializer)
        return tf.matmul(x, w) + b
    return tf.matmul(x, w)
