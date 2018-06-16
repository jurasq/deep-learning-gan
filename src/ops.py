import tensorflow as tf
from tflearn import global_avg_pool
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np
import math

he_init = variance_scaling_initializer()
# he_init = tf.truncated_normal_initializer(stddev=0.02)
"""
The weight norm is not implemented at this time.
"""

def weight_norm(x, output_dim) :
    input_dim = int(x.get_shape()[-1])
    g = tf.get_variable('g_scalar', shape=[output_dim], dtype=tf.float32, initializer=tf.ones_initializer())
    w = tf.get_variable('weight', shape=[input_dim, output_dim], dtype=tf.float32, initializer=he_init)
    w_init = tf.nn.l2_normalize(w, dim=0) * g  # SAME dim=1

    return tf.variables_initializer(w_init)

def mlp(name_scope,input_tensor,name_suffix=None,batch_norm=False,relu=False,is_training=True):
    """
    Return a 3 hidden layer perceptron, possibly with a ReLU at the end.
    :param name_scope:   where the variables live
    :param input_tensor: of shape [batch, in_height, in_width, in_channels], e.g. [15 500 4 1]
    """

    name_suffix = name_suffix if name_suffix else ""

    #E.g. batch_size x 500x4x1 for the first input
    input_flatten = reshape(input_tensor,[1,-1])
    input_shape = reshape(input_tensor,[-1]).get_shape().as_list()

    #input_channels = input_shape[-1]

    with tf.name_scope(name_scope):

        weights_shape = [input_shape[0],input_shape[0]]
        weights1 = tf.Variable(tf.random_normal(weights_shape),name='weights'+name_suffix)
        weights2 = tf.Variable(tf.random_normal(weights_shape),name='weights'+name_suffix)
        weightsout =tf.Variable(tf.random_normal(weights_shape),name='weights'+name_suffix)

        bias1 =  tf.Variable(tf.random_normal(input_shape),name='biases'+name_suffix)
        bias2 = tf.Variable(tf.random_normal(input_shape),name='biases'+name_suffix)
        biasout = tf.Variable(tf.random_normal(input_shape),name='biases'+name_suffix)

        #Define a perceptron layer
        layer_1 = tf.add(tf.matmul(input_flatten, weights1), bias1)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights2), bias2)
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weightsout) + biasout



        if relu:
            out_layer = tf.nn.leaky_relu(out_layer, name="relu_"+name_suffix)
        if batch_norm:
            out_layer = tf.contrib.layers.batch_norm(inputs = out_layer, center=True, scale=True, is_training=is_training)

        return out_layer


def conv_layer(name_scope, input_tensor, num_kernels, kernel_shape,
               stride=1, padding="VALID", relu=True, lrelu=False,
               name_suffix=None, batch_norm=False, is_training=True):
    """
    Return a convolution layer, possibly with a ReLU at the end.
    :param name_scope:   where the variables live
    :param input_tensor: of shape [batch, in_height, in_width, in_channels], e.g. [15 500 4 1]
    :param num_kernels:  number of kernels to use for this conv. layer
    :param kernel_shape: the shape of the kernel to use, [height, width]
    """
    name_suffix = name_suffix if name_suffix else ""

    #E.g. batch_size x 500x4x1 for the first input
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]

    with tf.name_scope(name_scope):

        weights_shape = kernel_shape + [input_channels, num_kernels]
        init_vals_weights = tf.truncated_normal(weights_shape, stddev=math.sqrt(2 / float(input_channels)))
        filter_weights = tf.Variable(init_vals_weights, name='weights'+name_suffix)


        biases = tf.Variable(tf.constant(0.1, shape=[num_kernels]), name='biases'+name_suffix)

        #Define a convolutional layer
        layer = tf.nn.conv2d(input_tensor, filter_weights, strides=[1, stride, stride, 1], padding=padding) + biases

        #Add batch normalisation if specified
        if batch_norm:
            layer = tf.contrib.layers.batch_norm(inputs = layer, center=True, scale=True, is_training=is_training)

        #Add (leaky) ReLU if specified
        if relu and lrelu:
            layer = tf.nn.leaky_relu(layer, name="lrelu_"+name_suffix)
        elif relu:
            layer = tf.nn.relu(layer, name="relu_"+name_suffix)

        return layer


def conv_layer_original(x, filter_size, kernel, stride=1, padding='SAME', wn=False, layer_name="conv"):
    with tf.name_scope(layer_name):
        if wn:
            w_init = weight_norm(x, filter_size)

            x = tf.layers.conv2d(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=w_init, strides=stride, padding=padding)
        else :
            x = tf.layers.conv2d(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=he_init, strides=stride, padding=padding)
        return x

def conv_max_forward_reverse(name_scope, input_tensor, num_kernels, kernel_shape,
                             stride=1, padding='VALID', relu=True, lrelu=False, name_suffix = None, batch_norm=False, is_training=True):
    """
    Returns a convolution layer
    """
    name_suffix = name_suffix if name_suffix else ""
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1] # number of input channels

    with tf.name_scope(name_scope):
        shape = kernel_shape + [input_channels, num_kernels]
        initer = tf.truncated_normal(shape, stddev=math.sqrt(2 / float(input_channels)))
        weights = tf.Variable(initer, name='weights')
        num_kernels = weights.get_shape()[3]
        biases = tf.Variable(tf.zeros([num_kernels]), name='biases')

        # If one component of shape is the special value -1, the size of that dimension is computed
        #  so that the total size remains constant.
        # In our case: -1 is inferred to be input_channels * out_channels:
        new_weights_shape = [-1] + kernel_shape + [1]
        w_image = tf.reshape(weights, new_weights_shape)
        tf.summary.image(name_scope + "_weights_im", w_image, weights.get_shape()[3])
        forward_conv = tf.nn.conv2d(input_tensor, weights, strides=[1, stride, stride, 1], padding=padding,
                               name="forward_conv") + biases
        # for reverse complement: reverse in dimension 0 and 1:
        rev_comp_weights = tf.reverse(weights, [0, 1], name="reverse_weights")
        reverse_conv = tf.nn.conv2d(input_tensor, rev_comp_weights,
                                    strides=[1, stride, stride, 1], padding=padding,
                                    name="reverse_conv") + biases
        # takes the maximum between the forward weights and the rev.-comp.-weights:
        max_conv = tf.maximum(forward_conv, reverse_conv, name="conv1")

        # Add batch normalisation if specified
        if batch_norm:
            max_conv = tf.contrib.layers.batch_norm(inputs=max_conv, center=True, scale=True, is_training=is_training)

        if relu and lrelu:
            return tf.nn.leaky_relu(max_conv, name="lrelu_"+name_suffix)
        elif relu:
            return tf.nn.relu(max_conv, name="relu_"+name_suffix)
        else:
            return max_conv


def deconv_layer(x, filter_size, kernel, stride=1, padding='SAME', wn=False, layer_name='deconv'):
    with tf.name_scope(layer_name):
        if wn :
            w_init = weight_norm(x, filter_size)
            x = tf.layers.conv2d_transpose(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=w_init, strides=stride, padding=padding)
        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=he_init, strides=stride, padding=padding)
        return x


def linear(x, unit, wn=False, layer_name='linear'):
    with tf.name_scope(layer_name):
        if wn :
            w_init = weight_norm(x, unit)
            x = tf.layers.dense(inputs=x, units=unit, kernel_initializer=w_init)
        else :
            x = tf.layers.dense(inputs=x, units=unit, kernel_initializer=he_init)
        return x


def nin(x, unit, wn=False, layer_name='nin'):
    # https://github.com/openai/weightnorm/blob/master/tensorflow/nn.py
    with tf.name_scope(layer_name):
        s = list(map(int, x.get_shape()))
        x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
        x = linear(x, unit, wn, layer_name)
        x = tf.reshape(x, s[:-1] + [unit])

        return x


def gaussian_noise_layer(x, std=0.15):
    noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=std, dtype=tf.float32)
    return x + noise


def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')


def max_pool_layer(name_scope, input_tensor, pool_size, strides = None, padding="SAME"):
    """
    Return a max pool layer.
    """
    if not strides:
        strides = [1] + pool_size + [1]

    with tf.name_scope(name_scope):
        layer = tf.nn.max_pool(input_tensor, [1] + pool_size + [1], strides=strides, padding=padding)
        return layer


def max_pooling_original(x, kernel, stride):
    return tf.layers.max_pooling2d(x, pool_size=kernel, strides=stride, padding='VALID')


def flatten(x):
    """
        Returns a flat (one-dimensional) version of the input
    """
    return tf.contrib.layers.flatten(x)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def sigmoid(x):
    return tf.nn.sigmoid(x)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.nn.tanh(x)

def conv_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()

    return concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], axis=3)


def concat(x, axis=1):
    return tf.concat(x, axis=axis)


def reshape(x, shape):
    return tf.reshape(x, shape=shape)


def batch_norm(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)

def instance_norm(x, is_training, scope):
    with tf.variable_scope(scope):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def dropout_layer(name_scope, input_tensor, keep_prob=0.5):
    """
    Return a dropout layer.
    """
    #TODO: is name_scope really needed?
    with tf.name_scope(name_scope):
        return tf.nn.droupout(input_tensor, keep_prob)


def dropout_original(x, rate, is_training):
    return tf.layers.dropout(inputs=x, rate=rate, training=is_training)

def rampup(epoch):
    if epoch < 80:
        p = max(0.0, float(epoch)) / float(80)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def rampdown(epoch):
    if epoch >= (300 - 50):
        ep = (epoch - (300 - 50)) * 0.5
        return math.exp(-(ep * ep) / 50)
    else:
        return 1.0
