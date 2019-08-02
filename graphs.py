import tensorflow as tf

# set trainable variable initialization routines
KERNEL_INIT = tf.keras.initializers.he_uniform()
WEIGHT_INIT = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BIAS_INIT = tf.constant_initializer(0.0)


def convolution_layer(x, kernel_size, num_out_channels, activation, batch_norm, is_training, name):
    """
    :param x: input data
    :param kernel_size: convolution kernel size
    :param num_out_channels: number of output channels
    :param activation: non-linearity
    :param batch_norm: whether to use batch norm
    :param is_training: whether we are training or testing (used by batch normalization)
    :param name: variable scope name (empowers variable reuse)
    :return: layer output
    """
    # run convolution layer
    x = tf.layers.conv2d(inputs=x,
                         filters=num_out_channels,
                         kernel_size=[kernel_size] * 2,
                         strides=[1, 1],
                         padding='same',
                         activation=None,
                         use_bias=True,
                         kernel_initializer=KERNEL_INIT,
                         bias_initializer=BIAS_INIT,
                         name=name)

    # run batch norm if specified
    if batch_norm:
        x = tf.contrib.layers.batch_norm(inputs=x, is_training=is_training, scope=name)

    # run activation
    x = activation(x)

    return x


def max_pooling_layer(x, pool_size, strides, name):
    """
    :param x: input data
    :param pool_size: pooling kernel size
    :param strides: pooling stride length
    :param name: variable scope name (empowers variable reuse)
    :return: layer output
    """
    # run max pooling
    x = tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=strides, padding='same', name=name)

    return x


def fully_connected_layer(x, num_outputs, activation, is_training, name):
    """
    :param x: input data
    :param num_outputs: number of outputs
    :param activation: non-linearity
    :param is_training: whether we are training or testing (used by batch normalization)
    :param name: variable scope name (empowers variable reuse)
    :return: layer output
    """
    # run dense layer
    x = tf.layers.dense(inputs=x,
                        units=num_outputs,
                        activation=None,
                        use_bias=True,
                        kernel_initializer=WEIGHT_INIT,
                        bias_initializer=BIAS_INIT,
                        name=name)

    # run batch norm
    x = tf.contrib.layers.batch_norm(inputs=x, activation_fn=activation, is_training=is_training)

    return x


class IICGraph(object):
    def __init__(self, config='B', batch_norm=True, fan_out_init=64):
        """
        :param config: character {A, B, C} that matches architecture in IIC supplementary materials
        :param fan_out_init: initial fan out (paper uses 64, but can be reduced for memory constrained systems)
        """
        # set activation
        self.activation = tf.nn.relu

        # save architectural details
        self.config = config
        self.batch_norm = batch_norm
        self.fan_out_init = fan_out_init

    def __architecture_b(self, x, is_training):
        """
        :param x: input data
        :param is_training: whether we are training or testing (used by batch normalization)
        :return: output of VGG
        """
        with tf.compat.v1.variable_scope('GraphB', reuse=tf.compat.v1.AUTO_REUSE):

            # layer 1
            num_out_channels = self.fan_out_init
            x = convolution_layer(x=x, kernel_size=5, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv1')
            x = max_pooling_layer(x=x, pool_size=2, strides=2, name='pool1')

            # layer 2
            num_out_channels *= 2
            x = convolution_layer(x=x, kernel_size=5, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv2')
            x = max_pooling_layer(x=x, pool_size=2, strides=2, name='pool2')

            # layer 3
            num_out_channels *= 2
            x = convolution_layer(x=x, kernel_size=5, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv3')
            x = max_pooling_layer(x=x, pool_size=2, strides=2, name='pool3')

            # layer 4
            num_out_channels *= 2
            x = convolution_layer(x=x, kernel_size=5, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv4')

            # flatten
            x = tf.contrib.layers.flatten(x)

            return x

    def evaluate(self, x, is_training):
        """
        :param x: input data
        :param is_training: whether we are training or testing (used by batch normalization)
        :return: output of graph
        """
        # run corresponding architecture
        if self.config == 'B':
            return self.__architecture_b(x, is_training)
        else:
            raise Exception('Unknown graph configuration!')


class VGG(object):
    def __init__(self, config='A', batch_norm=True, fan_out_init=64):
        """
        :param config: character {A, C, D} that matches architecture in VGG paper
        :param fan_out_init: initial fan out (paper uses 64, but can be reduced for memory constrained systems)
        """
        # set activation
        self.activation = tf.nn.relu

        # save architectural details
        self.config = config
        self.batch_norm = batch_norm
        self.fan_out_init = fan_out_init

    def __vgg_a(self, x, is_training):
        """
        :param x: input data
        :param is_training: whether we are training or testing (used by batch normalization)
        :return: output of VGG
        """
        with tf.compat.v1.variable_scope('VGG_A', reuse=tf.compat.v1.AUTO_REUSE):

            # layer 1
            num_out_channels = self.fan_out_init
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv1_1')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool1')

            # layer 2
            num_out_channels *= 2
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv2_1')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool2')

            # layer 3
            num_out_channels *= 2
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv3_1')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv3_2')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool3')

            # layer 4
            num_out_channels *= 2
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv4_1')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv4_2')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool4')

            # layer 5
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv5_1')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv5_2')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool5')

            # flatten
            x = tf.contrib.layers.flatten(x)

            # fully connected layers
            x = fully_connected_layer(x=x,
                                      num_outputs=4096,
                                      activation=self.activation,
                                      is_training=is_training,
                                      name='fc1')
            x = fully_connected_layer(x=x,
                                      num_outputs=4096,
                                      activation=self.activation,
                                      is_training=is_training,
                                      name='fc2')
            x = fully_connected_layer(x=x,
                                      num_outputs=4096,
                                      activation=self.activation,
                                      is_training=is_training,
                                      name='fc3')

            return x

    def __vgg_c(self, x, is_training):
        """
        :param x: input data
        :param is_training: whether we are training or testing (used by batch normalization)
        :return: output of VGG
        """
        with tf.compat.v1.variable_scope('VGG_C', reuse=tf.compat.v1.AUTO_REUSE):
            # layer 1
            num_out_channels = self.fan_out_init
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv1_1')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv1_2')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool1')

            # layer 2
            num_out_channels *= 2
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv2_1')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv2_2')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool2')

            # layer 3
            num_out_channels *= 2
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv3_1')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv3_2')
            x = convolution_layer(x=x, kernel_size=1, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv3_3')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool3')

            # layer 4
            num_out_channels *= 2
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv4_1')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv4_2')
            x = convolution_layer(x=x, kernel_size=1, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv4_3')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool4')

            # layer 5
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv5_1')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv5_2')
            x = convolution_layer(x=x, kernel_size=1, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv5_3')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool5')

            # flatten
            x = tf.contrib.layers.flatten(x)

            # fully connected layers
            x = fully_connected_layer(x=x,
                                      num_outputs=4096,
                                      activation=self.activation,
                                      is_training=is_training,
                                      name='fc1')
            x = fully_connected_layer(x=x,
                                      num_outputs=4096,
                                      activation=self.activation,
                                      is_training=is_training,
                                      name='fc2')
            x = fully_connected_layer(x=x,
                                      num_outputs=4096,
                                      activation=self.activation,
                                      is_training=is_training,
                                      name='fc3')

            return x

    def __vgg_d(self, x, is_training):
        """
        :param x: input data
        :param is_training: whether we are training or testing (used by batch normalization)
        :return: output of VGG
        """
        with tf.compat.v1.variable_scope('VGG_D', reuse=tf.compat.v1.AUTO_REUSE):
            # layer 1
            num_out_channels = self.fan_out_init
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv1_1')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv1_2')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool1')

            # layer 2
            num_out_channels *= 2
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv2_1')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv2_2')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool2')

            # layer 3
            num_out_channels *= 2
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv3_1')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv3_2')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv3_3')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool3')

            # layer 4
            num_out_channels *= 2
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv4_1')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv4_2')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv4_3')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool4')

            # layer 5
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv5_1')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv5_2')
            x = convolution_layer(x=x, kernel_size=3, num_out_channels=num_out_channels, activation=self.activation,
                                  batch_norm=self.batch_norm, is_training=is_training, name='conv5_3')
            x = max_pooling_layer(x=x, pool_size=3, strides=2, name='pool5')

            # flatten
            x = tf.contrib.layers.flatten(x)

            # fully connected layers
            x = fully_connected_layer(x=x,
                                      num_outputs=4096,
                                      activation=self.activation,
                                      is_training=is_training,
                                      name='fc1')
            x = fully_connected_layer(x=x,
                                      num_outputs=4096,
                                      activation=self.activation,
                                      is_training=is_training,
                                      name='fc2')
            x = fully_connected_layer(x=x,
                                      num_outputs=4096,
                                      activation=self.activation,
                                      is_training=is_training,
                                      name='fc3')

            return x

    def evaluate(self, x, is_training):
        """
        :param x: input data
        :param is_training: whether we are training or testing (used by batch normalization)
        :return: output of VGG
        """
        # run corresponding architecture
        if self.config == 'A':
            return self.__vgg_a(x, is_training)
        elif self.config == 'C':
            return self.__vgg_c(x, is_training)
        elif self.config == 'D':
            return self.__vgg_d(x, is_training)
        else:
            raise Exception('Unknown VGG configuration!')