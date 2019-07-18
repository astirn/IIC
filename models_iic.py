import copy
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib.ticker import FormatStrFormatter

# import data loader
from data import load

# import computational graphs
from graphs import IICGraph, VGG, KERNEL_INIT, BIAS_INIT

# import utility functions
from utils import unsupervised_labels, save_performance

# plot settings
DPI = 600


class ClusterIIC(object):
    def __init__(self, num_classes, learning_rate, num_repeats, save_dir=None):
        """
        :param num_classes: number of classes
        :param learning_rate: gradient step size
        :param num_repeats: number of data repeats for x and g(x), used to up-sample
        """
        # training indicating placeholder
        self.is_training = tf.placeholder(tf.bool)

        # number of repeats
        self.num_repeats = num_repeats

        # save configuration
        self.k_A = 5 * num_classes
        self.num_A_sub_heads = 1
        self.k_B = num_classes
        self.num_B_sub_heads = 5

        # initialize losses
        self.loss_A = None
        self.loss_B = None
        self.loss = None

        # initialize optimizer
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.train_ops = []

        # initialize outputs outputs
        self.y_hats = None

        # initialize performance dictionary
        self.perf = None
        self.save_dir = save_dir

        # configure performance plotting
        self.fig_learn, self.ax_learn = plt.subplots(1, 2)

    def __iic_loss(self, pi_x, pi_gx):

        # up-sample non-perturbed to match the number of repeat samples
        pi_x = tf.tile(pi_x, [self.num_repeats] + [1] * len(pi_x.shape.as_list()[1:]))

        # get K
        k = pi_x.shape.as_list()[1]

        # compute P
        p = tf.transpose(pi_x) @ pi_gx

        # enforce symmetry
        p = (p + tf.transpose(p)) / 2

        # enforce minimum value
        p = tf.clip_by_value(p, clip_value_min=1e-6, clip_value_max=tf.float32.max)

        # normalize
        p /= tf.reduce_sum(p)

        # get marginals
        pi = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=0), (k, 1)), (k, k))
        pj = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=1), (1, k)), (k, k))

        # complete the loss
        loss = -tf.reduce_sum(p * (tf.math.log(p) - tf.math.log(pi) - tf.math.log(pj)))

        return loss

    @staticmethod
    def __head_out(z, k, name):

        # construct a new head that operates on the model's output for x
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            phi = tf.layers.dense(
                inputs=z,
                units=k,
                activation=tf.nn.softmax,
                use_bias=True,
                kernel_initializer=KERNEL_INIT,
                bias_initializer=BIAS_INIT)

        return phi

    def __head_loss(self, z_x, z_gx, k, num_sub_heads, head):

        # loop over the number of sub-heads
        loss = tf.constant(0, dtype=tf.float32)
        for i in range(num_sub_heads):

            # run the model
            pi_x = self.__head_out(z_x, k, name=head + str(i + 1))
            num_vars = len(tf.compat.v1.global_variables())
            pi_gx = self.__head_out(z_gx, k, name=head + str(i + 1))
            assert num_vars == len(tf.compat.v1.global_variables())

            # accumulate the clustering loss
            loss += self.__iic_loss(pi_x, pi_gx)

        # take the average
        if num_sub_heads > 0:
            loss /= num_sub_heads

        return loss

    def __build(self, x, gx, graph):

        # run the graph
        z_x = graph.evaluate(x, is_training=self.is_training)
        z_gx = graph.evaluate(gx, is_training=self.is_training)

        # construct losses
        self.loss_A = self.__head_loss(z_x, z_gx, self.k_A, self.num_A_sub_heads, 'A')
        self.loss_B = self.__head_loss(z_x, z_gx, self.k_B, self.num_B_sub_heads, 'B')
        self.loss = self.loss_A + self.loss_B

        # set alternating training operations
        self.train_ops.append(tf.contrib.layers.optimize_loss(loss=self.loss_A,
                                                              global_step=self.global_step,
                                                              learning_rate=self.learning_rate,
                                                              optimizer=self.opt,
                                                              summaries=['loss', 'gradients']))
        self.train_ops.append(tf.contrib.layers.optimize_loss(loss=self.loss_B,
                                                              global_step=self.global_step,
                                                              learning_rate=self.learning_rate,
                                                              optimizer=self.opt,
                                                              summaries=['loss', 'gradients']))

        # initialize outputs outputs
        self.y_hats = [tf.argmax(self.__head_out(z_x, self.k_B, 'B' + str(i + 1)), axis=1)
                       for i in range(self.num_B_sub_heads)]

    def __performance_dictionary_init(self, num_epochs):
        """
        :param num_epochs: maximum number of epochs (used to size buffers)
        :return: None
        """
        # initialize performance dictionary
        init_dict = {'train': np.zeros(num_epochs),
                     'test': np.zeros(num_epochs)}
        self.perf = dict()

        # loss terms
        self.perf.update({'loss': copy.deepcopy(init_dict)})
        self.perf.update({'loss_A': copy.deepcopy(init_dict)})
        self.perf.update({'loss_B': copy.deepcopy(init_dict)})

        # classification error
        self.perf.update({'class_err_min': copy.deepcopy(init_dict)})
        self.perf.update({'class_err_avg': copy.deepcopy(init_dict)})
        self.perf.update({'class_err_max': copy.deepcopy(init_dict)})

    def __performance_dictionary_update(self, sess, iter_init, set_partition, idx, y_ph=None):
        """
        :param self: model class
        :param sess: TensorFlow session
        :param iter_init: TensorFlow data iterator initializer associated
        :param set_partition: set name (e.g. train, validation, test)
        :param idx: insertion index (i.e. epoch - 1)
        :param y_ph: TensorFlow placeholder for unseen labels
        :return: None
        """
        if self.perf is None:
            return

        # initialize results
        loss = []
        loss_A = []
        loss_B = []
        y = np.zeros([0, 1])
        y_hats = [np.zeros([0, 1])] * self.num_B_sub_heads

        # initialize unsupervised data iterator
        sess.run(iter_init)

        # configure metrics lists
        metrics = [self.loss, self.loss_A, self.loss_B, self.y_hats]
        if y_ph is not None:
            metrics.append(y_ph)

        # loop over the batches within the unsupervised data iterator
        print('Evaluating ' + set_partition + ' set performance... ')
        while True:
            try:
                # grab the results
                results = sess.run(metrics, feed_dict={self.is_training: False})

                # load metrics
                loss.append(results[0])
                loss_A.append(results[1])
                loss_B.append(results[2])
                for i in range(self.num_B_sub_heads):
                    y_hats[i] = np.concatenate((y_hats[i], np.expand_dims(results[3][i], axis=1)))
                if y_ph is not None:
                    y = np.concatenate((y, np.expand_dims(results[-1], axis=1)))

                # _, ax = plt.subplots(2, 10)
                # i_rand = np.random.choice(results[3].shape[0], 10)
                # for i in range(10):
                #     ax[0, i].imshow(results[3][i_rand[i]][:, :, 0], origin='upper', vmin=0, vmax=1)
                #     ax[0, i].set_xticks([])
                #     ax[0, i].set_yticks([])
                #     ax[1, i].imshow(results[4][i_rand[i]][:, :, 0], origin='upper', vmin=0, vmax=1)
                #     ax[1, i].set_xticks([])
                #     ax[1, i].set_yticks([])
                # plt.show()

            # iterator will throw this error when its out of data
            except tf.errors.OutOfRangeError:
                break

        # average the results
        self.perf['loss'][set_partition][idx] = np.mean(loss)
        self.perf['loss_A'][set_partition][idx] = np.mean(loss_A)
        self.perf['loss_B'][set_partition][idx] = np.mean(loss_B)

        # compute classification accuracy
        if y_ph is not None:
            class_errors = [unsupervised_labels(y, y_hats[i], self.k_B, self.k_B)
                            for i in range(self.num_B_sub_heads)]
            self.perf['class_err_min'][set_partition][idx] = np.min(class_errors)
            self.perf['class_err_avg'][set_partition][idx] = np.mean(class_errors)
            self.perf['class_err_max'][set_partition][idx] = np.max(class_errors)

        # metrics are done
        print('Done')

    def plot_learning_curve(self, epoch):
        """
        :param epoch: epoch number
        :return: None
        """
        # generate epoch numbers
        t = np.arange(1, epoch + 1)

        # colors
        c = {'train': '#1f77b4', 'test': '#ff7f0e'}

        # text position
        x50 = int(epoch / 2 + 0.5)
        x75 = int(3 * epoch / 4 + 0.5)

        # plot the loss
        self.ax_learn[0].clear()
        self.ax_learn[0].set_title('Loss')
        self.ax_learn[0].plot(t, self.perf['loss']['train'][:epoch], linestyle='-', color=c['train'])
        self.ax_learn[0].plot(t, self.perf['loss']['test'][:epoch], linestyle='-', color=c['test'])
        self.ax_learn[0].text(x75,
                              min(min(self.perf['loss']['train'][:x50]), min(self.perf['loss']['test'][:x50])),
                              'Total')
        self.ax_learn[0].plot(t, self.perf['loss_A']['train'][:epoch], linestyle='--', color=c['train'])
        self.ax_learn[0].plot(t, self.perf['loss_A']['test'][:epoch], linestyle='--', color=c['test'])
        self.ax_learn[0].text(x75,
                              min(min(self.perf['loss_A']['train'][:x50]), min(self.perf['loss_A']['test'][:x50])),
                              'Head A')
        self.ax_learn[0].plot(t, self.perf['loss_B']['train'][:epoch], linestyle=':', color=c['train'])
        self.ax_learn[0].plot(t, self.perf['loss_B']['test'][:epoch], linestyle=':', color=c['test'])
        self.ax_learn[0].text(x75,
                              min(min(self.perf['loss_B']['train'][:x50]), min(self.perf['loss_B']['test'][:x50])),
                              'Head B')
        self.ax_learn[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        self.ax_learn[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # plot the classification error
        self.ax_learn[1].clear()
        self.ax_learn[1].set_title('Class. Error (Min, Avg, Max)')
        self.ax_learn[1].plot(t, self.perf['class_err_avg']['train'][:epoch], color=c['train'])
        self.ax_learn[1].fill_between(t,
                                      self.perf['class_err_min']['train'][:epoch],
                                      self.perf['class_err_max']['train'][:epoch],
                                      facecolor=c['train'], alpha=0.5)
        self.ax_learn[1].plot(t, self.perf['class_err_avg']['test'][:epoch], color=c['test'])
        self.ax_learn[1].fill_between(t,
                                      self.perf['class_err_min']['test'][:epoch],
                                      self.perf['class_err_max']['test'][:epoch],
                                      facecolor=c['test'], alpha=0.5)
        self.ax_learn[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        self.ax_learn[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # make the legend
        self.ax_learn[1].legend(handles=[patches.Patch(color=val, label=key) for key, val in c.items()],
                                ncol=len(c),
                                bbox_to_anchor=(0.25, -0.06))

        # eliminate those pesky margins
        self.fig_learn.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95, wspace=0.25, hspace=0.3)

    def train(self, graph, train_set, test_set, num_epochs, early_stop_buffer=15):
        """
        :param graph: the computational graph
        :param train_set: TensorFlow Dataset object that corresponds to training data
        :param test_set: TensorFlow Dataset object that corresponds to validation data
        :param num_epochs: number of epochs
        :param early_stop_buffer: early stop look-ahead distance (in epochs)
        :return: None
        """
        # construct iterator
        iterator = tf.compat.v1.data.make_initializable_iterator(train_set)
        x, gx, y = iterator.get_next().values()

        # construct initialization operations
        train_iter_init = iterator.make_initializer(train_set)
        test_iter_init = iterator.make_initializer(test_set)

        # build the model using the supplied computational graph
        self.__build(x, gx, graph)

        # initialize performance dictionary
        self.__performance_dictionary_init(num_epochs)

        # start a monitored session
        cfg = tf.compat.v1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=cfg) as sess:

            # initialize model variables
            sess.run(tf.global_variables_initializer())

            # loop over the number of epochs
            for i in range(num_epochs):

                # start timer
                start = time.time()

                # get epoch number
                epoch = i + 1

                # get training operation
                i_train = i % len(self.train_ops)

                # initialize epoch iterator
                sess.run(train_iter_init)

                # loop over the batches
                while True:
                    try:

                        # run training and loss
                        loss = sess.run([self.train_ops[i_train], self.loss], feed_dict={self.is_training: True})[-1]

                        if np.isnan(loss):
                            print('\n NaN whelp!')
                            return

                        # print update
                        print('\rEpoch {:d}, Loss = {:.4f}'.format(epoch, loss), end='')

                    # iterator will throw this error when its out of data
                    except tf.errors.OutOfRangeError:
                        break

                # new line
                print('')

                # get data set performances
                self.__performance_dictionary_update(sess, train_iter_init, 'train', i, y)
                self.__performance_dictionary_update(sess, test_iter_init, 'test', i, y)

                # plot learning curve
                self.plot_learning_curve(epoch)

                # pause for plot drawing if we aren't saving
                if self.save_dir is None:
                    plt.pause(0.05)

                # print time for epoch
                stop = time.time()
                print('Time for Epoch = {:f}'.format(stop - start))

                # early stop check
                # i_best_elbo = np.argmin(self.perf['loss']['test'][:epoch])
                # i_best_class = np.argmin(self.perf['class_err']['test'][:epoch])
                # epochs_since_improvement = min(i - i_best_elbo, i - i_best_class)
                # print('Early stop checks: {:d} / {:d}\n'.format(epochs_since_improvement, early_stop_buffer))
                # if epochs_since_improvement >= early_stop_buffer:
                #     break

        # save the performance
        save_performance(self.perf, epoch, self.save_dir)


if __name__ == '__main__':
    # pick a data set
    DATA_SET = 'mnist'

    # define splits
    DS_CONFIG = {
        # mnist data set parameters
        'mnist': {
            'batch_size': 700,
            'num_repeats': 5,
            'mdl_input_dims': [24, 24, 1]}
    }

    # load the data set
    TRAIN_SET, TEST_SET, SET_INFO = load(data_set_name=DATA_SET, **DS_CONFIG[DATA_SET])

    # configure the common model elements
    MDL_CONFIG = {
        # mist hyper-parameters
        'mnist': {
            'num_classes': SET_INFO.features['label'].num_classes,
            'learning_rate': 1e-4,
            'num_repeats': DS_CONFIG[DATA_SET]['num_repeats'],
            'save_dir': None},
    }

    # declare the model
    mdl = ClusterIIC(**MDL_CONFIG[DATA_SET])

    # train the model
    mdl.train(IICGraph(config='B', fan_out_init=64), TRAIN_SET, TEST_SET, num_epochs=600)

    print('All done!')
    plt.show()
