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
    def __init__(self, graph, x, gx, **kwargs):
        """
        :param graph: a computational graph with an 'evaluate(x, is_training)' method
        :param x: image data
        :param gx: perturbed image data
        :param y: image labels (only for debugging and performance tracking)
        :param kwargs: configuration dictionary for the base class
        """
        # save inputs
        self.x = x
        self.gx = gx

        # save the graph object
        self.graph = graph

        # run the graph
        self.z_x = self.graph.evaluate(self.x, is_training=True)
        self.z_gx = self.graph.evaluate(self.gx, is_training=True)
        self.z_x_test = self.graph.evaluate(self.x, is_training=False)

        # number of repeats
        self.num_repeats = kwargs['num_repeats']

        # head configuration
        self.k_A = 50
        self.num_A_sub_heads = 1
        self.k_B = kwargs['num_classes']
        self.num_B_sub_heads = 5

        # compatibility variables
        self.num_classes = self.K = self.k_B

        # get the losses for each head
        self.loss_A = self.__head_loss(self.k_A, self.num_A_sub_heads, 'A')
        self.loss_B = self.__head_loss(self.k_B, self.num_B_sub_heads, 'B')
        self.loss = self.loss_A + self.loss_B

        # configure optimizers
        self.gs = tf.Variable(0, name='global_step', trainable=False)
        self.opt = tf.train.AdamOptimizer(kwargs['learning_rate'])

        # configure training ops
        self.train_ops = []
        self.train_ops.append(tf.contrib.layers.optimize_loss(loss=self.loss_A,
                                                              global_step=self.gs,
                                                              learning_rate=kwargs['learning_rate'],
                                                              optimizer=self.opt,
                                                              summaries=['loss', 'gradients']))
        self.train_ops.append(tf.contrib.layers.optimize_loss(loss=self.loss_B,
                                                              global_step=self.gs,
                                                              learning_rate=kwargs['learning_rate'],
                                                              optimizer=self.opt,
                                                              summaries=['loss', 'gradients']))

        # test outputs
        self.pis = [self.__head_out(self.z_x_test, self.k_B, 'B' + str(i + 1)) for i in range(self.num_B_sub_heads)]

        # initialize performance dictionary
        self.num_epochs = kwargs['num_epochs']
        self.perf = None
        self.save_dir = kwargs['save_dir']

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
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            pi = tf.layers.dense(inputs=z,
                                 units=k,
                                 activation=tf.nn.softmax,
                                 use_bias=True,
                                 kernel_initializer=KERNEL_INIT,
                                 bias_initializer=BIAS_INIT)

        return pi

    def __head_loss(self, k, num_sub_heads, head):

        # loop over the number of sub-heads
        loss = tf.constant(0, dtype=tf.float32)
        for i in range(num_sub_heads):

            # run the model
            pi_x = self.__head_out(self.z_x, k, name=head + str(i + 1))
            num_vars = len(tf.global_variables())
            pi_gx = self.__head_out(self.z_gx, k, name=head + str(i + 1))
            assert num_vars == len(tf.global_variables())

            # accumulate the clustering loss
            loss += self.__iic_loss(pi_x, pi_gx)

        # take the average
        if num_sub_heads > 0:
            loss /= num_sub_heads

        return loss

    def performance_dictionary_init(self, num_epochs):
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

    def performance_dictionary_update(self, sess, y_ph, iter_init, set_partition, idx):
        """
        :param self: model class
        :param sess: TensorFlow session
        :param y_ph: TensorFlow placeholder for unseen labels
        :param iter_init: TensorFlow data iterator initializer associated
        :param set_partition: set name (e.g. train, validation, test)
        :param idx: insertion index (i.e. epoch - 1)
        :return: None
        """
        if self.perf is None:
            return

        # initialize results
        loss = []
        loss_A = []
        loss_B = []
        y = np.zeros([0, 1])
        y_hat = [np.zeros([0, self.k_B])] * self.num_B_sub_heads

        # initialize unsupervised data iterator
        sess.run(iter_init)

        # loop over the batches within the unsupervised data iterator
        print('Evaluating ' + set_partition + ' set performance... ', end='')
        while True:
            try:
                # grab the results
                results = sess.run([self.loss, self.loss_A, self.loss_B, y_ph, self.pis])#, self.x, self.gx])

                # load metrics
                loss.append(results[0])
                loss_A.append(results[1])
                loss_B.append(results[2])
                y = np.concatenate((y, np.expand_dims(results[3], axis=1)))
                for i in range(self.num_B_sub_heads):
                    y_hat[i] = np.concatenate((y_hat[i], results[4][i]))

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

        # new line
        print('Done')

        # average the results
        self.perf['loss'][set_partition][idx] = sum(loss) / len(loss)
        self.perf['loss_A'][set_partition][idx] = sum(loss_A) / len(loss_A)
        self.perf['loss_B'][set_partition][idx] = sum(loss_B) / len(loss_B)

        # compute classification accuracy
        class_errors = [unsupervised_labels(y_hat[i], y, self, set_partition) for i in range(self.num_B_sub_heads)]
        self.perf['class_err_min'][set_partition][idx] = np.min(class_errors)
        self.perf['class_err_avg'][set_partition][idx] = np.mean(class_errors)
        self.perf['class_err_max'][set_partition][idx] = np.max(class_errors)

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


def train(mdl_class, graph, mdl_config, train_set, test_set, early_stop_buffer=15):
    """
    :param mdl_class: model class object
    :param graph: the computational graph
    :param mdl_config: configuration dictionary
    :param train_set: TensorFlow Dataset object that corresponds to training data
    :param test_set: TensorFlow Dataset object that corresponds to validation data
    :param early_stop_buffer: early stop look-ahead distance (in epochs)
    :return: None
    """
    # construct iterator
    iterator = train_set.make_initializable_iterator()
    x, gx, y = iterator.get_next().values()

    # construct initialization operations
    train_iter_init = iterator.make_initializer(train_set)
    test_iter_init = iterator.make_initializer(test_set)

    # construct the model
    mdl = mdl_class(graph, x, gx, **mdl_config)

    # initialize performance dictionary
    mdl.performance_dictionary_init(mdl.num_epochs)

    # start a monitored session
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:

        # initialize model variables
        sess.run(tf.global_variables_initializer())

        # loop over the number of epochs
        for i in range(mdl.num_epochs):

            # start timer
            start = time.time()

            # get epoch number
            epoch = i + 1

            # get training operation
            i_train = i % len(mdl.train_ops)

            # initialize epoch iterator
            sess.run(train_iter_init)

            # loop over the batches
            while True:
                try:

                    # run training and loss
                    loss = sess.run([mdl.train_ops[i_train], mdl.loss])[-1]

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
            mdl.performance_dictionary_update(sess, y, train_iter_init, 'train', i)
            mdl.performance_dictionary_update(sess, y, test_iter_init, 'test', i)

            # plot learning curve
            mdl.plot_learning_curve(epoch)

            # pause for plot drawing if we aren't saving
            if mdl.save_dir is None:
                plt.pause(0.05)

            # print time for epoch
            stop = time.time()
            print('Time for Epoch = {:f}'.format(stop - start))

            # early stop check
            # i_best_elbo = np.argmin(mdl.perf['loss']['test'][:epoch])
            # i_best_class = np.argmin(mdl.perf['class_err']['test'][:epoch])
            # epochs_since_improvement = min(i - i_best_elbo, i - i_best_class)
            # print('Early stop checks: {:d} / {:d}\n'.format(epochs_since_improvement, early_stop_buffer))
            # if epochs_since_improvement >= early_stop_buffer:
            #     break

    # save the performance
    save_performance(mdl.perf, epoch, mdl.save_dir)


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
            'num_epochs': 3200,
            'save_dir': None},
    }

    # run training
    train(mdl_class=ClusterIIC,
          graph=IICGraph(config='B', fan_out_init=64),
          mdl_config=MDL_CONFIG[DATA_SET],
          train_set=TRAIN_SET,
          test_set=TEST_SET)

    print('All done!')
    plt.show()
