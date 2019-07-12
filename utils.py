import os
import pickle
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment


def save_performance(perf, epoch, save_path):
    """
    :param perf: performance dictionary
    :param epoch: epoch number
    :param save_path: path to save plot to. if None, plot will be drawn
    :return: none
    """
    # return if save path is None
    if save_path is None:
        return

    # loop over the metrics
    for metric in perf.keys():

        # loop over the data splits
        for split in perf[metric].keys():

            # trim data to utilized epochs
            perf[metric][split] = perf[metric][split][:epoch]
            assert len(perf[metric][split]) == epoch

    # create the file name
    f_name = os.path.join(save_path, 'perf.pkl')

    # pickle it
    with open(f_name, 'wb') as f:
        pickle.dump(perf, f, pickle.HIGHEST_PROTOCOL)

    # make sure it worked
    with open(f_name, 'rb') as f:
        perf_load = pickle.load(f)
    assert str(perf) == str(perf_load), 'performance saving failed'


def unsupervised_labels(alpha, y, mdl, loss_type):
    """
    :param alpha: concentration parameter
    :param y: true label
    :param mdl: the model object
    :param loss_type: name used for printing updates
    :return: classification error rate
    """
    # same number of classes as labels?
    if mdl.K == mdl.num_classes:

        # construct y-hat
        y_hat = np.argmax(alpha, axis=1)

        # initialize count matrix
        cnt_mtx = np.zeros([mdl.K, mdl.K])

        # fill in matrix
        for i in range(len(y)):
            cnt_mtx[int(y_hat[i]), int(y[i])] += 1

        # find optimal permutation
        row_ind, col_ind = linear_sum_assignment(-cnt_mtx)

        # compute error
        error = 1 - cnt_mtx[row_ind, col_ind].sum() / cnt_mtx.sum()

    # different number of classes than labels
    else:

        # initialize y-hat
        y_hat = -np.ones(y.shape)

        # loop over the number of latent clusters
        for i in range(mdl.K):

            # find the real label corresponding to the largest concentration for this cluster
            i_sort = np.argsort(alpha[:, i])[-100:]
            y_real = stats.mode(y[i_sort])[0]

            # assign that label to all points where its concentration is maximal
            y_hat[np.argmax(alpha, axis=1) == i] = y_real

        # make sure we handled everyone
        assert np.sum(y_hat < 0) == 0

        # compute the error
        error = np.mean(y != y_hat)

    # print results
    print('Classification error for ' + loss_type + ' data = {:.4f}'.format(error))

    return error
