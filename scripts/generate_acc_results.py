#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def load(path, limits=None):
    """Load

    Load numpy binary with allow_pickle set to True.

    Parameters
    ----------
    path : str
        Path to file
    limits : tuple, optional
        Range of values to keep

    Returns
    -------
    numpy.ndarray
        Array loaded from numpy binary

    """

    data = np.load(path, allow_pickle=True)

    if limits:
        data = data[slice(*limits)]

    return data


class GetAcc:
    """Get Accuracy

    Class to calculate the average classification accuracy for a range of noise
    values and a series of noise realisations.

    Parameters
    ----------
    path : str
        Input path
    sigma_values : numpy.ndarray
        Noise standard deviation values
    labels : numpy.ndarray
        True classification labels
    prefix : str
        Input file prefix string
    n_noise_reals : int
        Number of noise realisations

    """

    def __init__(self, path, sigma_values, labels, param_x, param_y,
                 prefix='preds', n_noise_reals=5):

        self.path = path
        self.sigma_values = sigma_values
        self.labels = labels
        self.param_x = param_x
        self.param_y = param_y
        self.prefix = prefix
        self.noise_reals = np.array([''] + list(range(1, n_noise_reals)))
        self._load_datasets()
        self._get_stats()
        self._get_acc_wrt_dist()

    def _load_dataset(self, sigma, real, limits=None):
        """Load Dataset

        Parameters
        ----------
        sigma : float
            Noise standard deviation
        real : int
            Realisation index

        Returns
        -------
        numpy.ndarray
            Array loaded from numpy binary

        """

        return load('{}/{}{}{}.npy'.format(self.path, self.prefix, int(sigma),
                    real))

    def _load_datasets(self):

        self.datasets = [self._load_dataset(sigma, real)
                         for sigma in self.sigma_values
                         for real in self.noise_reals]

    def _get_acc(self, dataset):
        """Get Accuracy

        Calculate the classification accuracy for a given dataset.

        Parameters
        ----------
        dataset : numpy.ndarray
            Set of predictions for a given dataset

        Returns
        -------
        float
            Classification accuracy

        """

        if self.prefix == 'preds':

            return np.sum(dataset == self.labels) / self.labels.size

        else:

            return ((len(np.where(dataset[0:4000] == 1)[0]) +
                    len(np.where(dataset[4000:8000] == 0)[0])) /
                    (len(dataset[0:4000])+len(dataset[4000:8000])))

    def _get_dist(self):

        # self._load_dist()

        return np.array([np.sqrt(x ** 2 + y ** 2)
                         for x, y in zip(self.param_x, self.param_y)])

    def _get_errors(self, dataset, label='blended'):

        return np.where(dataset[0:4000] != label)[0]

    def _get_dist_hist(self, dist, bins=60, return_bins=False):

        counts, bins = np.histogram(dist, bins=bins)

        if return_bins:
            return counts, bins[1:], bins

        else:
            return counts

    def _get_acc_wrt_dist(self):

        distance = self._get_dist()

        if self.prefix == 'preds':
            label = 'blended'
        else:
            label = 1

        res = [distance[self._get_errors(dataset, label=label)]
               for dataset in self.datasets]

        n_total, bin_centres, bin_edges = (self._get_dist_hist(distance,
                                           return_bins=True))

        def ratio(x): return 1 - x / n_total

        acc_ratios = np.array([ratio(self._get_dist_hist(data,
                               bin_edges)) for data in res])

        self.dist_values = bin_centres
        self.mean_acc_dist = np.mean(acc_ratios.reshape(6, 5, 60), axis=1)

    def _get_stats(self):
        """Get Statistics

        Calculate the mean and standard deviation of the classification
        accuracy for a given set of noise realisations.

        """

        res = (np.array([self._get_acc(dataset) for dataset in
               self.datasets]).reshape(self.sigma_values.size,
               self.noise_reals.size))

        self.mean_acc = np.mean(res, axis=1)
        self.std_acc = np.std(res, axis=1)


# Set output path
results_path = '../results'

# Set sigma values and load true classification labels
sigma_values = np.array([5.0, 14.0, 18.0, 26.0, 35.0, 40.0])
labels = load('{}/{}'.format(results_path, 'labels.npy'))
param_x = load('{}/{}'.format(results_path, 'param_x_total.npy'),
               limits=(36000, 40000))
param_y = load('{}/{}'.format(results_path, 'param_y_total.npy'),
               limits=(36000, 40000))

# Get classification accuracy results for BlendHunter
bh_res = GetAcc(results_path + '/bh_results', sigma_values, labels,
                param_x, param_y)

# Get classification accuracy results for SExtractor
se_res = GetAcc(results_path + '/se_results', sigma_values, labels,
                param_x, param_y, prefix='flags_pad')

# Save noise standard deviation values
np.save(results_path + '/sigmas', sigma_values)

# Save classification accuracy results
np.save(results_path + '/acc_results', np.array([bh_res.mean_acc,
                                                 se_res.mean_acc,
                                                 bh_res.std_acc,
                                                 se_res.std_acc]))

# Save classification accuracy w.r.t. distance results
np.save(results_path + '/dist_results', np.array([bh_res.dist_values,
                                                  bh_res.mean_acc_dist,
                                                  se_res.mean_acc_dist]))
