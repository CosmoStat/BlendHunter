#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""SCRIPT NAME

This module contains methods for making plots.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

:Credits: Alice Lacan

"""

import numpy as np
from utils import load, DataHandler


class GetAcc(DataHandler):
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
    n_noise_reals : int
        Number of noise realisations

    """

    def __init__(self, path, labels, sigma_values=None, xy_param=None,
                 n_noise_reals=5, sort=True):

        self.path = path
        self.labels = labels
        self.param_x = param_x
        self.param_y = param_y
        if sigma_values is None:
            self._out_shape = (1, n_noise_reals)
        else:
            self._out_shape = (sigma_values.size, n_noise_reals)
        super().__init__(path, sort)

        self._get_stats()
        if xy_param is not None:
            self._get_acc_wrt_dist()

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

        return np.sum(dataset == self.labels) / self.labels.size

    def _get_dist(self):

        return np.array([np.sqrt(x ** 2 + y ** 2)
                         for x, y in zip(self.param_x, self.param_y)])

    def _get_errors(self, dataset, label='blended'):

        return np.where(dataset[:dataset.size // 2] != label)[0]

    def _get_dist_hist(self, dist, bins=60, return_bins=False):

        counts, bins = np.histogram(dist, bins=bins)

        if return_bins:
            return counts, bins[1:], bins

        else:
            return counts

    def _get_acc_wrt_dist(self):

        distance = self._get_dist()

        res = [distance[self._get_errors(dataset, label='blended')]
               for dataset in self.datasets]

        n_total, bin_centres, bin_edges = (self._get_dist_hist(distance,
                                           return_bins=True))

        def ratio(x): return 1 - x / n_total

        acc_ratios = np.array([ratio(self._get_dist_hist(data,
                               bin_edges)) for data in res])

        self.dist_values = bin_centres
        self.mean_acc_dist = np.mean(acc_ratios.reshape(self._out_shape[0],
                                     self._out_shape[1], 60), axis=1)

    def _get_stats(self):
        """Get Statistics

        Calculate the mean and standard deviation of the classification
        accuracy for a given set of noise realisations.

        """

        res = (np.array([self._get_acc(dataset) for dataset in
               self.datasets]).reshape(self._out_shape))

        self.acc = res
        self.mean_acc = np.mean(res, axis=1)
        self.std_acc = np.std(res, axis=1)


# Set paths
results_path = '../results'
sim_res = results_path + '/sim_results'
cosmos_res = results_path + '/cosmos_results'

# Set sigma values and load true classification labels
sigma_values = load('{}/{}'.format(results_path, 'sigmas.npy'))
labels = load('{}/{}'.format(sim_res, 'labels.npy'))
labels_cos = load('{}/{}'.format(cosmos_res, 'labels.npy'))
param_x = load('{}/{}'.format(sim_res, 'param_x_total.npy'),
               limits=(36000, 40000))
param_y = load('{}/{}'.format(sim_res, 'param_y_total.npy'),
               limits=(36000, 40000))

# Get sim classification accuracy results for BlendHunter
bh_res = GetAcc(sim_res + '/bh_results', labels,
                sigma_values=sigma_values, xy_param=(param_x, param_y))

# Get sim classification accuracy results for SExtractor
se_res = GetAcc(sim_res + '/sep_results', labels,
                sigma_values=sigma_values, xy_param=(param_x, param_y))

# Get cosmos classification accuracy results for BlendHunter
bhc_res = GetAcc(cosmos_res + '/bh_results', labels_cos,
                 sigma_values=sigma_values, n_noise_reals=1)

# Get cosmos classification accuracy results for SExtractor
sec_res = GetAcc(cosmos_res + '/sep_results', labels_cos, n_noise_reals=1,
                 sort=False)

print('Saveing results to {}'.format(sim_res))


# Save sim classification accuracy results
np.save(sim_res + '/acc_results', np.array([bh_res.mean_acc, se_res.mean_acc,
                                            bh_res.std_acc, se_res.std_acc]))

# Save sim classification accuracy w.r.t. distance results
np.save(sim_res + '/dist_results', np.array([bh_res.dist_values,
                                             bh_res.mean_acc_dist,
                                             se_res.mean_acc_dist]))

# Save sim classification accuracy results
np.save(cosmos_res + '/acc_results', np.array([bhc_res.acc, sec_res.acc]))

print('Saveing results to {}'.format(cosmos_res))
