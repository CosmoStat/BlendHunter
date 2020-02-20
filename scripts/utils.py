#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""SCRIPT NAME

This module contains methods for making plots.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

:Credits: Alice Lacan

"""

import os
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


class DataHandler:
    """Data Handler

    Parameters
    ----------
    path : str
        Input path

    """

    def __init__(self, path):

        self.path = path
        self._load_datasets()

    def _load_datasets(self):

        self.datasets = np.array([load('{}/{}'.format(self.path, file))
                                  for file in os.listdir(self.path)
                                  if file.endswith('.npy')])
