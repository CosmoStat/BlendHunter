import numpy as np
import os

"""Getting started"""

from os.path import expanduser
user_home = expanduser("~")

main_path = user_home+'/Documents/Cosmostat/Codes/BlendHunter'


"""Generate 35 folders for non padded images"""
for sigma in [5, 14, 18, 26, 35, 40]:
    for nb_real in ['', 1,2,3,4]:
        bh_path = main_path+'/bh_{}'.format(str(sigma)+str(nb_real))

        if os.path.isdir(bh_path):
            raise FileExistsError('{} already exists. Please remove this '
                          'directory or choose a new path.'
                          ''.format(bh_path))

        os.mkdir(bh_path)

"""Generate 35 folders for padded images"""
for sigma in [5, 14, 18, 26, 35, 40]:
    for nb_real in ['', 1,2,3,4]:
        bh_pad_path = main_path+'/bh_pad{}'.format(str(sigma)+str(nb_real))

        if os.path.isdir(bh_pad_path):
            raise FileExistsError('{} already exists. Please remove this '
                          'directory or choose a new path.'
                          ''.format(bh_pad_path))

        os.mkdir(bh_pad_path)

"""Generate results folder"""
bh_results = main_path+'/bh_results'
bh_pad_results = main_path+'/bh_pad_results'
sep_results = main_path+'/sep_results'
sep_pad_results = main_path+'/sep_pad_results'

if os.path.isdir(bh_results):
    raise FileExistsError('{} already exists. Please remove this '
                          'directory or choose a new path.'
                          ''.format(bh_results))
if os.path.isdir(bh_pad_results):
    raise FileExistsError('{} already exists. Please remove this '
                          'directory or choose a new path.'
                          ''.format(bh_pad_results))

if os.path.isdir(sep_results):
    raise FileExistsError('{} already exists. Please remove this '
                          'directory or choose a new path.'
                          ''.format(sep_results))

if os.path.isdir(sep_pad_results):
    raise FileExistsError('{} already exists. Please remove this '
                          'directory or choose a new path.'
                          ''.format(sep_pad_results))

os.mkdir(bh_results)
os.mkdir(sep_results)
os.mkdir(bh_pad_results)
os.mkdir(sep_pad_results)


"""Generate more datasets to test the weights (for both padded and non padded images)"""
more_noise_ranges= main_path+'/more_noise_ranges'
more_noise_ranges_pad= main_path+'/more_noise_ranges_pad'

if os.path.isdir(more_noise_ranges):
    raise FileExistsError('{} already exists. Please remove this '
                  'directory or choose a new path.'
                  ''.format(more_noise_ranges))
if os.path.isdir(more_noise_ranges_pad):
    raise FileExistsError('{} already exists. Please remove this directory or choose a new path.'
                  ''.format(more_noise_ranges_pad))

os.mkdir(more_noise_ranges)
os.mkdir(more_noise_ranges_pad)

"""Generate the folders for new test sets (for both padded and non padded images)"""
for i in [3,7,10,12,16,20,22,24,28,30,32,37,42,44]:
    bh_path =  main_path+'/more_noise_ranges/bh_{}'.format(i)

    if os.path.isdir(bh_path):
        raise FileExistsError('{} already exists. Please remove this '
                      'directory or choose a new path.'
                      ''.format(bh_path))
    os.mkdir(bh_path)

for i in [3,7,10,12,16,20,22,24,28,30,32,37,42,44]:
    bh_path =  main_path+'/more_noise_ranges_pad/bh_{}'.format(i)

    if os.path.isdir(bh_path):
        raise FileExistsError('{} already exists. Please remove this '
                      'directory or choose a new path.'
                      ''.format(bh_path))
    os.mkdir(bh_path)
