
"""Getting started"""

import numpy as np
import os
from blendhunter.config import BHConfig


def make_dir(dir_name):

    if os.path.isdir(dir_name):
        raise FileExistsError(
            f'{dir_name} already exists. Please remove this directory or ' +
            'choose a new path.'
        )

    else:
        os.mkdir(dir_name)


def create_dirs(out_path, noise_sigma, n_noise_real, dir_str='bh_'):

    for sigma in noise_sigma:
        for noise_real in range(n_noise_real):

            dir_name = f'{out_path}/{dir_str}{str(sigma)}{str(noise_real)}'

            make_dir(dir_name)
            make_dir(os.path.join(dir_name, 'weights'))


# Read BH configuration file
bhconfig = BHConfig().config
out_path = bhconfig['out_path']
noise_sigma = bhconfig['noise_sigma']
n_noise_real = bhconfig['n_noise_real']

# Generate 35 directories for non padded images
create_dirs(out_path, noise_sigma, n_noise_real)

# Generate 35 directories for padded images
create_dirs(out_path, noise_sigma, n_noise_real, dir_str='bh_pad')

# Generate results directories, directories for regrouping the pre-trained
# weights and directories for real data (Cosmos images)
for dir in (
    'bh_results',
    'bh_pad_results',
    'sep_results',
    'sep_pad_results',
    'pretrained_weights',
    'pretrained_weights_pad',
    'bh_real',
    'bh_real_pad'
):
    make_dir(os.path.join(out_path, dir))
