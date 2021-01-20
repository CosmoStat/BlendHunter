import numpy as np
from blendhunter.config import BHConfig
from blendhunter.data import CreateTrainData
import script_functions as sf


def prep_data(out_path, noise_sigma, n_noise_real, blended, not_blended,
              dir_str='bh_', verbose=True):

    for sigma in noise_sigma:
        for noise_real in range(n_noise_real):

            id = f'{str(sigma)}{str(noise_real)}'
            path = f'{out_path}/{dir_str}{id}'
            pad = 'pad' in dir_str

            if verbose:
                print(f'Processing {dir_str}{id}')

            images = [sf.get_images(sample, add_padding=pad,
                                    add_noise_sigma=True, sigma_noise=sigma)
                      for sample in (blended, not_blended)]

            # Save noisy test images for comparison w/ SExtractor
            np.save(f'{path}/blended_noisy{id}.npy', blended)
            np.save(f'{path}/not_blended_noisy{id}.npy', not_blended)

            # Train-valid-test split
            CreateTrainData(images, path).prep_axel(path_to_output=path)


# Read BH configuration file
bhconfig = BHConfig().config
out_path = bhconfig['out_path']
input = bhconfig['in_path']
noise_sigma = bhconfig['noise_sigma']
n_noise_real = bhconfig['n_noise_real']
sample_range = slice(*bhconfig['sample_range'])

# Getting the images
blended = np.load(input + '/blended/gal_obj_0.npy',
                  allow_pickle=True)[sample_range]
not_blended = np.load(input + '/not_blended/gal_obj_0.npy',
                      allow_pickle=True)[sample_range]

# Prepare non padded images
# prep_data(out_path, noise_sigma, n_noise_real, blended, not_blended)
# Prepare padded images
prep_data(out_path, noise_sigma, n_noise_real, blended, not_blended,
          dir_str='bh_pad')
