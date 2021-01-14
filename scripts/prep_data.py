import numpy as np
from modopt.signal.noise import add_noise
from modopt.base.np_adjust import pad2d
from blendhunter.config import BHConfig
from blendhunter.data import CreateTrainData


def noise_to_add(img=None, SNR_=None, map=None):
    """Function to add noise according to signal"""
    # Calculate std of central object signal
    signal = np.sum(img[map == 1]**2)

    # Calculate noise to add on image
    std_noise = np.sqrt(signal) / SNR_

    # add noise
    return add_noise(img, sigma=std_noise)


def get_images(sample, add_noise_img=False, sigma_noise=None,
               add_padding_noise=False, fixed_snr=False, SNR_target=None,
               seg_map=None):
    """Function to get images

    Notes
    -----
    1. To pad with noise, use 'add_padding_noise=True' + 'sigma_noise'
    2. To simply add noise, use 'add_noise_img=True' + 'sigma_noise'
    3. To create datasets with fixed SNR according to signal use
    'fixed_snr=True' + 'SNR_target'
    4. Otherwise, the function retrieves non noisy simulations by defaul

    """
    if add_noise_img:
        # Add noise to image and store array in dict
        for i in range(len(sample)):
            sample[i]['galsim_image_noisy'] = (
                add_noise(sample[i]['galsim_image'][0].array,
                          sigma=sigma_noise)
            )

        return np.array([sample[obj]['galsim_image_noisy'] for obj in
                         range(sample.size)])

    if add_padding_noise:
        # Pad image 7 by 7
        for i in range(len(sample)):
            sample[i]['galsim_image_pad'] = (
                pad2d(sample[i]['galsim_image'][0].array, (7, 7))
            )

        # Then add noise to image and store array in dict
        for i in range(len(sample)):
            sample[i]['galsim_image_noisy'] = (
                add_noise(sample[i]['galsim_image_pad'], sigma=sigma_noise)
            )

        return np.array([sample[obj]['galsim_image_noisy'] for obj in
                         range(sample.size)])

    # Add noise according to a fixed SNR level to reach
    if fixed_snr:
        for i in range(len(sample)):
            sample[i]['galsim_image_noisy'] = (
                noise_to_add(img=sample[i]['galsim_image'][0].array,
                             SNR_=SNR_target, map=seg_map[i])
            )
            # Normalise images
            sample[i]['galsim_image_noisy'] /= (
                sample[i]['galsim_image_noisy'].max()
            )

        return np.array([sample[obj]['galsim_image_noisy'] for obj in
                         range(sample.size)])

    # Return only non noisy image
    else:
        return np.array([sample[obj]['galsim_image'][0].array for obj in
                        range(sample.size)])


def prep_data(out_path, noise_sigma, n_noise_real, blended, not_blended,
              dir_str='bh_', verbose=True):

    for sigma in noise_sigma:
        for noise_real in range(n_noise_real):

            id = f'{str(sigma)}{str(noise_real)}'
            path = f'{out_path}/{dir_str}{id}'

            if verbose:
                print(f'Processing {dir_str}{id}')

            images = [get_images(sample, add_padding_noise=True,
                                 sigma_noise=sigma)
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
prep_data(out_path, noise_sigma, n_noise_real, blended, not_blended)
# Prepare padded images
prep_data(out_path, noise_sigma, n_noise_real, blended, not_blended,
          dir_str='bh_pad')
