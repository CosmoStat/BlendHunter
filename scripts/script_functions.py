import numpy as np
from modopt.signal.noise import add_noise
from modopt.base.np_adjust import pad2d


def sigma_from_SNR(data, SNR, map):
    """Function to add noise according to signal"""
    # Calculate std of central object signal
    signal = np.sum(data[map == 1] ** 2)

    # Calculate noise to add on image
    return np.sqrt(signal) / SNR


def get_images(sample, add_padding=True, add_noise_sigma=True,
               add_pad_noise=False, sigma_noise=None, fixed_snr=False,
               SNR_target=None, seg_map=None):
    """Function to get images

    Notes
    -----
    1. To pad with noise, use 'add_padding=True' + 'sigma_noise'
    2. To simply add noise, use 'add_noise=True' + 'sigma_noise'
    3. To create datasets with fixed SNR according to signal use
    'fixed_snr=True' + 'SNR_target'
    4. Otherwise, the function retrieves noiseless simulations by default

    """

    images = []

    for i, sample_image in enumerate(sample):

        image = sample_image['galsim_image'][0].array

        if add_padding:
            image = pad2d(image, (7, 7))

        if add_noise_sigma:
            image = add_noise(image, sigma=sigma_noise)

        if add_pad_noise:
            image = np.pad(image, 7, constant_values=np.nan)
            noise = sigma_noise * np.random.randn(image[np.isnan(image)].size)
            image[np.isnan(image)] = noise

        if fixed_snr:
            sigma = sigma_from_SNR(image, SNR=SNR_target, map=seg_map[i])
            image = add_noise(image, sigma=sigma)
            image /= image.max()

        images.append(image)
        sample_image['galsim_image'] = image

    return np.array(images)
