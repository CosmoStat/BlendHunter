# -*- coding: utf-8 -*-

import galsim
import numpy as np

def create_circular_mask(h, w, center=None, radius=None):
    """ Create circular mask

    This function create a circular mask in a numpy array.

    Parameters
    ----------
    h : int
        High of the image.
    w : int
        Width of the image.
    center : tuple
        Center of the object (x, y).
    radius : float
        Radius of the circle.

    Returns
    -------
    mask : numpy.ndarray
        Array containing the mask (in boolean)

    """

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def get_shapes(galsim_img, center):
    """ Get shapes

    This function compute the moments of an image. Then return the sigma of the
    window function used (size of the object) and the amplitude
    (flux of the object).

    Parameters
    ---------
    galsim_img : galsim.image.Image
        Galsim.image object containing the image.
    center : tuple
        Center of the object (x, y).

    Returns
    -------
    sigma : float
        Sigma of the window function, or -1 if an error occured.
    amp : float
        Moments amplitude, or -1 if an error occured.

    """

    shapes = galsim.hsm.FindAdaptiveMom(galsim_img,
                                        guess_centroid=galsim.PositionD(center),
                                        strict=False)
    if shapes.error_message == '':
        return shapes.moments_sigma, shapes.moments_amp
    else:
        return -1, -1

def get_snr(img, center, radius, sigma_noise):
    """ Get SNR

    This function compute a Galsim like SNR for an object.
    sqrt(sum(img**2) / sigma_noise**2)

    Parameters
    ----------
    img : numpy.ndarray
        Array representing the image.
    center : tuple
        Center of the object (x, y).
    radius : float
        "Size" of the object. (circular assumption).
    sigma_noise : float
        Sigma of the gaussian noise on the image.

    Returns
    -------
    snr : float
        SNR of the object.

    """

    mask = create_circular_mask(img.shape[0], img.shape[1],
                                center=center, radius=radius)

    snr = np.sqrt(np.sum(img[np.where(mask)]**2.)/sigma_noise**2.)

    return snr


def get_info_simu(obj, sigma_noise=14.5, ext=0):
    """ Get info simu

    This function return the fwhm and the snr of all the objects on an image.

    Paramters
    ---------
    obj : dict
        Dictionary containing the simulated object.
    sigma_noise : float
        Sigma of the gaussian noise on the image.
    ext : int
        In case of multi-epoch simulation, set the epoch you want [default : 0]

    Returns
    -------
    fwhm, snr : list, list
        List of the FWHM for all the objects.
    snr : list
        List of the SNR for all the objects.

    """

    galsim_img = obj['galsim_image_noisy'][ext]

    center_x = np.array([galsim_img.center.x-0.5])
    center_y = np.array([galsim_img.center.y-0.5])

    info_tmp = get_shapes(galsim_img, (center_x, center_y))

    fwhm = [info_tmp[0]*2.355]
    snr = [get_snr(galsim_img.array, (center_x, center_y),
                   info_tmp[0]*3., sigma_noise)]

    if obj['blended']:
        dx = obj['blend_param']['dx']
        dy = obj['blend_param']['dy']
        n_blend = len(dx)

        for  i in range(n_blend):

            info_tmp = get_shapes(galsim_img, (center_x + dx[i], center_y + dy[i]))

            fwhm.append(info_tmp[0]*2.355)
            snr.append(get_snr(galsim_img.array, (center_x + dx[i], center_y + dy[i]),
                           info_tmp[0]*3., sigma_noise))

    return fwhm, snr