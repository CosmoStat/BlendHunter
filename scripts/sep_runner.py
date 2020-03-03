# -*- coding: utf-8 -*-

"""SEP RUNNER

This module contains a class for calling the SEP (SExtractor) package.

:Author: Axel Guinot <axel.guinot@cea.fr>

:Credits: Samuel Farrens

"""


import numpy as np
import sep
import pandas as pd
from statistics import mean


class Run_Sep:
    """ Run SEP

    This class runs the SEP (SExtractor) package and assigns a flag to identify
    objects as blended or not.

    """

    def __init__(self):

        self.BIG_DISTANCE = 1e30
        self.NO_BLEND = 0
        self.BLEND = 1
        self.MISS_EXTRACTION = 16

    @staticmethod
    def get_power_2(x):
        """ Get power 2

        Decompose a number on power of 2 and return the powers.

        Example
        -------
        3 -> 1, 2
        16 -> 16
        13 -> 1, 4, 8

        Paramters
        ---------
        x : int
            Number to decompose.

        Returns
        -------
        powers : list
            List of the powers find.

        """

        powers = []
        i = 1
        while i <= x:
            if i & x:
                powers.append(i)
            i <<= 1
        return powers

    def run_sep(self, img, thresh=1.5, deblend_nthresh=32, deblend_cont=0.005,
                sig_noise=None):
        """ Run sep

        Run sep algorithm on a vignet.

        Parameters
        ----------
        img : numpy.ndarray
            Array containing the image.
        thresh : float
            Detection threshold. (x*sigma_noise with thresh=x)
        deblend_nthresh : int
            Number of bin to do for the blend identification (default = 32).
        deblend_cont : float
            Minimum flux ratio to consider a blend in [0, 1] (default = 0.005).
        sig_noise : float
            Sigma of the noise if known. If None it will be derive from the
            background which might not be accurate on small vignets.

        Returns
        -------
        res : list
            List return by sep.extract.

        """

        sigmas = []

        if sig_noise is None:
            bkg = sep.Background(img)
            sig_noise = bkg.globalrms

        res = sep.extract(img, thresh, err=sig_noise,
                          deblend_nthresh=deblend_nthresh,
                          deblend_cont=deblend_cont)
        return res

    def check_blend(self, res):
        """ Check blend

        Check if at least one object identify by sep.extract is flagged as 
        blended by SExtractor.

        Parameters
        ----------
        res : list
            Output of sep.extract method.

        Returns
        -------
        flag : bool
            flag = 1 if the object is blended and 0 otherwise.

        """

        n_obj = len(res)

        flag = np.array([1 if sep.OBJ_MERGED in self.get_power_2(res[i]['flag'])
                 else 0 for i in range(n_obj)], dtype=bool).sum()

        return flag

    def process(self, f, epoch=0):
        """ Process

        Make the all process of blend identification using SExtractor routines.
        Blend definition used :
        0 : no blend
        1 : blend well find
        16 : miss identification

        Parameters
        ----------
        f : numpy.ndarray
            Output of the LenSimu code.
        epoch : int
            Epoch of interest in case of multi-epoch simulations (default = 0).

        Returns
        -------
        blend_flag : numpy.ndarray
            Array containing the final flag for each vignets.

        """

        final_res = []
        final_flags = []
        dists = []
        all_res = []
        # 0 : no blend, 1 : blend well find, 16 : miss identification
        blend_flag = []

        for obj in f:
            # Retrieve image w/out noise
            # img = obj['galsim_image'][epoch].array

            # Retrieve image w/ noise
            img = obj['galsim_image_noisy']

            # Retrieve denoised image
            # img = obj['denoised_img']

            res = self.run_sep(img)

            if len(res) == 0:
                blend_tmp = self.MISS_EXTRACTION

            elif len(res) == 1:
                if obj['blended']:
                    if self.check_blend(res):
                        blend_tmp = self.BLEND
                    else:
                        blend_tmp = self.NO_BLEND
                else:
                    blend_tmp = self.NO_BLEND
            else:
                blend_tmp = self.BLEND
            

            blend_flag.append(blend_tmp)

            all_res.append(res)

        return np.array(blend_flag), all_res

    @staticmethod
    def get_estimated_sigma(img):
        bkg = sep.Background(img)
        sig_noise = bkg.globalrms

        return sig_noise

    @staticmethod
    def plot_sex_obj(res, img):
        """ Plot SExtractor object

        Plot contours where SExtractor find objects on top of the original
        image.

        Paramters
        ---------
        res : numpy.ndarray
            Structure array return by sep.extract method for one object.
        img : numpy.ndarray
            Array containing the image.

        """

        fig, ax = plt.subplots()
        m, s = np.mean(img), np.std(img)
        im = ax.imshow(img, interpolation='nearest', cmap='gray',
                       vmin=m-s, vmax=m+s, origin='lower')

        # plot an ellipse for each object
        for i in range(len(res)):
            e = Ellipse(xy=(res['x'][i], res['y'][i]),
                        width=6*res['a'][i],
                        height=6*res['b'][i],
                        angle=res['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)
        plt.show()
