import numpy as np
import sep
from statistics import mean
import pandas as pd

BIG_DISTANCE = 1e30
NO_BLEND = 0
BLEND = 1
MISS_EXTRACTION = 16


class Run_Sep:
    """Run SEP"""

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

        Check if objects identify by sep.extract are flagged as blended by
        SExtractor.

        Parameters
        ----------
        res : list
            Output of sep.extract method.

        Returns
        -------
        flags : list
            List of the flag for each objects. flag = 1 if the object is
            blended and 0 otherwise.

        """

        n_obj = len(res)

        flags = [1 if sep.OBJ_MERGED in self.get_power_2(res[i]['flag'])
                 else 0 for i in range(n_obj)]

        return flags

    @staticmethod
    def compute_dist(obj_res, obj_true_pos):
        """ Compute distance

        Compute the euclidean distance between 2 points.

        Parameters
        ----------
        obj_res : numpy.ndarray
            Structure array return by sep.extract method for one object.
        obj_true_pos : list
            List containing the true position. (example : [x, y])

        Returns
        -------
        dist : float
            Euclidean distance between 2 points.

        """

        dist = np.sqrt((obj_res['x'] - obj_true_pos[0]) ** 2. +
                       (obj_res['y'] - obj_true_pos[1])**2.)

        return dist

    def get_dist(self, res, true_pos, thresh=2):
        """ Get distance

        This method return the distance between the real objects and the best
        match find in the SExtractor output. If the distance is larger than the
        the threshold seted then the object is flag.

        Parameters
        ----------
        res : list
            Output of the sep.extract method.
        true_pos : list
            List of the true positions (example : [[x1, y1], [x2, y2], ...]
        thresh : float
            Maximum distance allowed for a match (default = 2)

        Returns
        -------
        numpy.ndarray
            Array containing all the distances.

        """

        dists = []
        ind_used = []
        for i in range(len(true_pos)):
            tmp = BIG_DISTANCE
            for j in range(len(res)):
                tmp2 = self.compute_dist(res[j], true_pos[i])
                ind_tmp = -10
                if (tmp2 < tmp) and (j not in ind_used) and (tmp2 < thresh):
                    tmp = tmp2
                    ind_tmp = j
            ind_used.append(ind_tmp)
            dists.append(tmp)

        return np.array(dists)

    def _check_dist(self, dists):
        """ Check distance

        Flag a list of object depending if SExtractor well identify them or
        not. If at least one object in the list is not well identify will
        return False.

        Parameters
        ----------
        dists : list
            List of distances return by the method get_dist

        Returns
        int
            1 if all the objects are good and 0 otherwise

        """

        return np.prod([1 if dist < BIG_DISTANCE else 0 for dist in dists],
                       dtype=bool).astype(int)

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
        blend_flag = []
        # 0 : no blend, 1 : blend well find, 16 : miss identification

        for obj in f:
            # Retrieve image w/out noise
            # img = obj['galsim_image'][epoch].array

            # Retrieve image w/ noise
            img = obj['galsim_image_noisy']

            # Retrieve denoised image
            # img = obj['denoised_img']

            true_nobj = 1
            if obj['blended'] is True:
                true_nobj += len(obj['blend_param']['dx'])

            res = self.run_sep(img)

            obj_pos = [[img.shape[0] / 2., img.shape[1] / 2.]]

            if len(res) == 0:
                blend_tmp = MISS_EXTRACTION

            elif len(res) == 1:
                if obj['blended'] is True:
                    obj_pos += [[obj_pos[0][0] + obj['blend_param']['dx'][i], obj_pos[0][1] + obj['blend_param']['dy'][i]] for i in range(true_nobj-1)]
                dists_tmp = self.get_dist(res, obj_pos)
                flag = self.check_blend(res)
                if (flag[0]) and self._check_dist(dists_tmp):
                    blend_tmp = BLEND
                else:
                    blend_tmp = NO_BLEND
            elif len(res) == true_nobj:
                obj_pos += [[obj_pos[0][0] + obj['blend_param']['dx'][i], obj_pos[0][1] + obj['blend_param']['dy'][i]] for i in range(true_nobj-1)]
                dists_tmp = self.get_dist(res, obj_pos)
                flag = self.check_blend(res)
                if self._check_dist(dists_tmp):
                    blend_tmp = BLEND
                else:
                    if np.sum([1 if flag[i] and self._check_dist([dists_tmp[i]]) else 0 for i in range(true_nobj)], dtype=bool):
                        blend = BLEND
                    else:
                        blend_tmp = NO_BLEND
            else:
                if obj['blended'] == True:
                    obj_pos += [[obj_pos[0][0] + obj['blend_param']['dx'][i], obj_pos[0][1] + obj['blend_param']['dy'][i]] for i in range(true_nobj-1)]
                dists_tmp = self.get_dist(res, obj_pos)
                flag = self.check_blend(res)
                n_good_dist = np.sum([1 if self._check_dist([dists_tmp[i]]) else 0 for i in range(true_nobj)], dtype=int)
                if n_good_dist == true_nobj:
                    blend_tmp = BLEND
                elif n_good_dist > true_nobj:
                    blend_tmp = MISS_EXTRACTION
                elif np.sum([1 if flag[i] and self._check_dist([dists_tmp[i]]) else 0 for i in range(true_nobj)], dtype=bool):
                    blend_tmp = BLEND
                else:
                    blend_tmp = NO_BLEND

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
