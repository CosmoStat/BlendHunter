# -*- coding: utf-8 -*-

""" DATA

This module defines classes and methods for preparing training data.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import os
import numpy as np
import cv2
from blendhunter.blend import Blender


class CreateTrainData(object):
    """ Create Training Data

    This class creates prepares training data for the BlendHunter class.

    Parameters
    ----------
    images : np.ndarray
        Stack of input images
    output_path : str
        Path to where the data will be saved
    train_fractions : tuple, optional
        Fraction of training, validation and testing samples from the input
        images, default is (0.45, 0.45, .1)
    classes : tuple, optional
        Names of the various data classes, default is ('blended',
        'not_blended')
    class_fractions : tuple, optional
        Fraction of samples to be allocated to each class, the default is
        (0.5, 0.5)

    Raises
    ------
    ValueError
        If train_fractions does not have 3 elements
    ValueError
        If the train_fractions elements do not sum to 1

    """

    def __init__(self, images, output_path, train_fractions=(0.45, 0.45, 0.1),
                 classes=('blended', 'not_blended'),
                 class_fractions=(0.5, 0.5), blend_images=True,
                 blend_fractions=(0.5, 0.5), blend_method='sf'):

        self.images = images
        self.path = output_path
        if len(train_fractions) != 3:
            raise ValueError('Fractions must be a tuple of length 3.')
        if sum(train_fractions) != 1:
            raise ValueError('Fractions must sum to 1.')
        self.train_fractions = train_fractions
        self.classes = classes
        self.class_fractions = class_fractions
        self.blend_images = blend_images
        self.blend_fractions = blend_fractions
        self.blend_method = blend_method
        self._image_num = 0

        self._make_output_dirs()

        if self.blend_images:
            self._rescale_class_fractions()

    def _rescale_class_fractions(self):
        """ Rescale Class Fractions

        Adjust the class fractions to take into account blending of images.

        """

        if len(self.class_fractions) == 2:

            frac1, frac2 = self.class_fractions

            frac1 *= (frac1 + 3 * frac2 / 2)
            frac2 = 1.0 - frac1

            self.class_fractions = (frac1, frac2)

    def _make_output_dirs(self):
        """ Make Output Directories

        This method creates the directories where the samples will be stored.

        """

        bh_path = '{}/BlendHunterData'.format(self.path)
        train_path = '{}/train'.format(bh_path)
        valid_path = '{}/validation'.format(bh_path)

        if os.path.isdir(bh_path):
            raise FileExistsError('{} already exists. Please remove this '
                                  'directory or choose a new path.'
                                  ''.format(bh_path))

        os.mkdir(bh_path)
        os.mkdir(train_path)
        os.mkdir(valid_path)

        self._train_paths = ['{}/{}'.format(train_path, _class)
                             for _class in self.classes]
        self._valid_paths = ['{}/{}'.format(valid_path, _class)
                             for _class in self.classes]

        for _t_path, _v_path in zip(self._train_paths, self._valid_paths):
            os.mkdir(_t_path)
            os.mkdir(_v_path)

        if self.train_fractions[-1] > 0:
            self._test_path = '{}/test/test'.format(bh_path)
            os.mkdir('{}/test'.format(bh_path))
            os.mkdir(self._test_path)

    @staticmethod
    def _get_slices(array, fractions):
        """ Get Slices

        This method converts sample fractions into slice elemets for a given
        array.

        Parameters
        ----------
        array : np.ndarray
            Input array
        fractions : tuple
            Sample fractions

        Returns
        -------
        list
            Slice elements

        """

        frac_int = np.around(np.array(fractions) * array.shape[0]).astype(int)

        return [np.sum(frac_int[:_i]) for _i in range(1, frac_int.size + 1)]

    @classmethod
    def _split_array(cls, array, fractions):
        """ Split Array

        Split input array by the sample fractions.

        Parameters
        ----------
        array : np.ndarray
            Input array
        fractions : tuple
            Sample fractions

        Returns
        -------
        list
            List of sub-arrays

        """

        n_frac = len(fractions)
        split = np.split(array, cls._get_slices(array, fractions))

        return [split[_i] if _i < n_frac - 1 else np.vstack(split[_i:])
                for _i in range(n_frac)]

    @staticmethod
    def _rescale(array):
        """ Rescale

        Rescale input image to RGB.

        Parameters
        ----------
        array : np.ndarray
            Input array

        Returns
        -------
        np.ndarray
            Rescaled array

        """

        array = np.abs(array)

        return np.array(array * 255 / np.max(array)).astype(int)

    @staticmethod
    def _pad(array, padding):
        """ Pad

        Pad array with specified padding.

        Parameters
        ----------
        array : np.ndarray
            Input array
        padding : np.ndarray
            Padding amount

        Returns
        -------
        np.ndarray
            Padded array

        """

        x, y = padding + padding % 2

        return np.pad(array, ((x, x), (y, y)), 'constant')

    def _write_images(self, images, path):
        """ Write Images

        Write images to jpeg files.

        Parameters
        ----------
        images : np.ndarray
            Array of images
        path : str
            Path where images should be written

        """

        min_shape = np.array([48, 48])

        for image in images:

            image = self._rescale(image)

            shape_diff = (min_shape - np.array(image.shape))[:2]

            if np.sum(shape_diff) > 0:
                image = self._pad(image, shape_diff)

            cv2.imwrite('{}/image_{}.png'.format(path, self._image_num), image)
            self._image_num += 1

    def _write_data_set(self, data_list, path_list):
        """ Write Data Set

        Write input data set to corresponding paths.

        Parameters
        ----------
        data_list : list
            List of image arrays
        path_list : list
            List of paths

        """

        for data, path in zip(data_list, path_list):
            self._write_images(data, path)

    def _write_labels(self, data_list):
        """ Write Labels

        Write test data labels to a numpy binary.

        Parameters
        ----------
        data_list : list
            List of image arrays

        """

        sizes = [array.shape[0] for array in data_list]

        labels = np.array([[self.classes[0]] * sizes[0] +
                           [self.classes[1]] * sizes[1]])

        np.save('{}/labels.npy'.format(self._test_path), labels)

    def _write_psf(obj):
        """ Write PSF

        Save test psf to a numpy array."""

        output = '/Users/alacan/Documents/Cosmostat/Codes/BlendHunter/bh/BlendHunterData/test/test'

        np.save(output+'/test_psf.npy', test_psf)

    def _write_positions(self, pos_list):

        np.save('{}/positions.npy'.format(self._test_path), np.array(pos_list))

    def _blend_data(self, data_set):
        """ Blend Data Set

        Blend the first sample of the data set and combine the other set
        without blending.

        Parameters
        ----------
        data_set : list
            List of data samples

        Returns
        -------
        list
            Blended data set

        """

        if len(data_set) == 2:

            # Blend overlapping images
            blended = Blender(data_set[0], ratio=0.8, method=self.blend_method)
            data_set[0] = blended.blend()

            # Blend non-overlapping images and pad isolated objects
            no_blend_1, no_blend_2 = (self._split_array(data_set[1],
                                      self.blend_fractions))

            not_blended_1 = Blender(no_blend_1)
            no_blend_1 = not_blended_1.pad()

            not_blended_2 = Blender(no_blend_2, ratio=1.5, overlap=False,
                                    method=self.blend_method,
                                    xwang_sigma=1.0)
            no_blend_2 = not_blended_2.blend()

            data_set[1] = np.vstack((no_blend_1, no_blend_2))

            # Save object positions
            positions = []
            for sample in (blended, not_blended_1, not_blended_2):
                positions.extend(sample.obj_centres)

        return data_set, positions

    def generate(self):
        """ Generate

        Generate training data.

        """

        self.images = np.random.permutation(self.images)

        image_split = self._split_array(self.images, self.train_fractions)

        train_set = self._split_array(image_split[0], self.class_fractions)
        valid_set = self._split_array(image_split[1], self.class_fractions)

        if self.blend_images:
            train_set, train_pos = self._blend_data(train_set)
            valid_set, valid_pos = self._blend_data(valid_set)

        self._write_data_set(train_set, self._train_paths)
        self._write_data_set(valid_set, self._valid_paths)

        if self.train_fractions[-1] > 0:
            frac1 = np.random.choice(np.arange(1, 5) * 0.2)
            frac2 = 1.0 - frac1
            test_set = self._split_array(image_split[2], (frac1, frac2))
            if self.blend_images:
                test_set, test_pos = self._blend_data(test_set)
                self._write_labels(test_set)
                self._write_positions(test_pos)
                test_set = np.vstack(test_set)
            self._write_images(test_set, self._test_path)

    def prep_axel(self, path_to_output, psf, param_1, param_2, map):

        #self.images[0] = np.random.permutation(self.images[0])
        #self.images[1] = np.random.permutation(self.images[1])

        split1 = self._split_array(self.images[0], self.train_fractions)
        split2 = self._split_array(self.images[1], self.train_fractions)

        #Split fwhm
        train_fractions=(0.45, 0.45, 0.1)
        psf_split1 = CreateTrainData._split_array(psf[0], train_fractions)
        psf_split2 = CreateTrainData._split_array(psf[1], train_fractions)

        #Split shift params
        x_split = CreateTrainData._split_array(param_1[0], train_fractions)
        y_split = CreateTrainData._split_array(param_2[0], train_fractions)

        #Split segmentation map
        map_split = CreateTrainData._split_array(map[0], train_fractions)

        train_set = split1[0], split2[0]
        valid_set = split1[1], split2[1]
        test_set = split1[2], split2[2]

        test_psf = psf_split1[2], psf_split2[2]
        test_param_x = x_split[2]
        test_param_y = y_split[2]
        test_im_blended = split1[2] #blended test images
        test_im_nb = split2[2]
        test_map = map_split[2]

        self._write_data_set(train_set, self._train_paths)
        self._write_data_set(valid_set, self._valid_paths)
        self._write_labels(test_set)
        self._write_images(np.vstack(test_set), self._test_path)

        #Save test_psf
        #_write_psf(test_psf)
        #output = '/Users/alacan/Documents/Cosmostat/Codes/BlendHunter/bh/BlendHunterData/test/test'
        np.save(path_to_output+'/test_psf.npy', test_psf)

        #Save test_params
        np.save(path_to_output+'/test_param_x.npy', test_param_x)
        np.save(path_to_output+'/test_param_y.npy', test_param_y)

        #Save blended test images
        np.save(path_to_output+'/gal_im_blended.npy', test_im_blended)
        np.save(path_to_output+'/gal_im_nb.npy', test_im_nb)
        np.save(path_to_output+'/test_images.npy', test_set)

        #Save seg_map
        np.save(path_to_output+'/test_seg_map.npy', test_map)
