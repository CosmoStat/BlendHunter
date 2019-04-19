# -*- coding: utf-8 -*-

""" BLEND

This module defines classes and methods for blending images.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np
from lmfit import Model
from lmfit.models import GaussianModel, ConstantModel
from sf_tools.image.stamp import postage_stamp


class Blender(object):

    def __init__(self, images, ratio=1.0, blended=True, method='sf',
                 xwang_sigma=0.15):

        self.ratio = ratio
        self.blended = blended
        if method in ('sf', 'xwang'):
            self.method = method
        else:
            raise ValueError('Method must be "sf" or "xwang".')
        self.xwang_sigma = xwang_sigma

        if images.shape[0] % 2:
            images = images[:-1]

        half_sample = images.shape[0] // 2

        self._centrals = images[:half_sample]
        self._companions = images[half_sample:]

    @staticmethod
    def _fit_gauss(xval, yval):

        model = GaussianModel()
        result = model.fit(yval, model.guess(yval, x=xval,
                           amplitude=np.max(yval)), x=xval)

        return result

    @classmethod
    def _fit_image(cls, image):

        sum_x = image.sum(axis=0)
        sum_y = image.sum(axis=1)
        x_vals = np.arange(sum_x.size)

        sum_x_fit = cls._fit_gauss(x_vals, sum_x)
        sum_y_fit = cls._fit_gauss(x_vals, sum_y)

        centre = (sum_x_fit.params['center'].value,
                  sum_y_fit.params['center'].value)
        width = min(sum_x_fit.params['fwhm'].value,
                    sum_y_fit.params['fwhm'].value)

        return centre, width

    @staticmethod
    def _random_shift(centre, radius, outer_radius=None):

        theta = np.random.ranf() * 2 * np.pi
        if outer_radius:
            r = radius + np.random.ranf() * (outer_radius - radius)
        else:
            r = np.random.ranf() * radius
        x = int(np.around(r * np.cos(theta)))
        y = int(np.around(r * np.sin(theta)))

        return x, y

    @staticmethod
    def _get_outer_rad(width, radius):

        return np.sqrt(0.5 * width ** 2 - width * radius + radius ** 2)

    @staticmethod
    def _pad_image(image, shift):

        pad = [(_shift, 0) if _shift >= 0 else (0, -_shift)
               for _shift in shift]

        return np.pad(image, pad, 'constant')

    @classmethod
    def _blend(cls, image1, image2, shift):

        dim = image1.shape
        image2 = cls._pad_image(image2, shift)

        image2 = image2[:dim[0]] if shift[0] >= 0 else image2[-shift[0]:]
        image2 = image2[:, :dim[1]] if shift[1] >= 0 else image2[:, -shift[1]:]

        return image1 + image2

    @staticmethod
    def _gal_size_xwang(image):

        size = [np.array(np.where(np.sum(image, axis=i) != 0)).shape[1]
                for i in range(2)]
        return np.array(size)

    @classmethod
    def _blend_xwang(cls, image1, image2, buffer=5, sigma=0.15):

        shape1, shape2 = np.array(image1.shape), np.array(image2.shape)

        padding = ((shape1[0] * buffer, shape1[0] * buffer),
                   (shape1[1] * buffer, shape1[1] * buffer))

        new_image = np.pad(image1, padding, 'constant')
        new_centre = np.array(new_image.shape) // 2

        dis = cls._gal_size_xwang(image1) + cls._gal_size_xwang(image2)

        blend_pos = [np.random.randint(new_centre[i] - sigma * dis[i],
                     new_centre[i] + sigma * dis[i]) for i in range(2)]
        blend_slice = [slice(blend_pos[i] - shape2[i] // 2,
                       blend_pos[i] + shape2[i] // 2 + 1) for i in range(2)]

        new_image[blend_slice[0], blend_slice[1]] += image2

        new_image = postage_stamp(new_image, pos=new_centre,
                                  pixel_rad=shape1 // 2)

        return new_image

    def _combine_images(self, image1, image2):

        if self.method == 'xwang':

            res = self._blend_xwang(image1, image2, sigma=self.xwang_sigma)

        else:

            centre1, radius1 = self._fit_image(image1)
            centre2, radius2 = self._fit_image(image2)
            radius = self.ratio * (radius1 + radius2)
            width = image1.shape[0]

            if self.blended:
                shift = self._random_shift(centre1, radius)
            else:
                shift = self._random_shift(centre1, radius,
                                           self._get_outer_rad(width, radius))

            res = self._blend(image1, image2, shift)

        return res

    def blend(self):

        blends = [self._combine_images(image1, image2) for image1, image2 in
                  zip(self._centrals, self._companions)]

        return np.array(blends)
