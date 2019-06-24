# -*- coding: utf-8 -*-

""" BLEND

This module defines classes and methods for blending images.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np
from lmfit import Model
from lmfit.models import GaussianModel, ConstantModel
from modopt.base.np_adjust import pad2d
from sf_tools.image.stamp import postage_stamp
from sf_tools.image.distort import recentre


class Blender(object):

    def __init__(self, images, ratio=1.0, overlap=True, stamp_shape=(116, 116),
                 method='sf', xwang_sigma=0.15, seed=None):

        self.ratio = ratio
        self.overlap = overlap
        self.stamp_shape = np.array(stamp_shape)
        if method in ('sf', 'xwang'):
            self.method = method
        else:
            raise ValueError('Method must be "sf" or "xwang".')
        self.xwang_sigma = xwang_sigma
        self.seed = seed

        if images.shape[0] % 2:
            images = images[:-1]

        half_sample = images.shape[0] // 2

        self._images = images
        self._centrals = images[:half_sample]
        self._companions = images[half_sample:]
        self.obj_centres = []

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

        centre = (int(sum_x_fit.params['center'].value),
                  int(sum_y_fit.params['center'].value))
        width = min(sum_x_fit.params['fwhm'].value,
                    sum_y_fit.params['fwhm'].value)

        return centre, width

    @staticmethod
    def _random_shift(radius, outer_radius=None, seed=None):

        if seed:
            np.random.seed(seed)

        theta = np.random.ranf() * 2 * np.pi
        if outer_radius:
            r = radius + np.random.ranf() * (outer_radius - radius)
        else:
            r = np.random.ranf() * radius
        x = int(np.around(r * np.cos(theta)))
        y = int(np.around(r * np.sin(theta)))

        return x, y

    @staticmethod
    def _pad_image_shift(image, shift):

        pad = [(_shift, 0) if _shift >= 0 else (0, -_shift)
               for _shift in shift]

        return np.pad(image, pad, 'constant')

    @classmethod
    def _blend(cls, image1, image2, shift):

        dim = image1.shape
        image2 = cls._pad_image_shift(image2, shift)

        image2 = image2[:dim[0]] if shift[0] >= 0 else image2[-shift[0]:]
        image2 = image2[:, :dim[1]] if shift[1] >= 0 else image2[:, -shift[1]:]

        return image1 + image2

    @staticmethod
    def _gal_size_xwang(image):

        return np.array([np.count_nonzero(image.sum(axis=ax))
                         for ax in range(2)])

    @staticmethod
    def _area_prob(shape1, shape2):

        shape1, shape2 = np.array(shape1), np.array(shape2)

        area = np.prod(shape1) - np.prod(shape2)
        shape_diff = (shape1 - shape2) // 2
        prob_ab = shape_diff[1] * shape1[0] / area
        prob_cd = 0.5 - prob_ab

        return prob_ab, prob_ab, prob_cd, prob_cd

    @classmethod
    def _blend_pos_xwang(cls, centre, box, limits, overlap=True):

        centre, box, limits = np.array(centre), np.array(box), np.array(limits)

        if overlap:
            blend_pos = [np.random.randint(centre[i] - box[i],
                         centre[i] + box[i]) for i in range(2)]
        else:
            sector = np.random.choice(['a', 'b', 'c', 'd'],
                                      p=cls.area_prob(centre * 2, box))
            blend_pos = [None, None]
            if sector == 'a':
                blend_pos[0] = np.random.randint(limits[0][0], limits[1][0])
                blend_pos[1] = np.random.randint(limits[0][1],
                                                 centre[1] - box[1])
            elif sector == 'b':
                blend_pos[0] = np.random.randint(limits[0][0], limits[1][0])
                blend_pos[1] = np.random.randint(centre[1] + box[1],
                                                 limits[1][1])
            elif sector == 'c':
                blend_pos[0] = np.random.randint(limits[0][0],
                                                 centre[0] - box[0])
                blend_pos[1] = np.random.randint(centre[1] - box[1],
                                                 centre[1] + box[1])
            elif sector == 'd':
                blend_pos[0] = np.random.randint(centre[0] + box[0],
                                                 limits[1][1])
                blend_pos[1] = np.random.randint(centre[1] - box[1],
                                                 centre[1] + box[1])

        return blend_pos

    @classmethod
    def _blend_xwang(cls, image1, image2, ps_shape=(116, 116), sigma=0.15,
                     overlap=True):

        shape1, shape2 = np.array(image1.shape), np.array(image2.shape)
        rad2 = shape2 // 2
        ps_shape = np.array(ps_shape)

        shape_diff = (ps_shape - shape1) // 2 + shape2

        dis = cls._gal_size_xwang(image1) + cls._gal_size_xwang(image2)
        box = np.around(sigma * dis).astype(int)

        padding = ((shape_diff[0], shape_diff[0]),
                   (shape_diff[1], shape_diff[1]))

        new_image = np.pad(image1, padding, 'constant')
        new_shape = np.array(new_image.shape)
        new_centre = new_shape // 2

        limits = rad2, new_shape - rad2

        bp = cls._blend_pos_xwang(new_centre, box, limits, overlap=True)

        blend_slice = [slice(bp[i] - shape2[i] // 2,
                       bp[i] + shape2[i] // 2 + 1) for i in range(2)]

        new_image[blend_slice[0], blend_slice[1]] += image2

        new_image = postage_stamp(new_image, pos=new_centre,
                                  pixel_rad=ps_shape // 2)

        return new_image

    def _pad_image(self, image):

        if not isinstance(image, np.ndarray):
            print(type(image))

        im_shape = np.array(image.shape)
        padding = (self.stamp_shape - im_shape) // 2

        return pad2d(image, padding)

    def _combine_images(self, image1, image2):

        if self.method == 'xwang':

            res = self._blend_xwang(image1, image2, ps_shape=self.stamp_shape,
                                    sigma=self.xwang_sigma,
                                    overlap=self.overlap)

        else:

            centre1, width1 = self._fit_image(image1)
            centre2, width2 = self._fit_image(image2)

            image1 = self._pad_image(recentre(image1, centre1))
            image2 = self._pad_image(recentre(image2, centre2))

            radius = self.ratio * (width1 + width2)
            outer_radius = image1.shape[0] / 2.

            if self.overlap:
                shift = self._random_shift(radius, seed=self.seed)
            else:
                shift = self._random_shift(radius, outer_radius=outer_radius,
                                           seed=self.seed)

            im1_cen = np.array(image1.shape) // 2
            im2_cen = np.copy(im1_cen) + np.array(shift)[::-1]

            self.obj_centres.append((tuple(im1_cen), tuple(im2_cen)))

            res = self._blend(image1, image2, shift)

        return res

    def blend(self):

        blends = [self._combine_images(image1, image2) for image1, image2 in
                  zip(self._centrals, self._companions)]

        return np.array(blends)

    def pad(self):

        return np.array([self._pad_image(image) for image in self._images])
