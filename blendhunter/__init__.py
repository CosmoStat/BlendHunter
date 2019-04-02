# -*- coding: utf-8 -*-

"""BLENDHUNTER PACKAGE

BlendHunter implements the VGG16 CNN to identify blended galaxy images in
postage stamps.

:Author: Samuel Farrens <samuel.farrens@cea.fr>, Alexandre Bruckert

"""

__all__ = ['blend', 'data', 'network']

from .network import BlendHunter
