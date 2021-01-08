#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# with open('requirements.txt') as open_file:
#     install_requires = open_file.read()

setup(
    name='BlendHunter',
    author='sfarrens',
    author_email='samuel.farrens@cea.fr',
    version='0.0.0',
    url='https://github.com/sfarrens/BlendHunter',
    download_url='https://github.com/sfarrens/BlendHunter',
    packages=find_packages(),
    install_requires=[],
    license='MIT',
    description=(
        'Deep learning tool for identifying blended galaxy images in survey ' +
        'images.',
    )
)
