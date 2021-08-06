# -*- coding: utf-8 -*-


from __future__ import absolute_import, print_function

from setuptools import setup, find_packages
import sys

requirements = [
        'cobra',
        'numpy',
        'six',
        'scipy',
        'symengine',
        'matplotlib',
        'sklearn',
        'statsmodels']

setup(
    name='dexom_python',
    version="0.1",
    packages=find_packages('.'),
    install_requires=requirements,
    include_package_data=True,
    author='Maximilian Stingl',
    author_email='maximilian.ha.stingl@gmail.com',
    description='Python implementation of DEXOM',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Utilities',
        'Programming Language :: Python :: 3.7'
    ],
)
sys.path.append('dexom_python')
