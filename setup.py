# -*- coding: utf-8 -*-


from __future__ import absolute_import, print_function

from setuptools import setup, find_packages
import sys

requirements = [
        'cobra',
        'numpy==1.20',
        'six',
        'scipy==1.7',
        'symengine',
        'matplotlib',
        'sklearn',
        'statsmodels']

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='dexom_python',
    version='0.4.2',
    packages=find_packages('.'),
    install_requires=requirements,
    include_package_data=True,
    author='Maximilian Stingl',
    author_email='maximilian.h.a.stingl@gmail.com',
    description='Python implementation of DEXOM using cobrapy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Metexplore/dexom-python',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Utilities',
        'Programming Language :: Python :: 3.7'
    ],
    python_requires=">=3.7,<3.10",
)
sys.path.append('dexom_python')
