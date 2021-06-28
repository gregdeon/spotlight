#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
readme = ""


requirements = [
    "torch",
    "tqdm",
    "pandas",
    "numpy",
    "scipy",
]

optional_packages = {
}

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='exchtensor',
    version='0.1.0',
    description="Implementation of deep models of interactions across sets",
    long_description=readme,# + '\n\n' + history,
    author="Jason Hartford",
    author_email='jasonhar@cs.ubc.ca',
    url='https://arxiv.org/abs/1803.02879',
    packages=[
        'exchtensor',
    ],
    package_dir={'exchtensor':
                 'exchangable_tensor'},
    include_package_data=True,
    install_requires=requirements,
    extras_require=optional_packages,
    license="MIT license",
    zip_safe=False,
    keywords='exchtensor'
)
