# -*- coding: utf-8 -*-
"""
gpd_lite_toolboox setup file
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='gpd_lite_toolbox',
    version='0.0.0',
    description='Convenience functions acting on GeoDataFrames',
    author='mthh',
    py_modules=['gpd_lite_toolbox'],
    ext_modules=cythonize("cycartogram.pyx"),
    license='MIT',
    )
