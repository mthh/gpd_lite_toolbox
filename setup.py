# -*- coding: utf-8 -*-
"""
gpd_lite_toolboox setup file
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import gpd_lite_toolbox


ext = Extension("cycartogram", ["gpd_lite_toolbox/cycartogram.pyx"])

setup(
    name='gpd_lite_toolbox',
    version=gpd_lite_toolbox.__version__,
    description='Convenience functions acting on GeoDataFrames',
    author='mthh',
    ext_modules=cythonize(ext),
    packages=['gpd_lite_toolbox'],
    license='MIT',
    )
