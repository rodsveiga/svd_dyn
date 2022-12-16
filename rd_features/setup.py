from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='rfcy',
    ext_modules=cythonize('rfcy.pyx'),
    zip_safe=False,
    include_dirs=[np.get_include()]
)
