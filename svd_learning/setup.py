from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='svdcy',
    ext_modules=cythonize('svdcy.pyx'),
    zip_safe=False,
    include_dirs=[np.get_include()]
)
