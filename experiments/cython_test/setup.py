from setuptools import setup
from Cython.Build import cythonize

setup(
    name='aggregator',
    ext_modules=cythonize("aggregator.pyx"),
    zip_safe=False,
)