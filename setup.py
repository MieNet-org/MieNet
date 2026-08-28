""" set up file """
from setuptools import setup, find_packages

requirements = ['numpy',
                'xarray',
                'miepython',
                'scipy',
                'matplotlib',
                'pyyaml',
                'h5netcdf']

setup(
    name='mienet',
    version='v0.1',
    packages=find_packages(),
    include_package_data=True,
    url='',
    author='Daisy Attaway',
    description='machine learning mie calculations',
    install_requires=requirements
)
