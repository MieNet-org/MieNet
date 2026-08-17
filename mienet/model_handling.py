""" Model handling functionalities """
# pylint: disable=C0415,R0902,R0912,R0914,R0915

import requests
from zipfile import ZipFile
from io import BytesIO
import os
import shutil
import numpy as np
import xarray as xr

def get_models(data_location):
    '''
    Download and unzip AI models from Zenodo
    '''
    # Zenodo link
    url = 'https://zenodo.org/records/20346256/files/models.zip?download=1'

    # download and unzip folder from Zenodo
    os.makedirs(data_location, exist_ok=True)
    r = requests.get(url)
    ZipFile(BytesIO(r.content)).extractall(data_location)

    # move files out of models folder
    models_folder = os.path.join(data_location, 'models')
    for f in os.listdir(models_folder):
        shutil.move(os.path.join(models_folder, f), data_location)
    shutil.rmtree(models_folder)

    # delete MACOSX folder
    shutil.rmtree(os.path.join(data_location, '__MACOSX'))

    return data_location

def generate_training_set(self, file_name, species, wavelength_sample, particle_size_sample, mixing_theory = 'LLL'):
    """
    Generate a neural network training set.

    Parameters
    ----------
    file_name : str
        Name for the training set file
    species : list[str]
        List of species to include in the training set
    wavelength_range : tuple
        (wavelength_min, wavelength_max, number_of_wavelengths)
    particle_size_range : tuple
        (particle_size_min, particle_size_max, number_of_particle_sizes)
    mixing_theory : str (optional)
        Mixing theory used, can either be 'LLL' (Default) or 'Bruggeman'
    """
    # ==== Setup and Checks ============================================================================================
    store_path = self.model_path + file_name + '.nc'
    wave_points = wavelength_sample[2] # number of wavelength points
    radii_points = particle_size_sample[2] # number of particle size points
    set_size = wave_points * radii_points # total training set size
    num_params = 5 + (len(species) - 1) # number of parameters in training set

    # check if training set exists
    if not os.path.exists(store_path):
        ds = xr.Dataset(
            data_vars = {'data': (['idx', 'dim'], np.zeros((set_size, num_params)))},
            coords = {
                'idx': range(set_size),
                'dim': ['wavelength', 'particle_size'] + species[:-1] + ['extinction', 'scattering', 'asymmetry']
            },
            attrs = {'idx': 0}
        )
        ds.to_netcdf(store_path)

    else:
        # load training set
        ds = xr.load_dataset(store_path)

    # check if training set is finished generating
    if ds.attrs['idx'] >= set_size:
        raise ValueError('Training set generation is complete, no more space in training set:' + store_path)

    # ==== Calculations ================================================================================================
    ma = MieNet() # initialize MieNet

    # create particle size sample
    particle_size_sample = np.logspace(particle_size_range[0], particle_size_range[1], radii_points) # radius sample, fixed grid

    while ds.attrs['idx'] < set_size:

        # generate volume mixing ratios
        alpha = [0.5] * len(species) # controls distribution
        N = 1 # size of wavelength and volume mixing ratios
        ratio_samples = np.random.dirichlet(alpha, size = N)
        a0 = np.zeros(radii_points) # zero array so vmr and radius_sample are same length
        ratio_dict = {} # prepare vmr dictionary
        for i, item in enumerate(species):
            ratio_dict[item] = a0 + ratio_samples[0, i]

        # create wavelength sample
        wavelength_sample = np.random.uniform(wavelength_range[0], wavelength_range[1], size = N)

        # calculate outputs
        extinction, scattering, asymmetry = \
            ma.efficiencies(wavelength_sample, particle_size_sample, ratio_dict, theory = mixing_theory)

        # ==== Save training set =======================================================================================

        # xarray inputs
        sol = np.zeros((radii_points, num_params))
        sol[:, 0] = wavelength_sample
        sol[:, 1] = particle_size_sample
        for i, item in enumerate(species[:-1]):
            sol[:, 2 + i] = a0 + ratio_samples[0, i]
        sol[:, -3] = extinction[:,0]
        sol[:, -2] = scattering[:,0]
        sol[:, -1] = asymmetry[:,0]

        # save data
        ds['data'][ds.attrs['idx']:ds.attrs['idx'] + radii_points] = sol
        ds.attrs['idx'] += radii_points
        ds.to_netcdf(store_path)