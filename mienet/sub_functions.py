""" General functionalities """
# pylint: disable=C0415,R0902,R0912,R0914,R0915

import os
import pandas as pd
import numpy as np
import yaml

def read_in_refindex(species, wavelength, files):
    """
    Read in and interpolate refractive index files.

    Parameters
    ----------
    species : List with size N
        Name of cloud species.
        wavelength : np.ndarray or float of size M
            Wavelength of the light [micron]
    files : List
        Refractive index files

    Return
    ------
    ref_index : np.ndarray of size (N, M, 2)
        Refractive index models: real, and imaginary part.
    """

    # prepare output
    ref_index = np.zeros((len(species), len(wavelength), 2))
    for s, spec in enumerate(species):

        # ==== Load models from files =====================================================

        # find species in files
        data = None
        for file in files:
            if spec in file:
                # get models using pandas
                content = pd.read_csv(file, sep=r'\s+', header=None, usecols=[1, 2, 3])
                # convert to array and flip vertically so wavelength increases
                data = np.flip(content.to_numpy(), axis=0)
        if data is None:
            raise ValueError('No refindex file found for ' + spec)

        # ==== Get the real(n) and imaginary (k) refractory index =======================
        # prepare output
        if not isinstance(wavelength, np.ndarray):
            wavelength = np.asarray([wavelength])

        # loop over all wavelengths
        for wav, wave in enumerate(wavelength):
            # if desired wavelength is smaller than models, use the smallest wavelength
            # models available
            if wave < float(data[0, 0]):
                ref_index[s, wav, 0] = float(data[0, 1])
                ref_index[s, wav, 1] = float(data[0, 2])
                continue

            # if wavelength is within range log-log interpolation
            for dnr, _ in enumerate(data):
                cur_wave = float(data[dnr, 0])  # current wavelength
                if wave < cur_wave:
                    nlo = float(data[dnr - 1, 1])  # lower n value
                    nhi = float(data[dnr, 1])  # higher n value
                    klo = float(data[dnr - 1, 2])  # lower k value
                    khi = float(data[dnr, 2])  # higher k value
                    prev_wave = float(data[dnr - 1, 0])  # previous wavelength
                    # calculate interpolation
                    fac = np.log(wave / prev_wave) / np.log(cur_wave / prev_wave)
                    ref_index[s, wav, 0] = np.exp(np.log(nlo) + fac * np.log(nhi / nlo))
                    if klo <= 0 or khi <= 0:
                        ref_index[s, wav, 1] = 0
                    else:
                        ref_index[s, wav, 1] = np.exp(np.log(klo) + fac * np.log(khi / klo))

                    break

            else:
                # if wavelength is out of range, extrapolate
                # non-conducting interpolation, linear decreasing k, constant n
                ref_index[s, wav, 0] = float(data[-1, 1])
                ref_index[s, wav, 1] = float(data[-1, 2]) * float(data[-1, 0]) / wave

    return ref_index


def calculate_subradii(particle_size, vmr):
    """
    Calculate subgrid for each radius.

    Parameters
    ----------
    particle_size : np.ndarray or float of size M
        Size of the cloud particle [micron]
    vmr : ndarray
        Fraction of each cloud material

    Return
    ------
    sub_rad, vmr : (ndarray(M*6), ndarray)
        Sub-spacing of radii and adjusted vmr.
    """
    if len(particle_size) > 1:
        if len(set(particle_size)) != 1:
            # prepare outputs
            rad_min = np.zeros_like(particle_size)
            rad_max = np.zeros_like(particle_size)
            mid_points = (particle_size[1:] + particle_size[:-1]) / 2

            # radius minimum and maximum from midpoints
            rad_min[1:] = mid_points
            # smallest radius value >0
            rad_min[0] = np.max([particle_size[0] - mid_points[0], 0])
            rad_max[:-1] = mid_points
            rad_max[-1] = particle_size[-1] + mid_points[-1]

            # prepare output
            sub_rad = np.zeros((len(particle_size) * 6))
            i = 0  # index

            for r_max, r_min in zip(rad_max, rad_min):
                # six radius points to average over
                r = (r_max - r_min) / 6
                rad_range = r_min + np.array([r, 2 * r, 3 * r, 4 * r, 5 * r, 6 * r])
                sub_rad[i:i + 6] = rad_range
                # index
                i += 6
            # make volume mixing ratios the same size as particle size
            vmr = np.repeat(vmr, 6, axis=0)

        else:
            sub_rad = np.zeros((len(particle_size) * 6))
            i = 0
            for rad in particle_size:
                sub_rad[i:i + 6] = np.linspace(rad * 0.7, rad * 1.3, 6)
                i += 6
            vmr = np.repeat(vmr, 6, axis=0)

    else:
        sub_rad = particle_size

    return sub_rad, vmr

def get_model_info(model_path):
    '''
    Get neural network files and information.

    Parameters
    ----------
    model_path: string

    Returns
    -------
    Models dictionary containing:
        List of model files
        List of species
        Name of Architecture function
        Dictionary of dependencies

    '''
    config_yaml = model_path + 'config.yaml'
    with open(config_yaml, 'r') as f:
        config = yaml.safe_load(f)

    models_dict = {}

    for model in config.keys():
        model_info = config[model]
        models_dict[model] = {}
        models_dict[model]['files'] = model_info['files']
        models_dict[model]['species'] = model_info['species']
        models_dict[model]['architecture'] = model_info['architecture']
        models_dict[model]['dependencies'] = model_info['dependencies']
        models_dict[model]['range'] = model_info['range']
        models_dict[model]['scale'] = model_info['scale']

    return models_dict

def initialize_ai_models(load_ai_model, model_path):
    '''
    Load ai tensorflow models.

    Parameters
    ----------
    load_model : String
        Either 'all' to load every model or the model name
    model_names: Dictionary
        Dictionary of species in each mixture

    Returns
    ----------
    Models dictionary containing:
        List of model files
        List of species
        List of Tensorflow models
        Name of Architecture function
        Dictionary of dependencies
    '''
    from tensorflow.keras.models import load_model

    # read  config file
    models_dict = get_model_info(model_path)

    # load all models by default if one is not specified
    if load_ai_model == 'all':

        # load all models for each mixture
        for model in models_dict.keys():

            # prepare model list for dictionary
            model_list = np.empty(len(models_dict[model]['files']), dtype = object)

            for i, file in enumerate(models_dict[model]['files']):

                # only load files that are downloaded
                if os.path.isfile(os.path.join(model_path + file)):
                    model_list[i] = load_model(os.path.join(model_path + file))

            models_dict[model]['models'] = model_list

            print(f'[INFO] Loaded {model} model for', models_dict[model]['species'],
                  f'from {models_dict[model]['range']['wavelength'][0]} to '
                  f'{models_dict[model]['range']['wavelength'][1]} micron.')

    # load specified model
    else:

        # check if specified model is available
        if load_ai_model not in models_dict.keys():
            raise ValueError('[ERROR] The model "' + str(load_ai_model) +
                             '" is not in the config.yaml file.')

        # save only desired mixture in the models dictionary
        models_dict = models_dict[load_ai_model]

        model_list = np.empty(len(models_dict['files']), dtype = object)

        for i, file in enumerate(models_dict['files']):

            # load files
            if os.path.isfile(os.path.join(model_path + file)):
                model_list[i] = load_model(os.path.join(model_path + file))

            else:
                raise ValueError(f'Model files for mixture {load_ai_model} not found.')

        models_dict['models'] = model_list

        print(f'[INFO] Loaded {load_ai_model} model for', models_dict['species'],
              f'from {models_dict['range']['wavelength'][0]} to '
              f'{models_dict['range']['wavelength'][1]} micron.')

    return models_dict

def select_best_dataset(type, auto, vmrs, datasets):
    '''
    Choose best grid or model.

    Parameters
    ----------
    type : String
        Either 'model' or 'grid'
    auto: Boolean
        True if using auto_efficiencies function for fastest calculation.
    volume_mixing_ratios: Dictionary
        Species in each mixture with respective volume mixing ratios.
    datasets: Dictionary
        Dictionary with information about each model/grid.

    Returns
    ----------
    best_dataset : tuple, (name, species)
        name: str of model/grid name
        species: list of species name
    '''
    # find all dataset that include all species
    l_set = set(vmrs.keys())
    valid_datasets = {
        name: data['species'] for name, data in datasets.items()
        if l_set.issubset(data['species'])
    }

    # check if there are matching datasets
    if valid_datasets:

        # Now pick the dataset with the smallest total size
        best_dataset = min(valid_datasets.items(), key=lambda item: len(item[1]))

    else:
        if auto == 'False':
            # raise value error if no datasets for the type of efficiency function called
            raise ValueError("No default" + f'{type}' + "for" + str(l_set) +
                                 " is available. Please provide one.")

        else:
            best_dataset = (None, None)

    return best_dataset