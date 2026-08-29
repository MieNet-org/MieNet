""" Model handling functionalities """
# pylint: disable=C0415,R0902,R0912,R0914,R0915

import requests
from zipfile import ZipFile
from io import BytesIO
import os
import shutil
import yaml
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

def get_models(data_location, overwrite=True):
    """
    Download and unzip AI models from Zenodo

    Parameters
    ----------
    data_location : str
        location where the data should be stored
    overwrite : bool, optional
        In case some files already exist: overwrite old data if True, discard new data if False
    """
    # Zenodo link
    url = 'https://zenodo.org/records/22147944/files/models.zip?download=1'

    # download and unzip folder from Zenodo
    os.makedirs(data_location, exist_ok=True)
    r = requests.get(url)
    ZipFile(BytesIO(r.content)).extractall(data_location)

    # move files out of models folder
    models_folder = os.path.join(data_location, 'models')
    for f in os.listdir(models_folder):
        # move and overwirte old files
        if overwrite:
            shutil.move(os.path.join(models_folder, f), os.path.join(data_location, f))
        # move and keep old files
        else:
            if not os.path.exists(os.path.join(data_location, f)):
                shutil.move(os.path.join(models_folder, f), data_location)

    shutil.rmtree(models_folder)

    # delete MACOSX folder
    shutil.rmtree(os.path.join(data_location, '__MACOSX'))

def initialize_ai_models(self):
    """
    Load ai tensorflow models and store them in self.models_dict.
    """
    # import tensorflow here, so MieNet can be used without it
    from tensorflow.keras.models import load_model

    # default value
    self.models_dict = {}

    # check if config.yaml exists
    if not os.path.isfile(self.data_path + 'config.yaml'):
        self.use_ai = False
        self.force_disabled_ai = True
        if not self.mute:
            print('[WARN] config.yaml file not found in the folder:\n'
                  '   -> ' + self.data_path)
            print('[WARN] No ANN models found, disabling ANN functionalities.')
        return

    # read config file
    config_yaml = self.data_path + 'config.yaml'
    with open(config_yaml, 'r') as f:
        config = yaml.safe_load(f)

    # check if there is at least one entry
    if config is None:
        self.use_ai = False
        self.force_disabled_ai = True
        if not self.mute:
            print('[WARN] No Entry found in config.yaml file.')
            print('[WARN] No ANN models found, disabling ANN functionalities.')
        return

    # create and store info in models dictionary
    models_dict = {}
    for model in config.keys():
        model_info = config[model]
        models_dict[model] = {}
        models_dict[model]['files'] = model_info['files']
        models_dict[model]['species'] = model_info['species']
        models_dict[model]['architecture'] = model_info['architecture']
        models_dict[model]['theory'] = model_info['theory']
        models_dict[model]['dependencies'] = model_info['dependencies']
        models_dict[model]['range'] = model_info['range']
        models_dict[model]['scale'] = model_info['scale']

    # load all models by default if one is not specified
    if self.load_ai_model == 'all':
        # get all model keys
        keys = list(models_dict.keys())

        # load all models for each mixture
        for model in keys:
            skip = False  # if not all keras file could be loaded, skip

            # prepare model list for dictionary
            model_list = np.empty(len(models_dict[model]['files']), dtype = object)

            for i, file in enumerate(models_dict[model]['files']):

                # only load files that are downloaded
                if os.path.isfile(os.path.join(self.data_path + file)):
                    model_list[i] = load_model(os.path.join(self.data_path + file))
                else:
                    del models_dict[model]  # remove model from dict
                    skip = True  # remember that this model is incomplete
                    break  # stop searching for more models

            # if not all keras file could be loaded, skip
            if skip:
                continue

            models_dict[model]['models'] = model_list

            if not self.mute:
                print(f'[INFO] Loaded {model} model for', models_dict[model]['species'],
                      f'from {models_dict[model]['range']['wavelength'][0]} to '
                      f'{models_dict[model]['range']['wavelength'][1]} micron.')

    # load specified model
    else:
        # convert to list if only one ai model specified
        lam = self.load_ai_model

        for model in lam:
            # check if specified model is available
            if model not in models_dict.keys():
                raise ValueError('[ERROR] The model "' + str(model) +
                                 '" is not in the config.yaml file.')

            # save only desired mixtures in the models dictionary
            loaded_models = {}
            loaded_models[model] = models_dict[model]

            model_list = np.empty(len(models_dict[model]['files']), dtype = object)

            for i, file in enumerate(models_dict[model]['files']):

                # load files
                if os.path.isfile(os.path.join(self.data_path + file)):
                    model_list[i] = load_model(os.path.join(self.data_path + file))
                else:
                    raise ValueError(f'Model files for mixture {model} not found.')

            loaded_models[model]['models'] = model_list

            if not self.mute:
                print(f'[INFO] Loaded {model} model for', models_dict[model]['species'],
                      f'from {models_dict[model]['range']['wavelength'][0]} to '
                      f'{models_dict[model]['range']['wavelength'][1]} micron.')

            # save only loaded models
            models_dict = loaded_models

    # remember models
    self.models_dict = models_dict

    # if no models were found, disable AI
    if len(self.models_dict) < 1:
        self.use_ai = False
        self.force_disabled_ai = True
        if not self.mute:
            print('[WARN] No ANN models found, disabling ANN functionalities.')

def generate_training_set(self, file_name, species, wavelength_sample, particle_size_sample, mixing_theory = 'LLL'):
    """
    Generate a neural network training set. The set will be saved as a xarray.Dataset, and will be formatted as:
    (wavelength, particle_size, vmr1, vmr2, ..., extinction, scattering, asymmetry).
    The first use of this function will create the xarray dataset and start generating and saving the data. Further
    uses of this function with the same input parameters will continue generating and saving data in the same xarray
    dataset until the dataset is full.

    Parameters
    ----------
    file_name : str
        Name for the training set file
    species : list[str]
        List of species to include in the training set
    wavelength_sample : tuple
        (wavelength_min, wavelength_max, number_of_wavelengths)
    particle_size_sample : tuple
        (particle_size_min, particle_size_max, number_of_particle_sizes)
    mixing_theory : str (optional)
        Mixing theory used, can either be 'LLL' (Default) or 'Bruggeman'
    """
    # ==== Setup and Checks ============================================================================================
    store_path = self.data_path + file_name + '.nc'
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
            attrs = {'idx': 0, 'species': species}
        )
        ds.to_netcdf(store_path)

    else:
        # load training set
        ds = xr.load_dataset(store_path)

    # check if training set is finished generating
    if ds.attrs['idx'] >= set_size:
        raise ValueError('Training set generation is complete, no more space in training set:' + store_path +
                         ' Please provide a new filename to generate more data.')

    # ==== Calculations ================================================================================================
   # create particle size sample from a fixed grid
    particle_size_range = np.logspace(particle_size_sample[0], particle_size_sample[1], radii_points)

    while ds.attrs['idx'] < set_size:

        # generate volume mixing ratios
        alpha = [0.5] * len(species) # controls distribution
        N = 1 # size of wavelength and volume mixing ratios
        ratio_samples = np.random.dirichlet(alpha, size = N)
        a0 = np.zeros(radii_points) # zero array so vmr and radius_range are same length
        ratio_dict = {} # prepare vmr dictionary
        for i, item in enumerate(species):
            ratio_dict[item] = a0 + ratio_samples[0, i]

        # create wavelength sample
        wavelength_range = np.random.uniform(wavelength_sample[0], wavelength_sample[1], size = N)

        # calculate outputs
        extinction, scattering, asymmetry = \
            self.efficiencies(wavelength_range, particle_size_range, ratio_dict, theory = mixing_theory)

        # ==== Save training set =======================================================================================

        # xarray inputs
        sol = np.zeros((radii_points, num_params))
        sol[:, 0] = wavelength_range
        sol[:, 1] = particle_size_range
        for i, item in enumerate(species[:-1]):
            sol[:, 2 + i] = a0 + ratio_samples[0, i]
        sol[:, -3] = extinction[:,0]
        sol[:, -2] = scattering[:,0]
        sol[:, -1] = asymmetry[:,0]

        # save data
        ds['data'][ds.attrs['idx']:ds.attrs['idx'] + radii_points] = sol
        ds.attrs['idx'] += radii_points
        ds.to_netcdf(store_path)

def train_ai_model(self, training_set, model_params, plot_training = False):
    """
    Train a neural network with TensorFlow.

    Parameters
    ----------
    training_set : array_like
        (Wavelength, Particle Size, VMR1, VMR2,..., Extinction, Scattering, Asymmetry)
    model_params : dict
        {'name' (optional): str(name to give to model),
        'layers' (optional): int(number of hidden layers),
        'nodes' (optional): int(number of nodes per hidden layer),
        'activation_function' (optional): str(activation function),
        'optimizer' (optional): str(optimizer),
        'loss' (optional): str(loss function),
        'metrics' (optional): str(metric function),
        'batch_size' (optional): int(batch size),
        'epochs' (optional): int(number of epochs),
        'log_wavelength' (optional): boolean, - True (default) trains model on log(wavelength)
        'log_particle_size' (optional): boolean,
        'log_extinction' (optional): boolean,
        'log_scattering' (optional): boolean,}
    plot_training : boolean, optional
        Whether or not to plot the training loss and accuracy
    """
    # ==== DEFAULT MODEL PARAMETERS ====================================================================================

    # set model name
    if 'name' in model_params:
        if os.path.exists(self.data_path + model_params['name'] + '.keras'):
            raise ValueError('Model with the name' + model_params['name'] + 'already exists. Please provide a new name.')
    else:
        now = datetime.now()
        model_params['name'] = 'model' + now.strftime('%m_%d_%Y_%H_%M_%S')

    # set defaults if model parameter not given
    if 'layers' not in model_params:
        model_params['nodes'] = 3
    if 'nodes' not in model_params:
        model_params['nodes'] = 100
    if 'activation_function' not in model_params:
        model_params['activation_function'] = 'gelu'
    if 'optimizer' not in model_params:
        model_params['optimizer'] = 'adam'
    if 'loss' not in model_params:
        model_params['loss'] = 'mse'
    if 'metrics' not in model_params:
        model_params['metrics'] = 'mae'
    if 'batch_size' not in model_params:
        model_params['batch_size'] = 32
    if 'epochs' not in model_params:
        model_params['epochs'] = 10
    if 'log_wavelength' not in model_params:
        model_params['log_wavelength'] = True
    if 'log_particle_size' not in model_params:
        model_params['log_particle_size'] = True
    if 'log_extinction' not in model_params:
        model_params['log_extinction'] = True
    if 'log_scattering' not in model_params:
        model_params['log_scattering'] = True

    # ==== PREPARE TRAINING AND VALIDATION INPUTS AND OUTPUTS ==========================================================
    # import tensorflow here so MieNet can be used without it
    from tensorflow import keras

    # check input and output scaling
    if model_params['log_wavelength'] == True:
        training_set[:, 0] = np.log10(training_set[:, 0])
    if model_params['log_particle_size'] == True:
        training_set[:, 1] = np.log10(training_set[:, 1])
    if model_params['log_extinction'] == True:
        training_set[:, -3] = np.log10(training_set[:, -3])
    if model_params['log_scattering'] == True:
        training_set[:, -2] = np.log10(training_set[:, -2])

    # define number of inputs and set size
    num_inputs = int(training_set.shape[1] - 3)  # number of model inputs
    set_size = training_set.shape[0]  # total dataset size
    N = int(0.9 * set_size)  # split into training and validation set

    # training inputs and outputs
    training_inputs = training_set[:N, :num_inputs]
    training_extinction = training_set[:N, -3]
    training_scattering = training_set[:N, -2]
    training_asymmetry = training_set[:N, -1]

    # validation inputs and outputs
    validation_inputs = training_set[N:, :num_inputs]
    validation_extinction = training_set[N:, -3]
    validation_scattering = training_set[N:, -2]
    validation_asymmetry = training_set[N:, -1]

    # ==== DEFINE MODEL STRUCTURE ======================================================================================
    # inputs
    inputs = keras.Input(shape=(num_inputs,), name='inputs')

    # layers
    hidden = keras.layers.Dense(model_params['nodes'], activation = model_params['activation_function'])(inputs)
    for i in range(model_params['layers'] - 1):
        hidden = keras.layers.Dense(model_params['nodes'], activation = model_params['activation_function'])(hidden)

    # outputs
    output1 = keras.layers.Dense(1, name='extinction')(hidden)
    output2 = keras.layers.Dense(1, name='scattering')(hidden)
    output3 = keras.layers.Dense(1, name='asymmetry')(hidden)

    # make model
    model = keras.Model(inputs, outputs=[output1, output2, output3])

    # ==== TRAIN AND SAVE MODEL ========================================================================================

    # compile model for training
    model.compile(optimizer = model_params['optimizer'],
                  loss = [model_params['loss']] * 3,
                  metrics = [model_params['metrics']] * 3)

    # train model
    history = model.fit(training_inputs, [training_extinction, training_scattering, training_asymmetry],
                           validation_data = (validation_inputs,
                                              [validation_extinction, validation_scattering, validation_asymmetry]),
                           batch_size = model_params['batch_size'], epochs = model_params['epochs'])

    # save model
    model.save(self.data_path + model_params['name'] + '.keras')

    # ==== PLOT LOSS AND ACCURACY ======================================================================================
    if plot_training == True:

        # Allow plotting
        self.mute = False

        epoch = range(1, model_params['epochs'] + 1)

        plt.figure()
        plt.plot(epoch, history.history['extinction_mae'], label='Extinction', color='blue')
        plt.plot(epoch, history.history['scattering_mae'], color='green', label='Scattering')
        plt.plot(epoch, history.history['asymmetry_mae'], color='#57B9FF', label='Asymmetry')
        plt.plot(epoch, history.history['val_extinction_mae'], label='Val. Extinction', alpha=0.5, color='blue')
        plt.plot(epoch, history.history['val_scattering_mae'], label='Val. Scattering', color='green', alpha=0.5)
        plt.plot(epoch, history.history['val_asymmetry_mae'], label='Val. Asymmetry', color='#57B9FF', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Model {model_params['name']} Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(epoch, history.history['extinction_loss'], label='Extinction', color='blue')
        plt.plot(epoch, history.history['scattering_loss'], color='green', label='Scattering')
        plt.plot(epoch, history.history['asymmetry_loss'], color='#57B9FF', label='Asymmetry')
        plt.plot(epoch, history.history['val_extinction_loss'], label='Val. Extinction', alpha=0.5, color='blue')
        plt.plot(epoch, history.history['val_scattering_loss'], label='Val. Scattering', color='green', alpha=0.5)
        plt.plot(epoch, history.history['val_asymmetry_loss'], label='Val. Asymmetry', color='#57B9FF', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Model {model_params['name']} Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # # ==== ADD MODEL TO CONFIG FILE ====================================================================================
    # # path to config file
    # config_path = self.data_path + 'config.yaml'
    #
    # # model info to add to file
    # new_data = {model_params['name']: {'architecture': 'one_network',
    #                                    'theory': theory,
    #                                    'dependencies': {},
    #                                    'species': species,
    #                                    'range': {'wavelength': wave_range, 'particle_size': size_range},
    #                                    'scale': scale,
    #                                    'files': model_params['name'] + '.keras'}
    #             }
    #
    # # check for existing config file
    # if os.path.exists(config_path):
    #     with open(config_path, 'r') as file:
    #         current_data = yaml.safe_load(file) or {}
    #
    #     with open(config_path, 'w') as file:
    #         yaml.safe_dump(current_data, file, default_flow_style=False, sort_keys=False)
    #
    # # create config file if none exist
    # else:
    #     with open(config_path, 'w') as file:
    #         yaml.safe_dump(new_data, file, default_flow_style=False, sort_keys=False)
