""" Integration tests """
import numpy as np
from mienet import MieNet
import xarray as xr

def test_sub_functions():
    # ==== test Bruggeman
    ma = MieNet(use_ai=False, mute=False)
    extinction, scattering, asymmetry = ma.efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
        {'SiO2': np.linspace(0, 1, 8), 'Fe': np.linspace(1, 0, 8)},
        theory='Bruggeman'
    )
    assert np.isclose(np.sum(extinction), 134.7829592828185)
    assert np.isclose(np.sum(scattering), 104.69387306717095)
    assert np.isclose(np.sum(asymmetry), 45.396489319908156)

def create_ai_model():
    # ==== Test generate_training_set
    ma = MieNet(use_ai=True, mute=False, default_model_location='../ci_test/')
    ma.generate_training_set('test_set', species=['SiO2', 'MgSiO3'],
                             wavelength_sample=(0.1, 10, 25), particle_size_sample=(0.001, 0.01, 25))

    # ==== Test train_ai_model
    dataset = xr.open_dataset('../ci_test/test_set.nc')
    dataset_arr = dataset['data'].to_numpy()

    # set model parameters
    model_params = {'name': 'tutorial_model',  # file name of AI model
                    'layers': 2}  # number of hidden layers
    # optional parameters: nodes, activation_function, optimizer, loss, metrics, batch_size, epochs,
    # log_wavelength, log_particle_size, log_extinction, log_scattering

    # train your model
    ma.train_ai_model(dataset_arr, model_params={'name': 'test_model', 'layers': 1}, plot_training=False)
