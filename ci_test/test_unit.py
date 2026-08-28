""" Integration tests """
import unittest
import numpy as np
import xarray as xr

from mienet import MieNet
from mienet.sub_functions import read_in_refindex, calculate_subradii, initialize_ai_models

testcase = unittest.TestCase()

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

    # ==== test refindex read in
    # single wavelength, and wavlength out of range
    res = read_in_refindex(['SiO2'], 1e5, ma.files)
    assert np.isclose(res[0, 0, 0], 2.17629000e+00)
    assert np.isclose(res[0, 0, 1], 7.29521269e-06)
    # non existing file error
    with testcase.assertRaises(ValueError):
        read_in_refindex(['NON-EXISTING'], [1], ma.files)

    # ==== calculate_subradii
    assert calculate_subradii([1], [0.1])[0][0] == 1
    assert np.sum(calculate_subradii([1, 1], [0.1])[0]) == 12


def create_ai_model():
    # ==== Test generate_training_set
    ma = MieNet(mute=False, default_model_location='../ci_test/')
    ma.generate_training_set('test_set', species=['SiO2', 'MgSiO3'],
                             wavelength_sample=(0.1, 10, 25), particle_size_sample=(0.001, 0.01, 25))

    # ==== Test train_ai_model
    dataset = xr.open_dataset('../ci_test/test_set.nc')
    dataset_arr = dataset['data'].to_numpy()
    ma.train_ai_model(dataset_arr, model_params={'name': 'test_model', 'layers': 1}, plot_training=False)

    # ==== Test created model predictions
    ma = MieNet(mute=False, default_model_location='../ci_test/')
    extinction, scattering, asymmetry = ma.ai_efficiencies(np.linspace(0.1, 10, 8),
                                                           np.linspace(0.001, 0.01, 8),
                                                           {'SiO2': np.linspace(0, 1, 8),
                                                            'MgSiO3': np.linspace(1, 0, 8)})

    assert np.isclose(np.sum(extinction), )
