""" Integration tests """
import unittest
import numpy as np
import xarray as xr
import os
from tensorflow import keras
import glob

from mienet import MieNet
from mienet.sub_functions import (read_in_refindex, calculate_subradii, initialize_ai_models,
                                  input_check)

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

    # input_check
    with testcase.assertRaises(ValueError):
        input_check([1], None, None, None)
    with testcase.assertRaises(ValueError):
        input_check(1, [1], None, None)
    with testcase.assertRaises(ValueError):
        input_check(1, 1, 1, None)
    with testcase.assertRaises(ValueError):
        input_check(1, 1, {'Fe': [1], 'Fe2': [1, 2]}, None)
    with testcase.assertRaises(ValueError):
        input_check(1, 1, {'Fe': [1, 2], 'Fe2': [1, 2]}, None)

def test_create_ai_model():
    # ==== Test generate_training_set
    ma = MieNet(mute=False, default_model_location='ci_test/')
    ma.generate_training_set('test_set', species=['SiO2', 'MgSiO3'],
                             wavelength_sample=(0.1, 10, 25), particle_size_sample=(0.001, 0.01, 25))

    # ==== Test train_ai_model
    dataset = xr.open_dataset('ci_test/test_set.nc')
    dataset_arr = dataset['data'].to_numpy()
    ma.train_ai_model(dataset_arr,
                      model_params={'name': 'test_model', 'layers': 2, 'batch_size': 32},
                      plot_training=False)

    # ==== Test created model predictions
    ma = MieNet(mute=False, default_model_location='ci_test/')
    extinction, scattering, asymmetry = ma.ai_efficiencies(np.linspace(0.1, 10, 8),
                                                           np.linspace(0.001, 0.01, 8),
                                                           {'SiO2': np.linspace(0, 1, 8),
                                                            'MgSiO3': np.linspace(1, 0, 8)})

    os.remove('ci_test/test_set.nc')
    os.remove('ci_test/test_model.keras')

    assert np.isclose(np.sum(extinction), 20.608023, rtol = 5, atol = 5)
    assert np.isclose(np.sum(scattering), 31.9421, rtol = 5, atol = 5)
    assert np.isclose(np.sum(asymmetry), -72.7534, rtol = 5, atol = 5)

# def test_multiple_network_models():
#     # ==== Generate dataset for test
#     ma = MieNet(mute=False, default_model_location='ci_test/')
#     ma.generate_training_set('test_set', species=['SiO2', 'MgSiO3'],
#                              wavelength_sample=(0.095, 0.1, 100), particle_size_sample=(0.095, 0.1, 100))
#
#     # ==== Open dataset and prep inputs and outputs
#     dataset = xr.open_dataset('ci_test/test_set.nc')
#     os.remove('ci_test/test_set.nc')
#     dataset_arr = dataset['data'].to_numpy()
#     training_inputs = dataset_arr[:, :3]
#     training_extinction = dataset_arr[:, -3]
#     training_scattering = dataset_arr[:, -2]
#     training_asymmetry = dataset_arr[:, -1]
#
#     # ==== masks
#     w1 = dataset_arr[:, 0] < 0.0975
#     w2 = (dataset_arr[:, 0] >= 0.0975) & (dataset_arr[:, 0] < 0.0985)
#     w3 = dataset_arr[:, 0] > 0.0985
#     s1 = ((2*np.pi*dataset_arr[:, 1]) / dataset_arr[:, 0]) > 80.5
#     s2 = ((2*np.pi*dataset_arr[:, 1]) / dataset_arr[:, 0]) <= 80.5
#     m1a, m1b, m1c, m2a, m2b, m2c = w1 * s1, w2 * s1, w3 * s1, w1 * s2, w2 * s2, w3 * s2
#     print(min((2*np.pi*dataset_arr[:, 1]) / dataset_arr[:, 0]))
#     print(max((2 * np.pi * dataset_arr[:, 1]) / dataset_arr[:, 0]))
#     print(len(dataset_arr[m1a]))
#     print(len(dataset_arr[m1b]))
#     print(len(dataset_arr[m1c]))
#     print(len(dataset_arr[m2a]))
#     #print(min((2 * np.pi * dataset_arr[m2b,1]) / dataset_arr[m2b,0]))
#     print(len(dataset_arr[m2b]))
#     print(len(dataset_arr[m2c]))
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.tricontourf(dataset_arr[:, 0], dataset_arr[:, 1], (2 * np.pi * dataset_arr[:, 1]) / dataset_arr[:, 0])
#     plt.axvline(0.0975, color='black')
#     plt.axvline(0.0985, color='black')
#     plt.colorbar()
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.show()
#
#     # ==== Model structure
#     inputs = keras.Input(shape=(3,), name='inputs')
#     hidden = keras.layers.Dense(100, activation='relu')(inputs)
#     hidden = keras.layers.Dense(100, activation='relu')(hidden)
#     output1 = keras.layers.Dense(1, name='extinction')(hidden)
#     output2 = keras.layers.Dense(1, name='scattering')(hidden)
#     output3 = keras.layers.Dense(1, name='asymmetry')(hidden)
#     model = keras.Model(inputs, outputs=[output1, output2, output3])
#
#     # ==== Three and six network models
#     for i, mask in enumerate([w1, w2, w3, m1a, m1b, m1c, m2a, m2b, m2c]):
#         print(i)
#         model.compile(optimizer='adam', loss=['mse', 'mse', 'mse'],
#                          metrics=['mae', 'mae', 'mae'])
#         model.fit(training_inputs[mask],
#                   [training_extinction[mask], training_scattering[mask], training_asymmetry[mask]],
#                   batch_size=32, epochs=10)
#         model.save(f'test_model_{i}.keras')
#
#     # ==== Test created model predictions
#     ma = MieNet(mute=False, default_model_location='ci_test/')
#     extinction, scattering, asymmetry = ma.ai_efficiencies(np.linspace(0.1, 5, 8),
#                                                            np.linspace(0.001, 0.01, 8),
#                                                            {'SiO2': np.linspace(0, 1, 8),
#                                                             'MgSiO3': np.linspace(1, 0, 8)})
#
#     os.remove(glob.glob('ci_test/**.keras'))
#
#     print(np.sum(extinction), np.sum(scattering), np.sum(asymmetry))
#
#     assert np.isclose(np.sum(extinction), 0.5209891121429818)
#     assert np.isclose(np.sum(scattering), 0.2536509761555041)
#     assert np.isclose(np.sum(asymmetry), 0.3659575144210919)
