""" Integration tests """
import unittest
import numpy as np
import xarray as xr
import os
from tensorflow import keras
import glob

from mienet import MieNet
from mienet.sub_functions import (read_in_refindex, calculate_subradii,
                                  input_check)
from mienet.architecture_functions import three_network, six_network

testcase = unittest.TestCase()

def test_mienet():
    ma = MieNet(default_data_location='.', mute=False)
    # check if missing ai models lead to assertion
    with testcase.assertRaises(ValueError):
        ma.ai_efficiencies(1, 1, 1)

def test_mixing_theory():
    # test wrong mixing theory
    ma = MieNet(use_ai=False)
    # test wrong mixing theory catch
    with testcase.assertRaises(ValueError):
        ma.efficiencies(1, 1, {'Fe': 1}, theory='WRONG')

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

    # ==== input_check
    # check if input checker works
    extinction, scattering, asymmetry = ma.efficiencies(1, 1, {'Fe': 1})
    assert np.isclose(np.sum(extinction), 2.5262886433126015)
    assert np.isclose(np.sum(scattering), 1.9303805158738072)
    assert np.isclose(np.sum(asymmetry), 0.6031789018842632)
    # assert tests
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

def test_model_handling():
    # ==== get models
    # link to files
    url = 'https://github.com/MieNet-org/MieNet/raw/refs/heads/main/ci_test/files/test_setup_1/test_dwl.zip'
    loc = os.path.dirname(__file__) + '/files/test_setup_1/'
    test = os.path.dirname(__file__) + '/files/test_setup_1/tutorial_model.keras'
    # delete old files if present
    try:
        os.remove(test)
    except OSError:
        pass
    # test if file is downloaded
    ma = MieNet(default_data_location=loc, mute=False)
    ma.use_ai = True
    ma.download_models(overwrite=False, url=url)
    assert os.path.exists(test)
    ma.download_models(overwrite=True, url=url)
    assert os.path.exists(test)
    os.remove(test)

    # ==== initialize ai
    ma = MieNet(use_ai=False, default_data_location=loc, mute=False)
    assert ma.initialize_ai_models() is None
    assert ma.force_disabled_ai
    loc = os.path.dirname(__file__) + '/files/test_setup_2/'
    ma = MieNet(use_ai=False, default_data_location=loc, mute=False)
    assert ma.initialize_ai_models() is None
    assert ma.force_disabled_ai
    with testcase.assertRaises(ValueError):
        ma = MieNet(default_data_location=loc, load_ai_model='TUTORIAL_MODEL', mute=False)


def test_create_ai_model():
    # ==== Test generate_training_set
    loc = os.path.dirname(__file__) + '/'
    try:
        os.remove(loc + 'test_set.nc')
    except OSError:
        pass
    try:
        os.remove(loc + 'test_model.keras')
    except OSError:
        pass
    try:
        os.remove(loc + 'test_set.keras')
    except OSError:
        pass
    try:
        os.remove(loc + 'test_set1.keras')
    except OSError:
        pass
    try:
        os.remove(loc + 'config.yaml')
    except OSError:
        pass

    ma = MieNet(mute=False, load_ai_model='test_set', default_data_location=loc)
    ma.generate_training_set('test_set', species=['SiO2', 'MgSiO3'],
                             wavelength_sample=(0.1, 10, 25), particle_size_sample=(0.001, 0.01, 25))
    with testcase.assertRaises(ValueError):
        ma.generate_training_set('test_set', species=['SiO2', 'MgSiO3'],
                                 wavelength_sample=(0.1, 10, 25), particle_size_sample=(0.001, 0.01, 25))

    # ==== Test train_ai_model
    ma.train_ai_model('test_set')
    assert os.path.exists(loc + 'test_set.keras')
    os.remove(loc + 'test_set.keras')
    ma.train_ai_model('test_set', overwrite=True)
    assert os.path.exists(loc + 'test_set.keras')
    ma.train_ai_model('test_set')
    assert os.path.exists(loc + 'test_set1.keras')
    os.remove(loc + 'test_set.keras')
    os.remove(loc + 'test_set1.keras')
    ma = MieNet(mute=False, default_data_location=loc)
    ma.train_ai_model('test_set',
                      model_params={'name': 'test_model'},
                      plot_training=False)
    with testcase.assertRaises(ValueError):
        ma.train_ai_model('test_set',
                      model_params={'name': 'test_model'})
    with testcase.assertRaises(ValueError):
        ma.train_ai_model('test_set',
                          model_params={'wavelength_scale': 'wrong',
                               'particle_size_scale': 'wrong',
                               'extinction_scale': 'wrong',
                               'scattering_scale': 'wrong'})

    # ==== Test created model predictions
    ma = MieNet(mute=False, default_data_location=loc)
    extinction, scattering, asymmetry = ma.ai_efficiencies(np.linspace(0.1, 10, 8),
                                                           np.linspace(0.001, 0.01, 8),
                                                           {'SiO2': np.linspace(0, 1, 8),
                                                            'MgSiO3': np.linspace(1, 0, 8)})

    os.remove(loc + 'test_set.nc')
    os.remove(loc + 'test_model.keras')
    os.remove(loc + 'config.yaml')

    assert np.isclose(np.sum(extinction), 20.608023, rtol = 5, atol = 5)
    assert np.isclose(np.sum(scattering), 31.9421, rtol = 5, atol = 5)
    assert np.isclose(np.sum(asymmetry), -72.7534, rtol = 5, atol = 5)

def test_architecture_functions():
    # ==== test three_network
    inp3 = np.asarray([[1, None, None], [2, None, None], [3, None, None]])
    dep3 = {'low_wave': 10**1.5, 'high_wave': 10**2.5}
    scale3 = {'wavelength': 'log', 'particle_size': 'log',
              'extinction': 'log', 'scattering': 'log'}
    res = three_network(inp3, dep3, scale3)
    assert (res[0] == np.asarray([True, False, False])).all()
    assert (res[1] == np.asarray([False, True, False])).all()
    assert (res[2] == np.asarray([False, False, True])).all()
    # test non log
    inp3 = np.asarray([[10**1, None, None], [10**2, None, None], [10**3, None, None]])
    dep3 = {'low_wave': 10**1.5, 'high_wave': 10**2.5}
    scale3 = {'wavelength': 'linear', 'particle_size': 'linear',
              'extinction': 'linear', 'scattering': 'linear'}
    res = three_network(inp3, dep3, scale3)
    assert (res[0] == np.asarray([True, False, False])).all()
    assert (res[1] == np.asarray([False, True, False])).all()
    assert (res[2] == np.asarray([False, False, True])).all()


    # ==== test six_network
    inp6 = np.asarray([
        [1, 1, None], [2, 2, None], [3, 3, None],
        [1, -1, None], [2, -2, None], [3, -3, None],
    ])
    dep6 = {'low_wave': 10**1.5, 'high_wave': 10**2.5, 'size_cutoff': 1}
    scale6 = {'wavelength': 'log', 'particle_size': 'log',
              'extinction': 'log', 'scattering': 'log'}
    res = six_network(inp6, dep6, scale6)
    for i in range(6):
        for j in range(6):
            if i == j:
                assert res[i][j]
            else:
                assert not res[i][j]
    # test non log
    inp6 = np.asarray([
        [10**1, 10**1, None], [10**2, 10**2, None], [10**3, 10**3, None],
        [10**1, 10**-1, None], [10**2, 10**-2, None], [10**3, 10**-3, None],
    ])
    dep6 = {'low_wave': 10**1.5, 'high_wave': 10**2.5, 'size_cutoff': 1}
    scale6 = {'wavelength': 'linear', 'particle_size': 'linear',
              'extinction': 'linear', 'scattering': 'linear'}
    res = six_network(inp6, dep6, scale6)
    for i in range(6):
        for j in range(6):
            if i == j:
                assert res[i][j]
            else:
                assert not res[i][j]
