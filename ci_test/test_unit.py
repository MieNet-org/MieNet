""" Integration tests """
import numpy as np
from mienet import MieNet
import xarray as xr
import os
from tensorflow import keras

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

def test_create_ai_model():
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

    os.remove('../ci_test/test_set.nc')
    os.remove('../ci_test/test_set.keras')

    assert np.isclose(np.sum(extinction), 0.5209891121429818)
    assert np.isclose(np.sum(scattering), 0.2536509761555041)
    assert np.isclose(np.sum(asymmetry), 0.3659575144210919)

def test_multiple_network_models():
    # ==== Generate dataset for test
    ma = MieNet(mute=False, default_model_location='../ci_test/')
    ma.generate_training_set('test_set', species=['SiO2', 'MgSiO3'],
                             wavelength_sample=(0.1, 10, 25), particle_size_sample=(0.001, 0.01, 25))

    # ==== Open dataset and prep inputs and outputs
    dataset = xr.open_dataset('../ci_test/test_set.nc')
    dataset_arr = dataset['data'].to_numpy()
    training_inputs = dataset_arr[:, :3]
    training_extinction = dataset_arr[:, -3]
    training_scattering = dataset_arr[:, -2]
    training_asymmetry = dataset_arr[:, -1]

    # ==== masks
    w1 = dataset_arr[:, 0] < 0.3
    w2 = (dataset_arr[:, 0] >= 0.3) & (dataset_arr[:, 0] < 0.7)
    w3 = dataset_arr[:, 0] > 0.7
    s1 = ((2*np.pi*dataset_arr[:, 1]) / dataset_arr[:, 0]) > 0.01
    s2 = ((2*np.pi*dataset_arr[:, 1]) / dataset_arr[:, 0]) <= 0.01
    m1a, m1b, m1c, m2a, m2b, m2c = w1 * s1, w2 * s1, w3 * s1, w1 * s2, w2 * s2, w3 * s2

    # ==== Model structure
    inputs = keras.Input(shape=(3,), name='inputs')
    hidden = keras.layers.Dense(100, activation='gelu')(inputs)
    output1 = keras.layers.Dense(1, name='extinction')(hidden)
    output2 = keras.layers.Dense(1, name='scattering')(hidden)
    output3 = keras.layers.Dense(1, name='asymmetry')(hidden)
    model = keras.Model(inputs, outputs=[output1, output2, output3])

    # make model
    model = keras.Model(inputs, outputs=[output1, output2, output3])

    # compile model for training
    model.compile(optimizer='adam', loss=['mse', 'mse', 'mse'],
                     metrics=['mae', 'mae', 'mae'])

    model.fit(training_inputs, [training_extinction, training_scattering, training_asymmetry],
                           batch_size=1024, epochs=10)

    model.save(f'vmodel{ver}_1A.keras')  # save entire model
    # model.save_weights(f'vmodel{ver}.weights.h5') # save model weights but not structure
    
    # ==== Test created model predictions
    ma = MieNet(mute=False, default_model_location='../ci_test/')
    extinction, scattering, asymmetry = ma.ai_efficiencies(np.linspace(0.1, 10, 8),
                                                           np.linspace(0.001, 0.01, 8),
                                                           {'SiO2': np.linspace(0, 1, 8),
                                                            'MgSiO3': np.linspace(1, 0, 8)})

    os.remove('../ci_test/test_set.nc')
    os.remove('../ci_test/test_set.keras')

    assert np.isclose(np.sum(extinction), 0.5209891121429818)
    assert np.isclose(np.sum(scattering), 0.2536509761555041)
    assert np.isclose(np.sum(asymmetry), 0.3659575144210919)
