""" Integration tests """
import os
import unittest
import numpy as np

from mienet import MieNet

testcase = unittest.TestCase()

def test_full():
    ma = MieNet(use_ai=False)

    # ==== Test same particle size input
    extinction, scattering, asymmetry = ma.auto_efficiencies(
        np.logspace(-0.5, 1, 8), np.asarray([1]),
        {
            'TiO2': np.linspace(0, 1, 1),
            'Fe': np.linspace(1, 0, 1),
        }
    )
    assert np.isclose(np.sum(extinction), 17.910632064187162)
    assert np.isclose(np.sum(scattering), 14.70757922824962)
    assert np.isclose(np.sum(asymmetry), 2.8180171560812473)

    # ==== Test auto call
    extinction, scattering, asymmetry = ma.auto_efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(-3, -2, 8),
        {
            'TiO2': np.linspace(0, 1, 8),
            'Fe': np.linspace(1, 0, 8),
        }
    )
    assert np.isclose(np.sum(extinction), 0.22395971588655875)
    assert np.isclose(np.sum(scattering), 0.03450202166172343)
    assert np.isclose(np.sum(asymmetry), -0.318803615274828)

    # ==== request non exisitng species
    with testcase.assertRaises(ValueError):
        ma.efficiencies(1, 1,{'WRONG': 1})

    # ==== Test wrongly sorted particle sizes
    scramb = [1, 7, 5, 3, 4, 2, 0, 6]
    ps_unsorted = np.logspace(-3, -2, 8)[scramb]
    tio2_vmr_unsorted = np.linspace(0, 1, 8)[scramb]
    fe_vmr_unsorted = np.linspace(1, 0, 8)[scramb]
    extinction, scattering, asymmetry = ma.efficiencies(
        np.logspace(-0.5, 1, 8), ps_unsorted,
        {
            'TiO2': tio2_vmr_unsorted,
            'Fe': fe_vmr_unsorted,
        }
    )
    assert np.isclose(np.sum(extinction), 0.22395971588655875)
    assert np.isclose(np.sum(scattering), 0.03450202166172343)
    assert np.isclose(np.sum(asymmetry), -0.318803615274828)

def test_ai():
    # ==== Set up
    loc = os.path.dirname(__file__) + '/../docs/tutorial_files/'
    ma = MieNet(default_data_location=loc, mute=False)

    # ==== Standard Ai run
    extinction, scattering, asymmetry = ma.ai_efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(-3, -2, 8),
        {
            'Mg2SiO4': np.linspace(0, 1, 8),
            'Fe': np.linspace(1, 0, 8),
        }
    )
    assert np.isclose(np.sum(extinction), 37.19869)
    assert np.isclose(np.sum(scattering), 12.870874)
    assert np.isclose(np.sum(asymmetry), -23.71189)

    # ==== Use load grid model
    ma = MieNet(default_data_location=loc, load_ai_model='TUTORIAL_MODEL')
    extinction, scattering, asymmetry = ma.ai_efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(-3, -2, 8),
        {
            'Mg2SiO4': np.linspace(0, 1, 8),
            'Fe': np.linspace(1, 0, 8),
        }
    )
    assert np.isclose(np.sum(extinction), 37.19869)
    assert np.isclose(np.sum(scattering), 12.870874)
    assert np.isclose(np.sum(asymmetry), -23.71189)
    # request wrong mixture
    with testcase.assertRaises(ValueError):
        ma.ai_efficiencies(1, 1,{'WRONG': 1,})

    # ==== Test float input
    extinction, scattering, asymmetry = ma.ai_efficiencies(3, 0.005,
        {
            'Mg2SiO4': [0.4],
            'Fe': [0.4],
        }
    )
    assert np.isclose(np.sum(extinction), 0.33872768)
    assert np.isclose(np.sum(scattering), 0.04487792)
    assert np.isclose(np.sum(asymmetry), -0.3625094)

    # ==== Test auto call
    extinction, scattering, asymmetry = ma.auto_efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(-3, -2, 8),
        {
            'Mg2SiO4': np.linspace(0, 1, 8),
            'Fe': np.linspace(1, 0, 8),
        }
    )
    assert np.isclose(np.sum(extinction), 37.19869)
    assert np.isclose(np.sum(scattering), 12.870874)
    assert np.isclose(np.sum(asymmetry), -23.71189)

    # ==== Test one fewer species than in model
    extinction, scattering, asymmetry = ma.auto_efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(-3, -2, 8),
        {
            'Mg2SiO4': np.linspace(1, 1, 8),
        }
    )
    assert np.isclose(np.sum(extinction), 43.0988)
    assert np.isclose(np.sum(scattering), 13.092415)
    assert np.isclose(np.sum(asymmetry), -21.044868)

    # ==== Test wavelength and paticle size limit
    with testcase.assertRaises(ValueError):
        ma.ai_efficiencies(1e10, 0.005,{'Mg2SiO4': [0.4], 'Fe': [0.4],})
    with testcase.assertRaises(ValueError):
        ma.ai_efficiencies(1e-10, 0.005,{'Mg2SiO4': [0.4], 'Fe': [0.4],})
    with testcase.assertRaises(ValueError):
        ma.ai_efficiencies(3, 1e10,{'Mg2SiO4': [0.4], 'Fe': [0.4],})
    with testcase.assertRaises(ValueError):
        ma.ai_efficiencies(3, 1e-10,{'Mg2SiO4': [0.4], 'Fe': [0.4],})

    # ==== Test wrong theory catch
    with testcase.assertRaises(ValueError):
        ma.ai_efficiencies(3, 0.005,{'Mg2SiO4': [0.4], 'Fe': [0.4],}, theory='WRONG')

    # ==== Test wrong model load
    with testcase.assertRaises(ValueError):
        MieNet(default_data_location=loc, load_ai_model='NON_EXISTING')

    # ==== Test non-initialisation error
    with testcase.assertRaises(ValueError):
        ma = MieNet(use_ai=False)
        ma.ai_efficiencies(None, None, None)


def test_grid():
    # ==== Set up
    ma = MieNet(use_ai=False, mute=False)
    test_vars = ['qext', 'qsca', 'asym', 'wavelength']

    # ==== create tiny grid
    ds = ma.produce_efficiency_grid(
        ['SiO2', 'Fe'], wavelengths=np.logspace(-1 ,1.3 ,5),
        particle_sizes=np.logspace(1,2 ,10), vmr_data_points=3,
        save_file='grid_test.nc'
    )
    expected_vals = [317.19851769, 263.92740723, 104.12352316, 27.14984254]
    for t, test in enumerate(test_vars):
        assert np.isclose(np.sum(ds[test]), expected_vals[t])

    # ==== calling
    with testcase.assertRaises(ValueError):
        ma.grid_efficiencies(
            np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
            {'SiO2': np.linspace(0, 1, 8), 'Fe': np.linspace(1, 0, 8)}
        )

    # ==== read in grid
    # test file read in
    ma.load_grid_efficiency(file_name='grid_test.nc')
    lo = ma.default_grids['grid_test.nc']['ds']
    for t, test in enumerate(test_vars):
        assert np.isclose(np.sum(lo[test]), expected_vals[t])
    assert ['SiO2', 'Fe'] == lo.attrs['species']
    # test ds_grid read in
    ma.load_grid_efficiency(ds_grid=ds, file_name=None)
    assert len(ma.default_grids) == 2
    # test asserts
    with testcase.assertRaises(ValueError):
        ma.load_grid_efficiency(file_name='grid_that_does_not_exist.nc')

    # === use grid evaluation
    extinction, scattering, asymmetry = ma.grid_efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
        {'SiO2': np.linspace(0, 1, 8), 'Fe': np.linspace(1, 0, 8)}
    )
    assert np.isclose(np.sum(extinction), 134.5516828568501)
    assert np.isclose(np.sum(scattering), 109.84721926458762)
    assert np.isclose(np.sum(asymmetry), 45.54068444245176)

    # === use grid_file to load in grid file
    ma = MieNet(use_ai=False, mute=False, grid_file='grid_test.nc')
    extinction, scattering, asymmetry = ma.grid_efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
        {'SiO2': np.linspace(0, 1, 8), 'Fe': np.linspace(1, 0, 8)}
    )
    assert np.isclose(np.sum(extinction), 134.5516828568501)
    assert np.isclose(np.sum(scattering), 109.84721926458762)
    assert np.isclose(np.sum(asymmetry), 45.54068444245176)

    # === use auto evaluation
    extinction, scattering, asymmetry = ma.auto_efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
        {'SiO2': np.linspace(0, 1, 8), 'Fe': np.linspace(1, 0, 8)}
    )
    assert np.isclose(np.sum(extinction), 134.5516828568501)
    assert np.isclose(np.sum(scattering), 109.84721926458762)
    assert np.isclose(np.sum(asymmetry), 45.54068444245176)

    # === use with less species than grid
    extinction, scattering, asymmetry = ma.grid_efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
        {'SiO2': np.linspace(0, 1, 8)}
    )
    assert np.isclose(np.sum(extinction), 118.03743267969153)
    assert np.isclose(np.sum(scattering), 106.39411449611903)
    assert np.isclose(np.sum(asymmetry), 47.528850364238835)

    # === test zerso fill
    extinction, scattering, asymmetry = ma.grid_efficiencies(
        np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
        {'Fe': np.linspace(0, 1, 8)}
    )
    assert np.isclose(np.sum(extinction), 133.4183907490462)
    assert np.isclose(np.sum(scattering), 114.47160632559543)
    assert np.isclose(np.sum(asymmetry), 37.74865263737628)

    # ==== request species that are not available
    with testcase.assertRaises(ValueError):
        _, _, _ = ma.grid_efficiencies(
            np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
            {'Not': np.linspace(0, 1, 8), 'Exist': np.linspace(1, 0, 8)},
        )

    # ==== Mismatch between dataset and request
    with testcase.assertRaises(ValueError):
        _, _, _ = ma.grid_efficiencies(
            np.logspace(-0.5, 1, 8), np.logspace(1.1, 1.9, 8),
            {'SiO2': np.linspace(0, 1, 8), 'Fe': np.linspace(1, 0, 8)},
            theory='NON-EXISTING'
        )

    # ==== finish up
    os.remove('grid_test.nc')
