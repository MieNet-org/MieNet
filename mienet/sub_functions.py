""" General functionalities """
# pylint: disable=C0415,R0902,R0912,R0914,R0915

import os
import pandas as pd
import numpy as np

def read_in_refindex(species, wavelength, files):
    """
    Read in and interpolate refractive index files.

    Parameters
    ----------
    species : List with size N
        Name of cloud species.
    wavelength : np.ndarray of size M
        Wavelength of the light [micron]
    files : List
        Refractive index files

    Return
    ------
    ref_index : np.ndarray of size (N, M, 2)
        Refractive index data: real, and imaginary part.
    """

    # ==== Input handling
    # make wavelength to array if it is a float
    if not isinstance(wavelength, np.ndarray):
        wavelength = np.asarray([wavelength])

    # prepare output
    ref_index = np.zeros((len(species), len(wavelength), 2))
    for s, spec in enumerate(species):

        # ==== Load data from files =====================================================

        # find species in files
        data = None
        for file in files:
            sp_name = os.path.basename(file).split('/')[0][:-8]
            if spec == sp_name:
                # get data using pandas
                content = pd.read_csv(file, sep=r'\s+', header=None, usecols=[1, 2, 3])
                # convert to array and flip vertically so wavelength increases
                data = np.flip(content.to_numpy(), axis=0)
        if data is None:
            raise ValueError('No refindex file found for ' + spec)

        # ==== Get the real(n) and imaginary (k) refractory index =======================
        # loop over all wavelengths
        for wav, wave in enumerate(wavelength):
            # if desired wavelength is smaller than data, use the smallest wavelength
            # data available
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

def select_best_dataset(type, vmrs, datasets, stop=True):
    """
    Choose best grid or model.

    Parameters
    ----------
    type : String
        Either 'model' or 'grid'
    volume_mixing_ratios: Dictionary
        Species in each mixture with respective volume mixing ratios.
    datasets: Dictionary
        Dictionary with information about each model/grid.
    stop: Bool, optional
        If True, an error is raised if no set is found. If False, (None, None) is returned.

    Returns
    ----------
    best_dataset : tuple, (name, species)
        name: str of model/grid name
        species: list of species name
    """
    # find all dataset that include all species
    l_set = set(vmrs.keys())
    valid_datasets = {
        name: data['species'] for name, data in datasets.items()
        if l_set.issubset(data['species'])
    }

    # check if there are matching datasets
    if valid_datasets:
        # Now pick the dataset with the smallest total size
        return min(valid_datasets.items(), key=lambda item: len(item[1]))

    # raise value error if no datasets for the type of efficiency function called
    if stop:
        raise ValueError("[ERROR] No default " + f'{type}' + " for " + str(l_set) +
                         " is available. Please provide one.")

    # if run should not be apported, return (None, None) dataset
    return None, None

def input_check(wavelength, particle_size, volume_mixing_ratios, species_list, mute=True):
    """
    This function assures that all inputs are in the correct format.
    """

    # check inputs are correct type
    if (not isinstance(wavelength, np.ndarray)
            and not isinstance(wavelength, (float, int))):
        raise ValueError('[ERROR] Wavelength must be of type np.ndarray or float')
    if (not isinstance(particle_size, np.ndarray)
            and not isinstance(particle_size, (float, int))):
        raise ValueError('[ERROR] Particle size must be of type np.ndarray or float')
    if not isinstance(volume_mixing_ratios, dict):
        raise ValueError('[ERROR] Volume mixing ratio must be of type dict or float')

    # convert floats to arrays
    if isinstance(wavelength, (float, int)):
        wavelength = np.array([wavelength])
    if isinstance(particle_size, (float, int)):
        particle_size = np.array([particle_size])
    for key, ratios in volume_mixing_ratios.items():
        if isinstance(ratios, (float, int)):
            volume_mixing_ratios[key] = np.array([ratios])

    # there must be the same number of VMRs for each species
    if len(set(map(len, volume_mixing_ratios.values()))) != 1:
        raise ValueError('[ERROR] Volume mixing ratios must have same shape')
    # each particle size needs a VMR
    if len(particle_size) != len(volume_mixing_ratios[next(iter(volume_mixing_ratios))]):
        raise ValueError('[ERROR] Particle size and volume mixing ratio must have '
                         'same shape')

    # create array with vmr values
    vmr = np.zeros((len(particle_size), len(species_list)))
    for s, spec in enumerate(species_list):
        vmr[:, s] = volume_mixing_ratios[spec]

    # Check if all VMRs add up to 1 and normalise if necessary
    vmr_sum = np.sum(vmr, axis=1)
    if any(vmr_sum != 1) and not mute:
        print('[WARN] Volume mixing ratios do not add up to 1. '
              'The ratios have been renormalized.')
    idx = np.asarray(np.where(vmr_sum != 1))
    for i in idx[0]:
        vmr[i] = vmr[i] / sum(vmr[i])

    return wavelength, particle_size, vmr
