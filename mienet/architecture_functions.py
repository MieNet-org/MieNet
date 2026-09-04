""" Architecture dependent functionalities """
# pylint: disable=C0415,R0902,R0912,R0914,R0915

import numpy as np

def three_network(inputs, dependencies, scale):
    """
    Predict outputs with three ANNs using two wavelength cutoffs.

    Parameters
    ----------
    inputs: np.array with shape: (num_parameters, data_length)
        num_parameters: number of parameters (contains wavelength, radius, and VMRs)
        data_length: number of data points
    dependencies: Dictionary
        Contains 2 wavelength cutoffs
    scale: Dictionary
        Contains scaling of wavelength, particle_size, extinction, and scattering

    Returns
    ----------
    masks: low_mask, mid_mask, and high_mask for masking model inputs
    """
    # change wavelength scaling to be the same as wavelength cutoffs
    if scale['wavelength'] == 'log':
        wavelength = 10**inputs[:, 0]
    else:
        wavelength = inputs[:, 0]

    # masks
    low_mask = wavelength <= dependencies['low_wave']
    mid_mask = ((wavelength > dependencies['low_wave']) & (wavelength < dependencies['high_wave']))
    high_mask = wavelength >= dependencies['high_wave']

    return low_mask, mid_mask, high_mask


def six_network(inputs, dependencies, scale):
    """
    Predict outputs with six ANNs using two wavelength cutoffs and one size parameter cutoff.

    Parameters
    ----------
    inputs: np.array with shape: (num_parameters, data_length)
        num_parameters: number of parameters (contains wavelength, radius, and VMRs)
        data_length: number of data points
    dependencies: Dictionary
        Contains 2 wavelength cutoffs and 1 size parameter cutoff
    scale: Dictionary
        Contains scaling of wavelength, particle_size, extinction, and scattering

    Returns
    ----------
    masks: 6 masks for masking model inputs
    """
    # change wavelength scaling to be the same as wavelength cutoffs
    if scale['wavelength'] == 'log':
        wavelength = 10**inputs[:, 0]
    else:
        wavelength = inputs[:, 0]

    # change particle size scaling to be the same as particle size cutoffs
    if scale['particle_size'] == 'log':
        particle_size = 10**inputs[:, 1]
    else:
        particle_size = inputs[:, 1]

    # calculate size parameter
    size_param = (2 * np.pi * particle_size) / wavelength

    # masks
    mask_1A = (wavelength <= dependencies['low_wave']) & (size_param >= dependencies['size_cutoff'])
    mask_1B = ((wavelength > dependencies['low_wave']) & (wavelength < dependencies['high_wave']) &
               (size_param >= dependencies['size_cutoff']))
    mask_1C = (wavelength >= dependencies['high_wave']) & (size_param >= dependencies['size_cutoff'])
    mask_2A = (wavelength <= dependencies['low_wave']) & (size_param < dependencies['size_cutoff'])
    mask_2B = ((wavelength > dependencies['low_wave']) & (wavelength < dependencies['high_wave']) &
               (size_param < dependencies['size_cutoff']))
    mask_2C = (wavelength >= dependencies['high_wave']) & (size_param < dependencies['size_cutoff'])

    return mask_1A, mask_1B, mask_1C, mask_2A, mask_2B, mask_2C