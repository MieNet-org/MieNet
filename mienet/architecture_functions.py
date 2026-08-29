""" Architecture dependent functionalities """
# pylint: disable=C0415,R0902,R0912,R0914,R0915

import numpy as np

def three_network(inputs, dependencies):
    """
    Predict outputs with three ANNs using two wavelength cutoffs.

    Parameters
    ----------
    inputs: np.array with shape: (num_parameters, data_length)
        num_parameters: number of parameters (contains wavelength, radius, and VMRs)
        data_length: number of data points
    dependencies: Dictionary
        Contains 2 wavelength cutoffs

    Returns
    ----------
    masks: low_mask, mid_mask, and high_mask for masking model inputs
    """
    low_mask = inputs[:, 0] <= np.log10(dependencies['low_wave'])
    mid_mask = ((inputs[:, 0] > np.log10(dependencies['low_wave'])) & (inputs[:, 0] < np.log10(dependencies['high_wave'])))
    high_mask = inputs[:, 0] >= np.log10(dependencies['high_wave'])

    return low_mask, mid_mask, high_mask


def six_network(inputs, dependencies):
    """
    Predict outputs with six ANNs using two wavelength cutoffs and one size parameter cutoff.

    Parameters
    ----------
    inputs: np.array with shape: (num_parameters, data_length)
        num_parameters: number of parameters (contains wavelength, radius, and VMRs)
        data_length: number of data points
    dependencies: Dictionary
        Contains 2 wavelength cutoffs and 1 size parameter cutoff

    Returns
    ----------
    masks: 6 masks for masking model inputs
    """
    size_param = 2 * np.pi * (10**inputs[:,1]) / (10**inputs[:,0])

    mask_1A = (10 ** inputs[:, 0] <= dependencies['low_wave']) & (size_param >= dependencies['size_cutoff'])
    mask_1B = (10 ** inputs[:, 0] > dependencies['low_wave']) & (10 ** inputs[:, 0] < dependencies['high_wave']) & (size_param >= dependencies['size_cutoff'])
    mask_1C = (10 ** inputs[:, 0] >= dependencies['high_wave']) & (size_param >= dependencies['size_cutoff'])
    mask_2A = (10 ** inputs[:, 0] <= dependencies['low_wave']) & (size_param < dependencies['size_cutoff'])
    mask_2B = (10 ** inputs[:, 0] > dependencies['low_wave']) & (10 ** inputs[:, 0] < dependencies['high_wave']) & (size_param < dependencies['size_cutoff'])
    mask_2C = (10 ** inputs[:, 0] >= dependencies['high_wave']) & (size_param < dependencies['size_cutoff'])

    return mask_1A, mask_1B, mask_1C, mask_2A, mask_2B, mask_2C