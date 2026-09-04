"""
Architecture dependent functionalities
----------
To increase the accuracy of ANN predictions, MieNet's default ANNs are trained on six seperate
regions of the parameter space. These architecture functions create masks to filter the given
inputs based which region they belong to, so the most accurate ANN for that input is used.

MieNet's default ANNs use the six_network architecture function, which filters inputs into three
wavelength and two size parameter regions. The three_network function divides inputs into just
three wavelength regions.

ANNs created by MieNet's train_ai_model function does not allow for training on different input
parameter spaces and is classified as "one_network" (as no masks are required for these
architecture types, there is no one_network architecture function).

If you choose to create your own models using another method and utilize an input split, add an
architecture function to this file. The name of the function must be the same as the architecture
parameter of your model in the config file. Architecture functions must have inputs, dependencies,
and scale as function parameters, and returns masks to work with MieNet's ai_efficiencies.
"""
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
