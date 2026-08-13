""" MieNet class """
# pylint: disable=C0415,R0902,R0912,R0914,R0915

import os
import glob
import numpy as np
import miepython as mie

from .grid import grid_efficiencies
from .sub_functions import read_in_refindex, calculate_subradii, initialize_ai_models, select_best_dataset
from .mixing_theory import mixing_theory
from .data_handling import get_models
from . import architecture_functions


class MieNet:
    """
    MieNet class to calculate mie opacities using one of three methods:
    - efficiencies: Use LLL and miepython and perform full calcu
    - ai_efficiencies
    - grid_efficiencies
    """

    # ==== Import functions from sub-files ========================================================
    from .grid import grid_efficiencies, produce_efficiency_grid, load_grid_efficiency

    def __init__(self, use_ai=True, default_model_location=None, mute=True, load_ai_model = 'all'):
        """
        Constructor

        Parameters
        ----------
        use_ai : bool
            If False, AI will be disabled. This allows to use MieNet without installing tensorflow.
        load_ai_model : str
            Which AI model to load. Defualt is 'all', which loads all models. User can input
            model names to load a specific model.
        default_model_location : str, optional
            Location of opacity models. If none, MieNet defaults are used.
        mute : bool, optional
            If True, MieNet will produce no diagnostic outputs and runs quietly.
        """

        # ==== General preparations ===============================================================
        # Load species models from files
        self.files = glob.glob(os.path.dirname(__file__) + '/opacity_files/*.refrind')
        self.available_species = [os.path.basename(path).split('/')[0][:-8] for path in self.files]
        # save ai initialization state
        self.use_ai = use_ai
        self.load_ai_model = load_ai_model
        # save efficiency calc state
        self.auto = False
        # save mute preference
        self.mute = mute

        # ==== Prepare Neural Network =============================================================
        if use_ai:

            # models location
            if default_model_location is not None:
                self.model_path = default_model_location
            # default models location
            else:
                self.model_path = os.path.join(os.path.dirname(__file__), '../models/')

            # only load models if the model files exist
            models = glob.glob(self.model_path + '*.keras')
            if models:
                # load models
                self.models_dict = initialize_ai_models(load_ai_model, self.model_path)

        # ==== List of default datasets
        # user input models location
        if default_model_location is not None:
            self.data_path = default_model_location
        # default models location
        else:
            self.data_path = os.path.join(os.path.dirname(__file__), '../models/')

        # ==== Load predetermined grid dataset
        # default datasets
        self.default_grids = {}
        self.load_grid_efficiency()


    def ai_efficiencies(self, wavelength, particle_size, volume_mixing_ratios):
        """
        Calculate mie coefficients using a pre-trained neural network.

        Parameters
        ----------
        wavelength : np.ndarray or float of size N
            Wavelength of the light [micron]
        particle_size : np.ndarray or float of size M
            Size of the cloud particle [micron]
        volume_mixing_ratios : dict of np.ndarray or float of size M for each species
            Fraction of each cloud material given as float or array

        Return
        ------
        optical properties : np.ndarray of size (M, N)
            extinction coefficient, scattering coefficient, and asymmetries parameter
        """

        # ==== network intialization & retrieval ==================================================

        # check if neural network is initalised
        if not self.use_ai:
            raise

        if self.load_ai_model == 'all':

            # find all models that include all species
            best_model = select_best_dataset('model', self.auto, volume_mixing_ratios, self.models_dict)

            # add zero array to vmr dictionary if using less than the total amount of species
            if len(volume_mixing_ratios.keys()) != len(best_model[1]):
                missing_species = [key for key in best_model[1] if key not in volume_mixing_ratios]
                for species in missing_species:
                    volume_mixing_ratios[species] = np.zeros_like(next(iter(volume_mixing_ratios.values())))

            # get info for the model
            model_dict = self.models_dict[best_model[0]]

        else:
            # model info
            best_model = [self.load_ai_model, self.models_dict['species']]

            # add zero array to vmr dictionary if using less than the total amount of species
            if len(volume_mixing_ratios.keys()) != len(best_model[1]):
                missing_species = [key for key in best_model[1] if key not in volume_mixing_ratios]
                for species in missing_species:
                    volume_mixing_ratios[species] = np.zeros_like(next(iter(volume_mixing_ratios.values())))

            # check correct model is initialized for given volume mixing ratios
            if sorted(best_model[1]) != sorted(volume_mixing_ratios.keys()):
                raise ValueError("Incorrect AI model initialized for this mixture")

            # get info for chosen model
            model_dict = self.models_dict

        # ==== Input checks =======================================================================

        # check inputs are correct type
        if (not isinstance(wavelength, np.ndarray)
                and not isinstance(wavelength, (float, int))):
            print('Wavelength must be of type np.ndarray or float')
        if (not isinstance(particle_size, np.ndarray)
                and not isinstance(particle_size, (float, int))):
            print('Particle size must be of type np.ndarray or float')
        if (not isinstance(volume_mixing_ratios, dict)
                and not isinstance(volume_mixing_ratios, (float, int))):
            print('Volume mixing ratio must be of type dict or float')

        # convert floats to arrays
        if isinstance(wavelength, (float, int)):
            wavelength = np.array([wavelength])
        if isinstance(particle_size, (float, int)):
            particle_size = np.array([particle_size])
        for key, ratios in volume_mixing_ratios.items():
            if isinstance(ratios, (float, int)):
                volume_mixing_ratios[key] = np.array([ratios])

        # make all possible combinations of wavelength & particle size
        final_wavelength = np.repeat(wavelength, len(particle_size))
        final_particle_size = np.tile(particle_size, len(wavelength))

        # reorder volume mixing ratios and turn into array
        vmr_arr = {key: volume_mixing_ratios[key] for key in best_model[1]}
        vmr = np.array(list(vmr_arr.values())).T

        if len(set(map(len, volume_mixing_ratios.values()))) != 1 and not self.mute:
            print('Volume mixing ratios must have same shape')

        vmr_sum = np.sum(vmr, axis=1)
        if any(vmr_sum != 1) and not self.mute:
            print('Volume mixing ratios do not add up to 1. The ratios have been renormalized.')
        idx = np.asarray(np.where(vmr_sum != 1))
        for i in idx[0]:
            vmr[i] = vmr[i] / sum(vmr[i])

        # make volume mixing ratios have the same dimensions as final wavelength & final
        # particle size
        final_vmr = np.tile(vmr, (len(wavelength), 1))

        # ==== Prepare model ======================================================================
        # stack inputs
        inputs = np.stack((np.asarray(np.log10(final_wavelength)),
                           np.asarray(np.log10(final_particle_size)),
                           np.asarray(final_vmr[:,0]),
                           np.asarray(final_vmr[:,1])), axis=1)

        # prepare output
        extinction = np.zeros((len(inputs), 1))
        scattering = np.zeros((len(inputs), 1))
        asymmetry = np.zeros((len(inputs), 1))

        # get architecture-dependent masks
        arch = model_dict['architecture']
        arch_func = getattr(architecture_functions, arch)

        masks = arch_func(inputs, model_dict['dependencies'])

        # predict outputs
        for i, mask in enumerate(masks):
            if mask.any():
                extinction[mask], scattering[mask], asymmetry[mask] = \
                model_dict['models'][i].predict(inputs[mask])

        # reshape outputs
        qext = 10**extinction[:, 0].reshape((len(wavelength), len(particle_size))).T
        qsca = 10**scattering[:, 0].reshape((len(wavelength), len(particle_size))).T
        asym = asymmetry[:, 0].reshape((len(wavelength), len(particle_size))).T

        return qext, qsca, asym

    def efficiencies(self, wavelength, particle_size, volume_mixing_ratios, theory='LLL'):
        """
        Calculate mie coefficients using mie python and LLL Approximation.

        Parameters
        ----------
        wavelength : np.ndarray or float of size N
            Wavelength of the light [micron]
        particle_size : np.ndarray or float of size M
            Size of the cloud particle [micron]
        volume_mixing_ratios : dict of np.ndarray or float of size M for each species
            Fraction of each cloud material given as float or array
        theory : str, optional
            Mixing theory used, can either be 'LLL' (Default) or 'Burggeman'

        Return
        ------
        optical properties : np.ndarray of size (M, N)
            extinction coefficient, scattering coefficient, and asymmetries parameter
        """
        # ==== Prepare inputs =====================================================================

        # check inputs are correct type
        if not self.mute:
            if (not isinstance(wavelength, np.ndarray)
                    and not isinstance(wavelength, (float, int))):
                print('Wavelength must be of type np.ndarray or float')
            if (not isinstance(particle_size, np.ndarray)
                    and not isinstance(particle_size, (float, int))):
                print('Particle size must be of type np.ndarray or float')
            if (not isinstance(volume_mixing_ratios, dict)
                    and not isinstance(volume_mixing_ratios, (float, int))):
                print('Volume mixing ratio must be of type dict or float')

        # convert floats to arrays
        if isinstance(wavelength, (float, int)):
            wavelength = np.array([wavelength])
        if isinstance(particle_size, (float, int)):
            particle_size = np.array([particle_size])
        for key, ratios in volume_mixing_ratios.items():
            if isinstance(ratios, (float, int)):
                volume_mixing_ratios[key] = np.array([ratios])

        # define species list according to entries in vmr
        species_list = list(volume_mixing_ratios.keys())

        # check if all species are available
        for spec in species_list:
            if spec not in self.available_species:
                raise ValueError("The species " + spec + " is not available")

        # create array with vmr values
        vmr = np.zeros((len(particle_size), len(species_list)))
        for s, spec in enumerate(species_list):
            vmr[:, s] = volume_mixing_ratios[spec]

        # check vmr is the correct input shape
        if len(set(map(len, volume_mixing_ratios.values()))) != 1 and not self.mute:
            print('Volume mixing ratios must have same shape')
        if len(particle_size) != len(vmr):
            print('Particle size and volume mixing ratio must have same shape')

        vmr_sum = np.sum(vmr, axis=1)
        if any(vmr_sum != 1) and not self.mute:
            print('Volume mixing ratios do not add up to 1. The ratios have been renormalized.')
        idx = np.asarray(np.where(vmr_sum != 1))
        for i in idx[0]:
            vmr[i] = vmr[i] / sum(vmr[i])

        # ==== Radius averaging ===================================================================
        sub_rad, vmr = calculate_subradii(particle_size, vmr)

        # ==== Load models for each species from files and get refractive index =====================
        ref_index = read_in_refindex(species_list, wavelength, self.files)

        # ==== Combination of all wavelengths and particle size ===================================
        final_wavelength = np.repeat(wavelength, len(sub_rad))
        final_sub_rad = np.tile(sub_rad, len(wavelength))
        final_vmr = np.tile(vmr, (len(wavelength), 1))
        final_ref_index = np.repeat(ref_index, len(sub_rad), axis=1)

        mixed_ref_index = mixing_theory(
            final_wavelength, final_ref_index, final_vmr, theory=theory
        )

        # ==== Calculate Mie Efficiencies =========================================================
        size_param = (2.0 * np.pi * final_sub_rad) / final_wavelength

        # qe_temp = extinction, qs_temp = scattering, g_temp = asymmetry
        qe_temp, qs_temp, _, g_temp = mie.efficiencies_mx(mixed_ref_index, size_param)

        # ==== Prepare outputs ====================================================================
        if len(sub_rad) != len(particle_size):
            extinction = np.mean(
                qe_temp.reshape(len(particle_size) * len(wavelength), 6), axis=1
            ).reshape(len(wavelength), len(particle_size)).T
            scattering = np.mean(
                qs_temp.reshape(len(particle_size) * len(wavelength), 6), axis=1
            ).reshape(len(wavelength), len(particle_size)).T
            asymmetry = np.mean(
                g_temp.reshape(len(particle_size) * len(wavelength), 6), axis=1
            ).reshape(len(wavelength), len(particle_size)).T

        else:
            extinction = qe_temp.reshape(len(wavelength), len(particle_size)).T
            scattering = qs_temp.reshape(len(wavelength), len(particle_size)).T
            asymmetry = g_temp.reshape(len(wavelength), len(particle_size)).T

        return extinction, scattering, asymmetry

    def auto_efficiencies(self, wavelength, particle_size, volume_mixing_ratios, theory='LLL'):
        """
        Calculate mie coefficients using mie fastest method available.

        Parameters
        ----------
        wavelength : np.ndarray or float of size N
            Wavelength of the light [micron]
        particle_size : np.ndarray or float of size M
            Size of the cloud particle [micron]
        volume_mixing_ratios : dict of np.ndarray or float of size M for each species
            Fraction of each cloud material given as float or array
        theory : str, optional
            Mixing theory used, can either be 'LLL' (Default) or 'Burggeman'

        Return
        ------
        optical properties : np.ndarray of size (M, N)
            extinction coefficient, scattering coefficient, and asymmetries parameter
        """
        auto = True

        # check grids
        best_dataset = select_best_dataset('grid', auto, volume_mixing_ratios, self.default_grids)

        # use grid if it exists
        if best_dataset[0] is not None:
            extinction, scattering, asymmetry = grid_efficiencies(wavelength, particle_size, volume_mixing_ratios)
            return extinction, scattering, asymmetry

        # check models if no grids
        else:
            best_model = select_best_dataset('model', auto, volume_mixing_ratios, self.models_dict)

            # use model if it exists
            if best_model[0] is not None:
                extinction, scattering, asymmetry = self.ai_efficiencies(wavelength, particle_size, volume_mixing_ratios)
                return extinction, scattering, asymmetry

            # use full calculations if no models or grids exist
            else:
                extinction, scattering, asymmetry = self.efficiencies(wavelength, particle_size, volume_mixing_ratios, theory)
                return extinction, scattering, asymmetry


    def download_models(self):
        '''
        Download MieNet models from Zenodo and load all models/specified model.
        '''
        # check if files already exsist
        models = glob.glob(self.model_path + '*.keras')
        if not models:
            # download models
            get_models(self.model_path)

            # load models
            self.models_dict = initialize_ai_models(self.load_ai_model, self.model_path)