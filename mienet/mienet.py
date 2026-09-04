""" MieNet class """
# pylint: disable=C0415,R0902,R0913,R0912,R0914,R0915,R0917

import sys
import os
import glob
import numpy as np
import miepython as mie

from .sub_functions import (read_in_refindex, calculate_subradii,
                            select_best_dataset, input_check)
from .mixing_theory import mixing_theory
from .model_handling import get_models
from . import architecture_functions


class MieNet:
    """
    MieNet class to calculate mie opacities. For more information please see the
    Documentation.
    """

    # ==== Import functions from sub-files ==============================================
    from .grid import grid_efficiencies, produce_efficiency_grid, load_grid_efficiency
    from .model_handling import (generate_training_set, train_ai_model,
                                 initialize_ai_models)

    def __init__(self, use_ai=True, default_data_location=None, mute=True,
                 load_ai_model='all', grid_file=None):
        """
        Constructor

        Parameters
        ----------
        use_ai : bool
            If False, AI will be disabled. This allows to use MieNet without installing
            tensorflow.
        default_data_location : str, optional
            Location of opacity data and/or grids. If none, MieNet defaults are used.
        mute : bool, optional
            If True, MieNet will produce no diagnostic outputs and runs quietly.
        load_ai_model : str
            Which AI model to load. Default is 'all', which loads all models. User can
            input model names to load a specific model.
        grid_file : str, optional
            If a grid file is given, only this file will be loaded.
        """

        # ==== General preparations =====================================================
        # save user inputs
        self.use_ai = use_ai
        self.use_grid = False  # this is default until grids are loaded
        self.mute = mute
        if load_ai_model != 'all':
            if isinstance(load_ai_model, str):
                self.load_ai_model = [load_ai_model]
        else:
            self.load_ai_model = load_ai_model

        # working variables
        self.force_disabled_ai = False  # This will give a warning if MieNet breaks

        # Load species data from files
        self.files = glob.glob(os.path.dirname(__file__) + '/opacity_files/*.refrind')
        self.available_species = [os.path.basename(path).split('/')[0][:-8] for path in self.files]

        # ==== Intro
        if not self.mute:
            print('=============================================')
            print('              Welcome to MieNet              ')
            print('=============================================')

        # ==== Data location
        # user input data location
        if default_data_location is not None:
            self.data_path = default_data_location
        # default data location
        else:
            self.data_path = os.path.join(os.path.dirname(__file__), '../data/')

        # ==== Prepare Neural Network ===================================================
        self.models_dict = {}  # default initialisation
        if self.use_ai:
            # check if right python version for tensorflow
            if (sys.version_info[0] > 3 or
                    (sys.version_info[0] == 3 and sys.version_info[1] > 12)):
                ver = str(sys.version_info[0]) + '.' + str(sys.version_info[1])
                print(f'[WARN] Your python version ({ver}) is to new and might not work '
                      f'with tensorflow.')
            # initialize ai models
            self.initialize_ai_models()

        # ==== Load predetermined grid dataset
        # default datasets
        self.grids_dict = {}
        self.load_grid_efficiency(file_name=grid_file)

        # ==== Information about the status of MieNet
        if not self.mute:
            print('[INFO] MieNet successfully initialised')
            print('   -> Data location: ' + self.data_path)
            if self.use_ai:
                print(f'   -> AAN models available: {len(self.models_dict)}')
            else:
                print('   -> ANN models are unavailable')
            print(f'   -> Grid models available: {len(self.grids_dict)}')


    def ai_efficiencies(self, wavelength, particle_size, volume_mixing_ratios, theory='LLL'):
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
        theory: str, optional
            Mixing theory used to train ai model

        Return
        ------
        optical properties : np.ndarray of size (M, N)
            extinction coefficient, scattering coefficient, and asymmetries parameter
        """

        # ==== network initialization & retrieval =======================================

        # check if neural network is initialized
        if not self.use_ai:
            if self.force_disabled_ai:
                raise ValueError('[ERROR] No ANNs were successfully loaded, '
                                 'ai_efficiencies is not available.')
            raise ValueError('[ERROR] use_ai must be set to true to use ai_efficiencies.')

        # check at least one correct model is initialized if using specific model
        if self.load_ai_model != 'all':
            if not any(set(volume_mixing_ratios).issubset(self.models_dict[key]['species'])
                       for key in self.load_ai_model):
                raise ValueError(
                    "[ERROR] No correct AI model initialized for this mixture:\n"
                    "   -> Species selected: " + str(volume_mixing_ratios.keys()) + "\n"
                    "   -> Find out which ANNs are loaded by setting MieNet(mute=False) "
                )

        # find all models that include all species
        best_model = select_best_dataset(
            'model', wavelength, particle_size, volume_mixing_ratios, self.models_dict,
            theory=theory
        )

        # add zero array to vmr dictionary if using less than the total amount of species
        if len(volume_mixing_ratios.keys()) != len(best_model[1]):
            missing_species = [key for key in best_model[1]
                               if key not in volume_mixing_ratios]
            for species in missing_species:
                volume_mixing_ratios[species] = (
                    np.zeros_like(next(iter(volume_mixing_ratios.values()))))

        # get info for the model
        model_dict = self.models_dict[best_model[0]]

        # ==== Input checks =============================================================
        wavelength, particle_size, vmr = input_check(
            wavelength, particle_size, volume_mixing_ratios, best_model[1], self.mute
        )

        # ==== Information
        if not self.mute:
            print('[INFO] Perform AAN calculation for ',
                  list(volume_mixing_ratios.keys()))

        # make all possible combinations of wavelength & particle size
        final_wavelength = np.repeat(wavelength, len(particle_size))
        final_particle_size = np.tile(particle_size, len(wavelength))

        # make volume mixing ratios have the same dimensions as final wavelength & final
        # particle size
        final_vmr = np.tile(vmr, (len(wavelength), 1))

        # ==== Prepare model ============================================================
        # define input array
        num_inputs = 2 + (final_vmr.shape[1] - 1)
        inputs = np.zeros((len(final_wavelength), num_inputs))

        # assign wavelength input
        if model_dict['scale']['wavelength'] == 'log':
            inputs[:, 0] = np.log10(final_wavelength)
        else:
            inputs[:, 0] = final_wavelength

        # assign particle size input
        if model_dict['scale']['particle_size'] == 'log':
            inputs[:, 1] = np.log10(final_particle_size)
        else:
            inputs[:, 1] = final_particle_size

        # assign volume mixing ratio inputs
        for material in range(final_vmr.shape[1] - 1):
            inputs[:, 2 + material] = final_vmr[:, material]

        # prepare output
        extinction = np.zeros((len(inputs), 1))
        scattering = np.zeros((len(inputs), 1))
        asymmetry = np.zeros((len(inputs), 1))

        # get architecture-dependent masks
        arch = model_dict['architecture']

        # predict models if no masking required
        if arch == 'one_network':
            extinction, scattering, asymmetry = model_dict['models'][0].predict(inputs)

        # get masks from arch function
        else:
            arch_func = getattr(architecture_functions, arch)

            masks = arch_func(inputs, model_dict['dependencies'], model_dict['scale'])

            # predict outputs
            for i, mask in enumerate(masks):
                if mask.any():
                    print(inputs.shape, inputs[mask].shape)
                    extinction[mask], scattering[mask], asymmetry[mask] = \
                    model_dict['models'][i].predict(inputs[mask])

        # reshape outputs
        ext = extinction[:, 0].reshape((len(wavelength), len(particle_size))).T
        sca = scattering[:, 0].reshape((len(wavelength), len(particle_size))).T
        asym = asymmetry[:, 0].reshape((len(wavelength), len(particle_size))).T

        # check extinction scaling
        if model_dict['scale']['extinction'] == 'log':
            qext = 10**ext
        else:
            qext = ext

        # check scattering scaling
        if model_dict['scale']['scattering'] == 'log':
            qsca = 10 ** sca
        else:
            qsca = sca

        return qext, qsca, asym

    def efficiencies(self, wavelength, particle_size, volume_mixing_ratios, theory=None):
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
            Mixing theory used, can either be 'LLL' (Default) or 'Bruggeman'

        Return
        ------
        optical properties : np.ndarray of size (M, N)
            extinction coefficient, scattering coefficient, and asymmetries parameter
        """
        # ==== Prepare inputs ===========================================================
        if theory is None:
            theory = 'LLL'

        # define species list according to entries in vmr
        species_list = list(volume_mixing_ratios.keys())

        # check if all species are available
        for spec in species_list:
            if spec not in self.available_species:
                raise ValueError(
                    "[ERROR] The species " + spec + " is not available\n"
                    "   -> Available are: " + str(self.available_species)
                )

        # check input validity
        wavelength, particle_size, vmr = input_check(
            wavelength, particle_size, volume_mixing_ratios, species_list, self.mute
        )

        # ==== Information
        if not self.mute:
            print('[INFO] Perform full Mie calculation for ',
                  list(volume_mixing_ratios.keys()))

        # ==== Radius averaging =========================================================
        sub_rad, vmr = calculate_subradii(particle_size, vmr)

        # ==== Load data for each species from files and get refractive index ===========
        ref_index = read_in_refindex(species_list, wavelength, self.files)

        # ==== Combination of all wavelengths and particle size =========================
        final_wavelength = np.repeat(wavelength, len(sub_rad))
        final_sub_rad = np.tile(sub_rad, len(wavelength))
        final_vmr = np.tile(vmr, (len(wavelength), 1))
        final_ref_index = np.repeat(ref_index, len(sub_rad), axis=1)

        mixed_ref_index = mixing_theory(
            final_wavelength, final_ref_index, final_vmr, theory=theory
        )

        # ==== Calculate Mie Efficiencies ===============================================
        size_param = (2.0 * np.pi * final_sub_rad) / final_wavelength

        # qe_temp = extinction, qs_temp = scattering, g_temp = asymmetry
        qe_temp, qs_temp, _, g_temp = mie.efficiencies_mx(mixed_ref_index, size_param)

        # ==== Prepare outputs ==========================================================
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

    def auto_efficiencies(self, wavelength, particle_size, volume_mixing_ratios,
                          theory=None):
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

        if self.use_ai:
            # check models
            best_model = select_best_dataset(
                'model', wavelength, particle_size, volume_mixing_ratios,
                self.models_dict, theory, False
            )

            # use model if it exists
            if best_model[0] is not None:
                extinction, scattering, asymmetry = self.ai_efficiencies(
                    wavelength, particle_size, volume_mixing_ratios, theory
                )
                return extinction, scattering, asymmetry

        best_dataset = select_best_dataset(
            'grid', wavelength, particle_size, volume_mixing_ratios,
            self.grids_dict, theory, False
        )

        # use grid if it exists
        if best_dataset[0] is not None:
            extinction, scattering, asymmetry = self.grid_efficiencies(
                wavelength, particle_size, volume_mixing_ratios, theory
            )
            return extinction, scattering, asymmetry

        extinction, scattering, asymmetry = self.efficiencies(
            wavelength, particle_size, volume_mixing_ratios, theory
        )
        return extinction, scattering, asymmetry


    def download_models(self, overwrite=True, url=None):
        """
        Download MieNet data from Zenodo and load all data/specified model.

        Parameters
        ----------
        overwrite : bool, optional
            If True, old files will be overwritten.
        url : str, optional
            Download link. If None is given, use the default zotero repository.
        """
        # info
        if not self.mute and url is None:
            print('[INFO] Downloading data from Zenodo')
        else:
            print('[INFO] Downloading data from: ' + url)
        print('   -> Saving data to: ' + self.data_path)

        # download models
        get_models(self.data_path, overwrite, url)

        # done
        if not self.mute:
            print('   -> Download successful')

        # load data
        self.load_grid_efficiency()
        if self.use_ai:
            self.initialize_ai_models()
