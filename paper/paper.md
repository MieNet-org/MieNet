---
title: 'MieNet: A Python package for fast well-mixed cloud particle opacity calculations'
tags:
- python
- exoplanets
- cloud particle opacities
- neural networks
date: "TBD"
authors:
- name: Attaway Daisy
  orcid: "0009-0000-5047-8107"
  affiliation: '1'
  corresponding: true
- name: Kiefer Sven
  orcid: "0000-0003-1285-3433"
  affiliation: '1'
- name: Yinan Zhao 
  orcid: "0000-0002-5009-4645"
  affiliation: '1'
- name: Morley Caroline
  orcid: "0000-0002-4404-0456"
  affiliation: '1'
affiliations:
- name: Department of Astronomy, University of Texas at Austin, 2515 Speedway, Austin, TX 78712, USA
  index: 1
bibliography: paper.bib
---

# Summary
`MieNet` is a Python package for quick calculations of well-mixed cloud particle opacities. `MieNet` produces the extinction, scattering, and asymmetry Mie coefficients given the wavelength, cloud particle radius, and cloud species with their respective volume fraction (VMR). Cloud particles are assumed to be well-mixed and spherical. Artificial neural networks (ANNs), made with  `TensorFlow` [@abadi_tensorflow_2016], are used to overcome the computational challenge associated with well-mixed cloud particles. There are three calculation methods available: full calculations, grid interpolations, or ANN predictions.

# Statement of Need
Modeling exoplanet clouds is necessary to better understand the formation of exoplanets and to interpret their observations [@helling_exoplanet_2019; @powell_transit_2019]. Cloud models require assumptions to be made about the shape [@m_i_mishchenko_light_2000; @min_shape_2003; @min_absorption_2006; @samra_mineral_2020], formation [@gao_microphysics_2018; @helling_exoplanet_2019] and composition [@gao_universal_2021; @kiefer_why_2024]  \red{virga3} of cloud particles. Exoplanet clouds are often assumed to be homogeneous, or comprised of only one material, but theoretical models [@helling_dust_2006; @ormel_arcis_2019; @min_arcis_2020; @lee_modelling_2023] have predicted and some observations [@dyrek_so2_2024; @hoch_silicate_2025], show evidence of heterogeneous cloud particles. However, modeling these heterogeneous clouds is computationally intensive.

Mie theory [@mie] uses an effective refractive index and size parameter, which is related to the cloud particle radius and wavelength of light. Mie calculation is already a slow process, but for heterogeneous cloud particles, calculating the effective refractive indices adds additional computation time. The high dimensionality (wavelength, radius, VMR 1, VMR 2, etc.) of mixed Mie particles results in a large parameter space which makes grid calculations unfeasible when high resolutions are required. These computational challenges caused by calculating the effective refractive index is why machine learning techniques are necessary. 

# State of the Field
 Other studies have successfully applied neural networks to Mie theory and are able to make fast and accurate opacity calculations. However, their work utilizes different assumptions and is not applicable to well-mixed cloud particles. `MieAi` [@kumar_mieai_2024] and `NeuralMie` [@geiss_neuralmie_2025] both assume core-shell configuration and are intended for Earth climate and weather modeling. `Glitterin` [@lin_glitterin_2025}] focuses on debris disks and assumes non-spherical particles. `MieNet` is unique in its assumption of well-mixed, spherical cloud particles. It utilizes neural networks to overcome the computational challenge of effective refractive indices and optimizes well-mixed opacity calculations.

# Software Design
`MieNet` is designed with a focus on computational speed, easy integration, and adaptability. Functions to produce Mie coefficients are:
- `efficiencies`: fully calculate coefficients with user-defined mixing theory and Mie theory
- `ai_efficiencies`: use ANNs to predict coefficients
- `grid_efficiencies`: interpolate coefficients from a pre-calculated grid
- `auto_efficiencies`: choose the fastest available method (full, AI, or grid) to produce coefficients

All functions are easily interchangeable since they have the same input and output structure. The outputs are also similar to other Mie-theory codes, such as  `miepython` [@prahl_miepython_2026] or `PyMieScatt` [@sumlin_retrieving_2018], so working with `MieNet` is familiar and allows for easy integration of `MieNet` into existing code. 

For the `efficiencies` function, two mixing theories are available: Landau, Lifshitz, and Looyenga (LLL) [@landau_electrodynamics_1960; @looyenga_dielectric_1965}] and Bruggeman [@bruggeman_berechnung_1935]. Mixing theories evaluate the effective refractive index of a well-mixed particle. LLL and Bruggeman are the most common used in the field and have both been shown to match lab results in different setups [@kolokolova_scattering_2001; @voshchinnikov_effective_2007; @thomas_investigations_2009]. While these theories are similar, there are some differences [see @kiefer_why_2024]. `MieNet`'s current ANNs are trained on datasets made using the LLL approximation, because it is less prone to computational artifacts.

Users can create their own training set with either mixing theory using the `generate_training_set` function. Once a training set is made, one can use `train_ai_model` to train and save custom ANNs. These functionalities allows users to easily train `MieNet`-compatible ANNs for specific mixtures, wavelength and particle size ranges, or mixing theory that `MieNet`'s current ANNs do not cover. For a further discussion on how `MieNet`'s ANNs are trained, see \red{virga3}.

To create grids for the `grid_efficiencies` function, `MieNet` contains `produce_efficiency_grid` and `load_grid_efficiency` to allow users to precalculate and load grids. However, these grids are limited to lower resolution, as the dataset size required for a high-accuracy grid is too large.

# Research Impact Statement
Mie theory is used for calculating the opacity of exoplanet clouds by the vast majority of the field [@powell_transit_2019; @grant_JWST-TST_2023; @steinrueck_limb_2025; @kiefer_under_2024]. `MieNet`'s use of ANNs and optimization of mixed particle Mie calculations makes the inclusion and study of heterogeneous cloud particles feasible for the research community. This package has already shown direct impact in works that have investigated interplanetary dust particle (IDP) infall as an exoplanet cloud seeding mechanism \red{meteo paper} and the mixing state of YSES-1c's cloud particles \red{virga v3}. `MieNet` is integrated into the `Virga` [@batalha_condensation_2026; moran_fractal_2025] and `PICASO` [@batalha_exoplanet_2019; @mukherjee_picaso_2023; @mang_PICASO_2026] codes, which are publicly available, open-source atmospheric models commonly used by the exoplanet community [@ahrer_early_2023; @hamill_reflected-light_2024; @mukherjee_impact_2026; @beichman_worlds_2025; @kiefer_connecting_2026; @bardalez_gagliuffi_jwst_2025; @he_optical_2024]. The similarity of MieNet's outputs to other common Mie-theory libraries, like `miepython` [@prahl_miepython_2026] or `PyMieScatt` [@sumlin_retrieving_2018], allows for easy integration of heterogeneous particles into other cloud and planetary codes which use these libraries [@bhattacharjee_thick_2025; @xie_water_2025; @shefferson-nagata_radiatively_2026; @pater_characterization_2026; @espinoza_inhomogeneous_2024]. While this package was developed for exoplanet atmospheres, `MieNet` can be utilized in other areas that also use Mie theory, such as climate and weather [@ji_mechanism_2026; @chakrabarty_shortwave_2023,; @fierce_radiative_2020; @karlsson_long-term_2021; @karlsson_physical_2022], protoplanetary and debris disks [@andrews_resolved_2011; @bhattacharjee_thick_2025; @xie_water_2025; @wang_constraining_2021; @kranhold_ironii_2026], solar-system [@pater_characterization_2026; @hedman_features_2026; @dai_investigation_2024; @vidwans_size_2022], interstellar medium dust [@compiegne_global_2011; @galliano_non-standard_2011; @guerlet_global_2014; @de_looze_dust_2017; @draine_magnetic_2013], optical sensing and imaging [@hagan_assessing_2020; @tryner_effects_2020; @kemppinen_imaging_2020; @thiele_single-protein_2024; @carrico_humidified_2021] and other applications.

# AI Usage Disclosure
AI tools were used during software development for debugging and generating minor code snippets, which were all human-validated. No generative AI tools were used in the writing of this manuscript or MieNet documentation.

# Acknowledgements

# References
