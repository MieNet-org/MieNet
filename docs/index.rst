.. figure:: TBD ADD LOGO
   :alt: MiNet logo
   :align: center

Welcome to MieNet
=================

.. image:: https://codecov.io/gh/MieNet-org/MieNet/graph/badge.svg?token=FDPN19ZPGW
.. image:: https://raw.githubusercontent.com/MieNet-org/MieNet/refs/heads/main/.github/workflows/pylint.svg
.. image:: https://raw.githubusercontent.com/MieNet-org/MieNet/refs/heads/main/.github/workflows/python.svg

MieNet is a software package to calculate the opacities of heterogeneous cloud particles. To accelerate the otherwise slow calculations, it provides three methods:

- 'Full': Perform effective refractive indices and Mie theory calculation.
- 'Grid': Precalculate opacity grids provide fast approximation to cloud particle opacities.
- 'AI': Artificial Neural Networks trained to deliver high accuracy at a fraction of the computation time.

To choose the fastest method available:

- 'Auto': Use fastest MieNet method available to calculate Mie coefficients.

Fully trained models are provided on `Zenodo <https://zenodo.org/records/20346256>`_, or can be trained by yourself according to your needs. MieNet is under active development and contributions are welcomed. If you want to run MieNet checkout the `quick start guide <Install_And_Quick_Start.ipynb>`_.

A Note on Requirements
----------------------
To use MieNet's AI functionalities, `TensorFlow <https://www.tensorflow.org/>`_ is requied, which currently only works with Python 3.9-3.12. We recommend using versions 3.0 and older of `miepython <https://miepython.readthedocs.io/en/latest/>`_ for faster computation times.

Credit
------
If you use MieNet, please cite the following papers:

- `Attaway et al. (2026) <LINK TO PAPER>`_
- `Kiefer et al. (2026) <LINK TO PAPER>`_

Also consider citing the softwares MieNet is based on:

- `Prahl et al. (2026) <LINK TO PAPER>`_
- `Tensorflow citation (2026) <LINK TO PAPER>`_

.. toctree::
   :maxdepth: 1
   :caption: Contents

   Install_And_Quick_Start.ipynb
   Tutorials.rst
   Github <https://github.com/MieNet-org/MieNet>


