# HIRAXmcmc
An MCMC pipeline for cosmological parameter estimation from the 21cm estimated power spectra from HIRAX m-mode formalism based simulations

## Pre-requisites
1. CAMB and CLASS (Requires Cython) python wrappers
2. GCC, OpenMPI
3. Python libraries: h5py, configobj, wheel, twine, setuptools, mpi4py

## Installation
1. Clone the repository using "git clone ..."
2. Change the working directory to the HIRAXmcmc directory
3. Run "python setup.py bdist_wheel"
4. Run "pip install dist/hiraxmcmc*.whl"
