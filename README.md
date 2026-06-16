# HIRAXmcmc
An MCMC pipeline for cosmological parameter estimation from the 21cm estimated power spectra from HIRAX m-mode formalism based simulations

## Pre-requisites
1. CAMB and CLASS (Requires Cython) python wrappers
2. GCC, OpenMPI
3. Python libraries: ```h5py```, ```configobj```, ```wheel```, ```twine```, ```setuptools```, ```mpi4py```

## Installation
1. Clone the repository using ```git clone ...```
2. Change the working directory to the HIRAXmcmc directory
3. Run bootstrap script (installs uv if missing, then creates/syncs .venv with dev deps): ```./setup_venv.sh```
4. Build distribution files: ```uv build```
5. Install the built wheel: ```uv pip install dist/hiraxmcmc*.whl```

## Basic run - example
1. Copy an example input file from the location ```</path/to/cloned-rep/>hiraxmcmc/inputfiles/input_example_fcdep.json``` to wherever you desire. This particular file *input_example_fcdep.json* is for a mcmc run for redshift dependent parameters, h(z), dA(z), and f(z) for a particular frequency channel of the HIRAX m-mode sims run.
2. Go to the location of the input file. (You can also simply go to the original location of the input file.)
3. Run ```uv run python </path/to/hiraxmcmc/>scripts/mcmc.py </path/to/input-file/>input_example_fcdep.json```

## Important Notes:
1. For running the MCMC over frequency dependent scaling parameters (Eg. ```[h(z), d_A(z), f(z)]```), it is very important that the parameters specified in the input file have ```(z)``` in their names.
