from setuptools import find_packages, setup


with open('requirements.txt', 'r') as fh:
    requires = fh.read().split()


setup(
    name='hiraxmcmc',
    packages=find_packages(),
    version='0.1.1',
    description='MCMC pipeline for parameter estimation from 21cm PS in HIRAX m-mode simulations',
    author='Viraj Nistane',
    license='MIT',
    ## install_requires=requires,
    install_requires = [],
    include_package_data=True,
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
