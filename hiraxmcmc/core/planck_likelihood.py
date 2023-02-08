#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:43:34 2022

@author: Viraj.Nistane
"""

import os,sys
import numpy as np
import matplotlib.pyplot as plt


from cobaya.model import get_model
from cobaya.yaml import yaml_load


info_txt = r"""
likelihood:
  planck_2018_highl_plik.TTTEEE_lite_native:
packages_path:
  /Users/Viraj.Nistane/Desktop/phdmywork/planck2
theory:
  classy:
    extra_args: {N_ur: 2.0328, N_ncdm: 1}
params:
  logA:
    prior: {min: 2, max: 4}
    ref: {dist: norm, loc: 3.05, scale: 0.001}
    proposal: 0.001
    latex: \log(10^{10} A_\mathrm{s})
    drop: true
  A_s: {value: 'lambda logA: 1e-10*np.exp(logA)', latex: 'A_\mathrm{s}'}
  n_s:
    prior: {min: 0.8, max: 1.2}
    ref: {dist: norm, loc: 0.96, scale: 0.004}
    proposal: 0.002
    latex: n_\mathrm{s}
  H0:
    prior: {min: 40, max: 100}
    ref: {dist: norm, loc: 70, scale: 2}
    proposal: 2
    latex: H_0
  omega_b:
    prior: {min: 0.005, max: 0.1}
    ref: {dist: norm, loc: 0.0221, scale: 0.0001}
    proposal: 0.0001
    latex: \Omega_\mathrm{b} h^2
  omega_cdm:
    prior: {min: 0.001, max: 0.99}
    ref: {dist: norm, loc: 0.12, scale: 0.001}
    proposal: 0.0005
    latex: \Omega_\mathrm{c} h^2
  m_ncdm: {renames: mnu, value: 0.06}
  Omega_Lambda: {latex: \Omega_\Lambda}
  YHe: {latex: 'Y_\mathrm{P}'}
  tau_reio:
    prior: {min: 0.01, max: 0.8}
    ref: {dist: norm, loc: 0.06, scale: 0.01}
    proposal: 0.005
    latex: \tau_\mathrm{reio}
"""
    
info = yaml_load(info_txt)
# info['packages_path'] = '/Users/Viraj.Nistane/Desktop/phdmywork/planck2'


model = get_model(info)

point = dict(zip(model.parameterization.sampled_params(),
                 model.prior.sample(ignore_external=True)[0]))
point.update({'omega_b': 0.0223, 'omega_cdm': 0.120, 'H0': 67.01712,
              'logA': 3.06, 'n_s': 0.966, 'tau_reio': 0.065})


logposterior = model.logposterior(point, as_dict=True)

print('   loglikelihoods:', logposterior["loglikes"])


















