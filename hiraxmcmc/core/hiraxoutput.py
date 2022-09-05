#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import h5py as hh

import hiraxmcmc
from hiraxmcmc.util.basicfunctions import *
from hiraxmcmc.util.cosmoparamvalues import ParametersFixed

# =============================================================================
# HIRAX cov matrix and k-params
# =============================================================================


class HiraxOutput:
    
    def __init__(self, inputforhiraxoutput):
        

        
        self.modulelocation = os.path.dirname(hiraxmcmc.__file__)
        self.hiraxrundir_name = inputforhiraxoutput['result_dir_name']
        self.hiraxrundir_fullpath = os.path.join(self.modulelocation, 'mmoderesults', self.hiraxrundir_name)
        self.redshift = inputforhiraxoutput['redshift']
        self.psetype = inputforhiraxoutput['estimator_type']
        self.klmode = inputforhiraxoutput['klmode']
        self.uname = os.uname()[1]
        self.parameters_fixed = ParametersFixed()
        self.h = self.parameters_fixed.h_fid
        
        self.psfileload = hh.File(
            os.path.join(
                self.hiraxrundir_fullpath, 
                'draco/psmc_%s_wnoise_fgfilter_%s_group_0.h5'%(self.psetype,self.klmode) ),'r')
        self.fisherfileload = hh.File(
            os.path.join(
                self.hiraxrundir_fullpath, 
                'drift_prod_hirax_survey_49elem_7point_64bands/bt/%s/psmc_%s_1threshold/fisher.hdf5'%(self.klmode,self.klmode)),'r')
        
        
        # "None"-Initiated attributes
        self.cinv = None
        self.extent1 = None
        
        
    
    def __repr__(self):
        return f"HiraxOutput({self.hiraxrundir_name!r}, {self.psetype!r}, {self.redshift!r}, {self.klmode!r})"
    
    
    
    ##### 
    # If h-units are to be removed, confirm the h-dependence of the covariance matrix from m-mode
    #####
    @property
    def covhirax(self):
        self.cinv = self.psfileload['C_inv'][:];
        
        if self.psetype == "unwindowed":
            covhirax = np.linalg.inv(self.cinv.T.reshape(int(self.cinv.shape[0]* self.cinv.shape[1]),int(self.cinv.shape[2]* self.cinv.shape[3])));
        elif self.psetype == "minvar":
            M = np.diag(self.cinv.T.reshape(int(self.cinv.shape[0]*self.cinv.shape[1]),int(self.cinv.shape[2]*self.cinv.shape[3])).sum(axis=1)** -1)
            covhirax = np.matmul(M, np.matmul(self.cinv.T.reshape(int(self.cinv.shape[0]*self.cinv.shape[1]),int(self.cinv.shape[2]*self.cinv.shape[3])), M.T))
        # return 0.05**2 * np.identity(len(covhirax)) # 
        covhirax1 = covhirax #/ self.h**3
        return covhirax1
    
    # @property
    # def covhirax(self):
    #     return np.loadtxt(os.path.relpath(os.path.join(self.MCMCmodulespath,'viraj_cov_matrix.dat')))
    
    
    @property
    def ps_relative_estimated_from_hirax(self):
        return self.psfileload['powerspectrum'][:]
    
    
    
    def k_space_parameters(self):

        f2 = self.fisherfileload
        
        kpar = f2['kpar_center'][:] * self.h
        kpar_bands = f2['kpar_bands'][:] * self.h
        kpar_size = len(kpar_bands)-1
        self.kpar_size = kpar_size
        
        kperp = f2['kperp_center'][:] * self.h
        kperp_bands = f2['kperp_bands'][:] * self.h
        kperp_size = len(kperp_bands)-1;   
        self.kperp_size = kperp_size
        
        
        kpar = kpar.reshape((kperp_size, kpar_size)).T;
        kperp = kperp.reshape((kperp_size, kpar_size)).T;
        kcenter = (kpar**2 + kperp**2)**0.5;
        
        
        self.extentlower = (kperp_bands[0], kperp_bands[-1], kpar_bands[0], kpar_bands[-1])    # for <<origin='lower'>> in imshow
        
        kperp_center_1d = 0.5*(kperp_bands[1:]+kperp_bands[:-1])
        kpar_center_1d = 0.5*(kpar_bands[1:]+kpar_bands[:-1])
        
        #### for driftscan functions ####
        # kparb, kperpb = np.broadcast_arrays(kpar_bands[np.newaxis, :], kperp_bands[:, np.newaxis])
        kparb, kperpb = np.broadcast_arrays(kpar_bands[:,np.newaxis], kperp_bands[np.newaxis,:])
        
        kpar_start = kparb[:-1, 1:].flatten()
        kpar_end = kparb[1:, 1:].flatten()
        kpar_center = 0.5 * (kpar_end + kpar_start)
        
        kperp_start = kperpb[1:, :-1].flatten()
        kperp_end = kperpb[1:, 1:].flatten()
        kperp_center = 0.5 * (kperp_end + kperp_start)
        #### #### #### #### #### #### ###
        
        
        
        
        kpar_params = {'kpar':kpar, 'kpar_bands':kpar_bands, 'kpar_size':kpar_size, 'kpar_center_1d':kpar_center_1d, 
                       'kpar_start_flat':kpar_start , 'kpar_end_flat':kpar_end , 'kpar_center_flat':kpar_center}
        kperp_params = {'kperp':kperp, 'kperp_bands':kperp_bands, 'kperp_size':kperp_size, 'kperp_center_1d':kperp_center_1d,
                        'kperp_start_flat':kperp_start , 'kperp_end_flat':kperp_end , 'kperp_center_flat':kperp_center}
        kcenter_params = {'kcenter':kcenter }
        
        return kpar_params, kperp_params, kcenter_params
        
        
    @property
    def rel_err (self):
        """
        
        
        Returns
        -------
        errs : np.array
            DESCRIPTION. Square root of diagonal elements of HIRAX cov matrix 
            reshaped into <kpar_size> x <kperp_size>

        """
        kpar_params, kperp_params, kcenter_params = self.k_space_parameters()
        
        errs = np.sqrt(abs(np.diag(self.covhirax))).reshape(kpar_params['kpar_size'],kperp_params['kperp_size'])
        
        errs1 = errs # / self.h**(3/2)
        
        return errs1 
        
    
    # def k_space_parameters_al(self):
        
    #     kpar_params, kperp_params, kcenter_params = self.k_space_parameters()
        
    #     kpar_bands = kpar_params['kpar_bands']
    #     kperp_bands = kperp_params['kperp_bands']
        
    #     kparb, kperpb = np.broadcast_arrays(kpar_bands[np.newaxis, :], kperp_bands[:, np.newaxis])
        
    #     # Pull out the start, end and centre of the bands in k, mu directions
    #     kpar_start = kparb[1:, :-1].flatten()
    #     kpar_end = kparb[1:, 1:].flatten()
    #     kpar_center = 0.5 * (kpar_end + kpar_start)
        
    #     kperp_start = kperpb[:-1, 1:].flatten()
    #     kperp_end = kperpb[1:, 1:].flatten()
    #     kperp_center = 0.5 * (kperp_end + kperp_start)
        
    #     kpar_params_al = {'kpar_start':kpar_start , 'kpar_end':kpar_end , 'kpar_center':kpar_center}
    #     kperp_params_al = {'kperp_start':kperp_start , 'kperp_end':kperp_end , 'kperp_center':kperp_center}
        
    #     return kpar_params_al, kperp_params_al


























