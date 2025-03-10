#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import numpy.linalg as la
import h5py as hh
import zarr
import yaml
from yaml import Loader, Dumper
import pdb

import hiraxmcmc
from hiraxmcmc.util.basicfunctions import *
from hiraxmcmc.util.cosmoparamvalues import ParametersFixed


class HiraxOutput:
    
    def __init__(self, inputforhiraxoutput, k_hunits=False):
        

        
        self.modulelocation = os.path.dirname(hiraxmcmc.__file__)
        self.hiraxrundir_name = inputforhiraxoutput['result_dir_name']
        try:
            assert os.path.dirname(self.hiraxrundir_name) == ''
            self.hiraxrundir_fullpath = os.path.join(self.modulelocation, 'mmoderesults', self.hiraxrundir_name)
        except:
            self.hiraxrundir_fullpath = self.hiraxrundir_name
        # self.redshift = inputforhiraxoutput['redshift']
        
        self.psetype = inputforhiraxoutput['estimator_type']
        self.klmode = inputforhiraxoutput['klmode']
        self.uname = os.uname()[1]
        self.parameters_fixed = ParametersFixed()
        
        self.driftproddir_fullpath = find_subdirs_begin_with('drift', 
                                                             self.hiraxrundir_fullpath, 
                                                             fullpathoutput=True)[0]
        # driftproddir = 'drift_prod_hirax_survey_49elem_7point_64bands'
        
        config_filename = os.path.join(self.driftproddir_fullpath, 'config.yaml')
        
        with open(config_filename) as configfile:
            self.configs_runs = yaml.load(configfile, Loader=Loader)
        
        try:
            try:
                freqlower = self.configs_runs['telescope']['freq_lower']
                frequpper = self.configs_runs['telescope']['freq_upper']
            except:
                freqlower = self.configs_runs['telescope']['freq_start']
                frequpper = self.configs_runs['telescope']['freq_end']
        except:
            try:
                freqlower = self.configs_runs['telescope']['hirax_spec']['freq_lower']
                frequpper = self.configs_runs['telescope']['hirax_spec']['freq_upper']
            except:
                freqlower = self.configs_runs['telescope']['hirax_spec']['freq_start']
                frequpper = self.configs_runs['telescope']['hirax_spec']['freq_end']
        self.redshift = freq2z(np.mean([freqlower, frequpper]))
        
        self.fisherfileload = hh.File(os.path.join(self.driftproddir_fullpath, 
                                                   'bt/%s/'%(self.klmode),
                                                   find_subdirs_containing(
                                                       self.klmode,
                                                       os.path.join(self.driftproddir_fullpath, 
                                                                    'bt/%s/'%(self.klmode)))[0],
                                                   'fisher.hdf5'),
                                      'r')
        
        
        
        if k_hunits == False:
            self.h = self.parameters_fixed.h_fid
        elif k_hunits == True:
            self.h = 1
        
        
        if self.psetype == 'minvar':
            self.psetypeshort = 'mv'
        elif self.psetype == 'unwindowed':
            self.psetypeshort = 'uw'
        
        self.psfileload = None
        self.psmcfile_fullpath = None
        # try:
        #     psfile = [i for i in find_files_containing(self.klmode, 
        #                                                os.path.join(self.hiraxrundir_fullpath, 
        #                                                             'draco'), fullpathoutput=True)
        #               if ((self.psetypeshort in os.path.basename(i))
        #                   or 
        #                   (self.psetype in  os.path.basename(i)))][0]
        #     self.psfileload = hh.File(psfile,'r')
        # except:
        #     psfile = [i for i in find_subdirs_containing(self.klmode, 
        #                                                  os.path.join(self.hiraxrundir_fullpath, 
        #                                                               'draco'), fullpathoutput=True)
        #               if ((self.psetypeshort in os.path.basename(i))
        #                   or 
        #                   (self.psetype in  os.path.basename(i)))][0]
        #     self.psfileload = zarr.load(psfile)
        
        # driftproddir = find_subdirs_begin_with('drift', self.hiraxrundir_fullpath)[0]
        # driftproddir = 'drift_prod_hirax_survey_49elem_7point_64bands'
        # self.fisherfileload = hh.File(os.path.join(self.hiraxrundir_fullpath, 
        #                                            driftproddir,
        #                                            'bt/%s/psmc_%s_1threshold/fisher.hdf5'%(self.klmode,
        #                                                                                    self.klmode)),'r')
        
        
        # None-Initiated attributes
        self.cinv = None
        self.extentlower = None
        
        
    
    def __repr__(self):
        return f"HiraxOutput({self.hiraxrundir_name!r}, {self.psetype!r}, {self.redshift!r}, {self.klmode!r})"
    
    
    
    ##### 
    # If h-units are to be removed, confirm the h-dependence of the covariance matrix from m-mode
    #####
    @property
    def cov(self):
        try:
            self.cinv = self.psfileload['C_inv'][:]
            if self.psestimator == "unwindowed":
                cov_val = la.pinv(self.cinv.reshape(int(self.cinv.shape[0]* self.cinv.shape[1]),int(self.cinv.shape[2]* self.cinv.shape[3])), atol=1e-8).T
            elif self.psestimator == "min var":
                M = np.diag(self.cinv.T.reshape(int(self.cinv.shape[0]*self.cinv.shape[1]),int(self.cinv.shape[2]*self.cinv.shape[3])).sum(axis=1)** -1)
                cov_val = np.matmul(M, np.matmul(self.cinv.T.reshape(int(self.cinv.shape[0]*self.cinv.shape[1]),int(self.cinv.shape[2]*self.cinv.shape[3])), M.T))
            return cov_val
        except:
            return self.fisherfileload['covariance'][:]
    
    # @property
    # def covhirax(self):
    #     return np.loadtxt(os.path.relpath(os.path.join(self.MCMCmodulespath,'viraj_cov_matrix.dat')))
    
    
    @property
    def relPS_amp(self):
        if self.psfileload != None:
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
    def relPS_err(self):
        """
        Returns
        -------
        errs : np.array
            DESCRIPTION. Square root of diagonal elements of HIRAX cov matrix 
            reshaped into <kpar_size> x <kperp_size>
        """
        kpar_params, kperp_params, kcenter_params = self.k_space_parameters()

        # errs = np.sqrt(abs(np.diag(self.cov))).reshape(kpar_params['kpar_size'],kperp_params['kperp_size'])

        if self.psmcfile_fullpath != None:
            errs = np.sqrt(abs(np.diag(self.cov))).reshape(kperp_params['kperp_size'],kpar_params['kpar_size']).T
        else:
            errs = np.sqrt(abs(np.diag(self.cov))).reshape(kperp_params['kperp_size'],kpar_params['kpar_size']).T

        return errs
    
        
    def masked_cov_and_ps(self, kperp_limits_tuple: tuple = None, kpar_limits_tuple: tuple = None, kcenter_limits_tuple: tuple = None):
        kpar_all, kperp_all, kc_all = self.k_space_parameters()
        
        kpar       =   kpar_all['kpar']
        kperp      =   kperp_all['kperp']
        kcenter    =   kc_all['kcenter']
        
        kparstart = kpar_all['kpar_bands'][:-1]
        kparend = kpar_all['kpar_bands'][1:]
        kperpstart = kperp_all['kperp_bands'][:-1]
        kperpend = kperp_all['kperp_bands'][1:]        
        
        if kperp_limits_tuple == None:
            kperp_limits = {'l':kperpstart[2], 'u':kperpend[6]}  # {'l':0.025, 'u':0.1}
        else:
            kperp_lower = kperpstart[np.argmin(abs(kperp_limits_tuple[0] - kperpstart))]
            kperp_upper = kperpend[np.argmin(abs(kperp_limits_tuple[1] - kperpend))]
            kperp_limits = {'l':kperp_lower, 'u':kperp_upper}
        
        self.kperp_limits = kperp_limits
        
        if kpar_limits_tuple == None:
            kpar_limits  = {'l':kparstart[4] , 'u':kparend[12]} # {'l':0.025, 'u':0.25}
        else:
            kpar_lower = kparstart[np.argmin(abs(kpar_limits_tuple[0] - kparstart))]
            kpar_upper = kparend[np.argmin(abs(kpar_limits_tuple[1] - kparend))]
            kpar_limits = {'l':kpar_lower, 'u':kpar_upper}
            
        self.kpar_limits = kpar_limits
        
        if kcenter_limits_tuple == None:
            kcenter_limits  = {'l':0.0 , 'u':np.sqrt(kperpend[-1]**2 + kparend[-1]**2)} # {'l':0.025, 'u':0.25}
        else:
            kcenter_lower = kcenter.flatten()[np.argmin(abs(kcenter_limits_tuple[0] - 
                                                            np.sqrt(kpar_all['kpar_start_flat']**2 
                                                                    + kperp_all['kperp_start_flat']**2
                                                                    ).reshape(kcenter.shape)
                                                            ))]
            kcenter_upper = kcenter.flatten()[np.argmin(abs(kcenter_limits_tuple[1] - 
                                                            np.sqrt(kpar_all['kpar_end_flat']**2 
                                                                    + kperp_all['kperp_end_flat']**2
                                                                    ).reshape(kcenter.shape)
                                                            ))]
            kcenter_limits = {'l':kcenter_lower, 'u':kcenter_upper}
            
            # kcenter_limits = {'l':kcenter_limits_tuple[0], 'u':kcenter_limits_tuple[1]}
            
        
        xmasked =  ((kperp > kperp_limits['l']) 
                    * (kperp < kperp_limits['u']) 
                    * (kpar > kpar_limits['l']) 
                    * (kpar < kpar_limits['u'])
                    * (abs(self.relPS_err) < 1)
                    * (kcenter > kcenter_limits['l'])
                    * (kcenter < kcenter_limits['u'])
                    )
    
        xmaskedflat = np.array(xmasked.flatten())
        ymasked = np.outer(xmaskedflat,xmaskedflat)
        
        cov_masked = self.cov[ymasked]
        newshape_masked = int(cov_masked.shape[0]**0.5)
        cov_masked = cov_masked.reshape(newshape_masked,newshape_masked)
        
        try:
            ps_masked_flat = self.relPS_amp.flatten()[xmaskedflat]
            
            return cov_masked, ps_masked_flat, newshape_masked
        except:
            return cov_masked, None, newshape_masked
    
    def chi2_relPSamp(self, kperp_limits_tuple: tuple = None, kpar_limits_tuple: tuple = None, kcenter_limits_tuple: tuple = None):
        cov_mask, ps_mask_flat, newshape_masked = self.masked_cov_and_ps(kperp_limits_tuple, kpar_limits_tuple, kcenter_limits_tuple)
        return (ps_mask_flat.dot(la.inv(cov_mask))).dot(ps_mask_flat)
    
    def chi2_relPSamp_perDOF(self, kperp_limits_tuple: tuple = None, kpar_limits_tuple: tuple = None, kcenter_limits_tuple: tuple = None):
        cov_mask, ps_mask_flat, newshape_masked = self.masked_cov_and_ps(kperp_limits_tuple, kpar_limits_tuple, kcenter_limits_tuple)
        return (ps_mask_flat.dot(la.inv(cov_mask))).dot(ps_mask_flat)/newshape_masked
    
    def sensitivity_relPScov(self, kperp_limits_tuple: tuple = None, kpar_limits_tuple: tuple = None, kcenter_limits_tuple: tuple = None):
        cov_mask, ps_mask_flat, newshape_masked = self.masked_cov_and_ps(kperp_limits_tuple, kpar_limits_tuple, kcenter_limits_tuple)
        return np.exp(1/(2* newshape_masked) * la.slogdet(cov_mask)[1])
    
        
    def off_dia_sum(self, M):
        return (np.sum(abs(M),axis=0)-1)/M.shape[0]
    
    @property
    def corr_from_cov_diag(self):
        
        kpar_params, kperp_params, kcenter_params = self.k_space_parameters()
        cov = self.cov
        corr = np.zeros_like(cov)
        for i, iv in enumerate(cov):
            for j, jv in enumerate(cov[i]):
                corr[i,j] = cov[i,j]/np.sqrt(cov[i,i] * cov[j,j])
                
        corr_rs = np.log10(self.off_dia_sum(corr).reshape(kpar_params['kpar_size'],kperp_params['kperp_size']))
        return corr_rs
















