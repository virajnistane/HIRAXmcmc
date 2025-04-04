#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,sys
import numpy as np
import numpy.linalg as la

from scipy.constants import speed_of_light as cc

import hiraxmcmc

from hiraxmcmc.util.basicfunctions import *
from hiraxmcmc.core.hiraxoutput import HiraxOutput
from hiraxmcmc.core.powerspectrum import CreatePs2d
from hiraxmcmc.core.powerspectrum import Ps2dFromPofk
from hiraxmcmc.util.cosmoparamvalues import ParametersFixed




# =============================================================================
# χ^2
# =============================================================================



class Chi2Func:
    
    def __init__(self, inputforhiraxoutput, k_hunits = False, rank_mpi=None, INPUT=None):
        
        self.modulelocation = os.path.dirname(hiraxmcmc.__file__)
        self.inputforhiraxoutput = inputforhiraxoutput
        self.INPUT = INPUT
        
        """ hirax output """
        self.hirax_output        =   HiraxOutput(inputforhiraxoutput, k_hunits = k_hunits)
        # self.kpar_all, self.kperp_all, self.kc_all = self.hirax_output.k_space_parameters()
        
        # self.kpar       =   self.kpar_all['kpar']
        # self.kperp      =   self.kperp_all['kperp']
        # self.kcenter    =   self.kc_all['kcenter']
        
        self.redshift = self.hirax_output.redshift
        
        try:
            if INPUT != None:
                assert INPUT['likelihood']['PS_cov']['override'] == 'no'
            self.covhirax = self.hirax_output.cov
            self.errs = self.hirax_output.relPS_err
            if rank_mpi == 0:
                print('Covariance in likelihood used from m-mode sims')
        except:
            if INPUT != None:
                assert INPUT['likelihood']['PS_cov']['override'] == 'yes'
            cov_override_file = find_files_containing(find_freqchannel_for_redshift(self.redshift) ,
                                                      INPUT['likelihood']['PS_cov']['files_dirfullpath'])[0]
            self.covhirax = np.loadtxt(os.path.join(INPUT['likelihood']['PS_cov']['files_dirfullpath'],
                                                    cov_override_file))
            self.errs  =  np.sqrt(abs(np.diag(self.covhirax))).reshape(self.kpar_all['kpar_size'],self.kperp_all['kperp_size'])
            if rank_mpi == 0:
                print('Covariance in likelihood used from external file: \n %s'%(cov_override_file))
        
        # self.kparstart = self.kpar_all['kpar_bands'][:-1]
        # self.kparend = self.kpar_all['kpar_bands'][1:]
        # self.kperpstart = self.kperp_all['kperp_bands'][:-1]
        # self.kperpend = self.kperp_all['kperp_bands'][1:]
        
        try:
            self.relPS_amp_fromHirax = self.hirax_output.relPS_amp
        except:
            self.relPS_amp_fromHirax = None   
        
        
        # print("Inside the chi2_function instance, hiraxrundirname is ",hirax_output.hiraxrundir_name)
        
        """ select k-bins """
        
        if 400<z2freq(self.redshift)<500:
            kperplimits=(0.03, 0.08)
            kparlimits=(0.05,0.22)
            kcenterlimits=(0.05,0.2)
        elif 500<z2freq(self.redshift)<600:
            kperplimits=(0.03, 0.12)
            kparlimits=(0.05,0.22)
            kcenterlimits=(0.05,0.2)
        elif 600<z2freq(self.redshift)<700:
            kperplimits=(0.04, 0.16)
            kparlimits=(0.05,0.22)
            kcenterlimits=(0.05,0.2)
        elif 700<z2freq(self.redshift)<800:
            kperplimits=(0.05, 0.18)
            kparlimits=(0.05,0.22)
            kcenterlimits=(0.05,0.2)

        
        # self.kperp_limits_hunits =  {'l':self.kperpstart[2], 'u':self.kperpend[6]}  # {'l':0.025, 'u':0.1}
        # self.kpar_limits_hunits =    {'l':self.kparstart[4], 'u':self.kparend[12]} # {'l':0.025, 'u':0.25}
        
        # if self.hirax_output.psetype == 'minvar':
        #     self.kpar_limits_hunits['l'] = self.kparstart[3]
        #     self.kpar_limits_hunits['u'] = self.kparend[16]
        #     # self.kpar_limits_hunits['l'] = self.kparstart[5]
        #     # self.kpar_limits_hunits['u'] = self.kparend[21]
        # elif self.hirax_output.psetype == 'unwindowed':
        #     self.kpar_limits_hunits['l'] = self.kparstart[3]
        #     self.kpar_limits_hunits['u'] = self.kparend[16]
            
        # self.kcenter_limits_hunits = {'l':0.05 * self.hirax_output.h, 'u':0.15 * self.hirax_output.h}
        
        
        # self.xsens =  ((self.kperp > self.kperp_limits_hunits['l']) 
        #                * (self.kperp < self.kperp_limits_hunits['u']) 
        #                * (self.kpar > self.kpar_limits_hunits['l']) 
        #                * (self.kpar < self.kpar_limits_hunits['u']) 
        #                * (abs(self.errs) < 1) )
        #                # * (self.kcenter > self.kcenter_limits_hunits['l']) 
        #                # * (self.kcenter < self.kcenter_limits_hunits['u'])) 
        # #(k_center.flat > 0.1) * (k_center.flat < 0.15) * (np.diag(y['cov']) < 1)
        
        
        # self.xsensflat = np.array(self.xsens.flat)
        # self.ysens = np.outer(self.xsensflat,self.xsensflat)
        
        # cov_masked_sens = self.covhirax[self.ysens]
        # self.newshape_sens = int(cov_masked_sens.shape[0]**0.5)
        # self.cov_masked_sens = cov_masked_sens.reshape(self.newshape_sens,self.newshape_sens);
        
        (self.cov_masked, 
         self.ps_masked_flat, 
         self.Nbins_masked) = self.hirax_output.masked_cov_and_ps(kperp_limits_tuple = kperplimits,
                                                                  kpar_limits_tuple = kparlimits,
                                                                  kcenter_limits_tuple = kcenterlimits)
        
        """ params_fixed """
        self.params_fixed = ParametersFixed()
        # self.currentparamsfixed = self.params_fixed.current_params_fixed
        self.allparams_fixed = self.params_fixed.current_allparams_fixed
        self.cosmoparams_fixed = self.params_fixed.cosmoparams_fixed
        
        try:
            self.h_fiducial = self.cosmoparams_fixed['h']
            H0 = self.h_fiducial * 100
        except:
            H0 = self.cosmoparams_fixed['H0']
            self.h_fiducial = H0/100
        
        """ createPS2D instances """
        
        self.ps2d_from_Pofk = Ps2dFromPofk(inputforhiraxoutput = self.inputforhiraxoutput, k_hunits=k_hunits)
        
        # Instances (sample and params (varying)) of PS 1D generating class "CreatePs2d"
        self.cp_sample = CreatePs2d(inputforhiraxoutput = inputforhiraxoutput, pspackage='class', pstype = 'sample', k_hunits=k_hunits, INPUT=INPUT)
        self.cp_params = CreatePs2d(inputforhiraxoutput = inputforhiraxoutput, pspackage='class', pstype = 'param', k_hunits=k_hunits, INPUT=INPUT)
        
        
        
        # PS 1D and corresponding properties (sample)
        self.pk_z_estimated, self.pspackage_properties = self.cp_sample.get_pk_and_prop(currentparams = self.params_fixed.cosmoparams_fixed)
        
        try:
            assert self.cp_sample.pspackage == 'class'
            self.dA_fid = self.pspackage_properties.angular_distance(self.redshift)
            self.hz_fid = self.pspackage_properties.Hubble(self.redshift)
            
            self.f_growth_for_ps_estimated = self.pspackage_properties.scale_independent_growth_factor_f(self.redshift)
            self.q_perp_for_ps_estimated =  self.pspackage_properties.angular_distance(self.redshift) / self.dA_fid   #* self.cosmoparams_fixed['h']/self.h_fiducial
            self.q_par_for_ps_estimated =  self.hz_fid / self.pspackage_properties.Hubble(self.redshift)              #* self.cosmoparams_fixed['h']/self.h_fiducial
        except:
            assert self.cp_sample.pspackage == 'camb'
            self.dA_fid = self.pspackage_properties.angular_diameter_distance(self.redshift)
            self.hz_fid = self.pspackage_properties.h_of_z(self.redshift)
            
            self.f_growth_for_ps_estimated = self.pspackage_properties.get_redshift_evolution(q=0.215, z=self.redshift, vars=['growth'])[0,0]
            self.q_perp_for_ps_estimated =  self.pspackage_properties.angular_diameter_distance(self.redshift) / self.dA_fid   * self.cosmoparams_fixed['h']/self.h_fiducial
            self.q_par_for_ps_estimated =  self.hz_fid / self.pspackage_properties.h_of_z(self.redshift)              * self.cosmoparams_fixed['h']/self.h_fiducial
        
        self.powerspectra_rescaling_factor_ps_estimated = 1/ (self.q_perp_for_ps_estimated**2 * self.q_par_for_ps_estimated)
        
        # PS 1D --> PS 2D
        self.ps_estimated = self.cp_sample.get_ps2d_from_pok(PK_k_zClass = self.pk_z_estimated,
                                                             q_perp_input = self.q_perp_for_ps_estimated, 
                                                             q_par_input = self.q_par_for_ps_estimated,
                                                             z=self.redshift,
                                                             f_growth = self.f_growth_for_ps_estimated,
                                                             D_growth_z = self.pspackage_properties.scale_independent_growth_factor(self.redshift),
                                                             powerspectra_rescaling_factor = self.powerspectra_rescaling_factor_ps_estimated)
                                                             # currentparams_input = self.params_fixed.cosmoparams_fixed,
                                                             # )
        
        
    
    def chi2_multiz(self, PK_k_z_currentstep, PK_properties_currentstep, z, currentparams, cosmoparams, scalingparams=None):  
        
        freqdep_paramstovary = checkconditionforlist(list(currentparams.keys()), allelements_have_subpart='(z)')
        
        try:
            assert not(freqdep_paramstovary)
            
            currentparamstemp = currentparams.copy()
            for i in cosmoparams.keys():
                try:
                    assert i in currentparamstemp.keys()
                except:
                    currentparamstemp[i] = cosmoparams[i]
            
            try:
                h = currentparamstemp['h']
                H0 = h * 100
            except:
                H0 = currentparamstemp['H0']
                h = H0/100
            
            try:
                assert self.cp_params.pspackage == 'class'
                q_perp = PK_properties_currentstep.angular_distance(z) / self.dA_fid # * h/self.h_fiducial
                q_par = self.hz_fid / PK_properties_currentstep.Hubble(z)            # * h/self.h_fiducial
                
                powerspectra_rescaling_factor = 1/(q_perp**2 * q_par) # * (h/self.h_fiducial)**3
                # this second ratio is to remove the h-units of the k-values (so, it is only needed when the k values are in h/Mpc units)
                # for example: kpar_obs[h/Mpc] = kpar_fid[h/Mpc]/ q_par * (h/h_fid) = kpar_fid[h/Mpc]/ (q_par * (h_fid/h))
                    # Then this h/h_fid ratio, when including in the q_par, becomes (h_fid/h)
                f_growth = PK_properties_currentstep.scale_independent_growth_factor_f(z)
                
            except:
                assert self.cp_params.pspackage == 'camb'
                raise ValueError("Hellooo! It seems you are using CAMB for generating \
                                 signal for varying parameters. It is recommended to \
                                     use CLASS code instead. If you insist on using \
                                         CAMB, please comment out this line from the code.")
                # q_perp = PK_properties_currentstep.angular_diameter_distance(z) / self.dA_fid   * self.h_fiducial/h
                # q_par = self.hz_fid / PK_properties_currentstep.h_of_z(z)                       * self.h_fiducial/h
                # # this second ratio is to remove the h-units of the k-values (so, it is only needed when the k values are in h/Mpc units)
                # # for example: kpar_obs[h/Mpc] = kpar_fid[h/Mpc]/ q_par * (h/h_fid) = kpar_fid[h/Mpc]/ (q_par * (h_fid/h))
                #     # Then this h/h_fid ratio, when including in the q_par, becomes (h_fid/h)
                # f_growth = PK_properties_currentstep.get_redshift_evolution(q=0.215, z=z, vars=['growth'])[0,0]
                
            currentparams_input_for_pscalc = currentparamstemp
        except:
            assert freqdep_paramstovary
            
            try:
                h = cosmoparams['h']
                H0 = h * 100
            except:
                H0 = cosmoparams['H0']
                h = H0/100
            
            currentparamstemp = currentparams.copy()
            
            if scalingparams != None:    
                for i in scalingparams.keys():
                    try:
                        assert i in currentparamstemp.keys()
                    except:
                        currentparamstemp[i] = scalingparams[i]
                    
            try:
                q_par = currentparamstemp['qpar(z)']                #* self.h_fiducial/h
            except:
                q_par = self.hz_fid / currentparamstemp['h(z)']     #* self.h_fiducial/h
                
            try:
                q_perp = currentparamstemp['qperp(z)']              #* self.h_fiducial/h
            except:
                q_perp = currentparamstemp['dA(z)'] / self.dA_fid   #* self.h_fiducial/h
                
            powerspectra_rescaling_factor = 1/(q_perp**2 * q_par)
            
            f_growth = currentparamstemp['f(z)']
            currentparams_input_for_pscalc = cosmoparams
        
        
        self.q_perp = q_perp
        self.q_par = q_par
        self.fz = f_growth
        
        # try:
        #     assert not(freqdep_paramstovary)
        #     pkz_input_temp = self.pk_z_estimated
        # except:
        #     assert freqdep_paramstovary
        #     pkz_input_temp = PK_k_z_currentstep
        
        D_growth_here = self.pspackage_properties.scale_independent_growth_factor(z)
        
        pscalc = self.cp_params.get_ps2d_from_pok(PK_k_zClass = self.pk_z_estimated,
                                                  q_perp_input = q_perp,
                                                  q_par_input = q_par,
                                                  z=z,
                                                  f_growth = f_growth,
                                                  D_growth_z = D_growth_here,
                                                  powerspectra_rescaling_factor = powerspectra_rescaling_factor
                                                  )
                                                  # currentparams_input = currentparams_input_for_pscalc,
        
        ps = (pscalc/self.ps_estimated - 1)
        
        ps_masked = ps.flat[:][self.hirax_output.xmasked.flatten()]
        
        self.ps_masked = ps_masked
        
        # chi2 = np.matmul(np.matmul(ps_masked_sens_chi2, la.inv(self.cov_masked_sens)), ps_masked_sens_chi2)
        chi2 = (self.ps_masked).dot( (la.inv(self.cov_masked)).dot( self.ps_masked))
        
        
        return chi2
    
        
    @property
    def kpar_limits(self):
        return self.kpar_minlimit, self.kpar_maxlimit
    @kpar_limits.setter
    def kpar_limits(self, newlimitstuple):
        self.kpar_minlimit, self.kpar_maxlimit = newlimitstuple
    
    @property
    def kperp_limits(self):
        return self.kperp_minlimit, self.kperp_maxlimit
    @kperp_limits.setter
    def kperp_limits(self, newlimitstuple):
        self.kperp_minlimit, self.kperp_maxlimit = newlimitstuple
        
    @property
    def kcenter_limits(self):
        return self.kcenter_minlimit, self.kcenter_maxlimit
    @kcenter_limits.setter
    def kcenter_limits(self, newlimitstuple):
        self.kcenter_minlimit, self.kcenter_maxlimit = newlimitstuple
    
    # def chi2(self, current_params, z, get_pscalc_out = False):
        
    #     # t0 = time.time()
        
    #     pscalc = self.cp_class_param.create_ps2d_from_class(currentparams = current_params, z=z)
        
    #     # pscalc = self.cp_camb_param.create_ps2d_from_camb(currentparams = current_params, z=z)
        
        
    #     ps = (pscalc/self.ps_estimated - 1).T
        
    #     ps_masked_sens_chi2 = ps.flat[:][self.x_sens] 
        
    #     chi2 = np.matmul(np.matmul(ps_masked_sens_chi2,la.inv(self.cov_masked_sens)), ps_masked_sens_chi2)
        
    #     # print(time.time() - t0)
        
    #     if get_pscalc_out:
    #         return chi2, pscalc
    #     else:
    #         return chi2, None
    
    def chi2_from_hirax(self, current_params, z, get_pscalc_out = False):
        
        # t0 = time.time()
        
        pscalc = self.cp_class_param.create_ps2d_from_class(current_params, z = self.redshift)
        
        ps = (pscalc/self.relPS_amp_fromHirax - 1)
        
        ps_masked = (ps.flat[:][self.hirax_output.xmasked.flatten()])
        
        chi2 = np.matmul(np.matmul(ps_masked,la.inv(self.cov_masked)), ps_masked)
        
        # print(time.time() - t0)
        
        if get_pscalc_out:
            return chi2, pscalc
        else:
            return chi2, None
        
    # def pofk_interpolator_for_pscalc(self, current_params):
    #     return self.cp_class_param.pofk_interpolator_from_class(current_params)
        
    
        
        
        
        
        
        

