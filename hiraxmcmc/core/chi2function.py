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
    
    def __init__(self, inputforhiraxoutput, INPUT=None):
        
        self.modulelocation = os.path.dirname(hiraxmcmc.__file__)
        self.inputforhiraxoutput = inputforhiraxoutput
        self.INPUT = INPUT
        
        """ hirax output """
        self.hirax_output        =   HiraxOutput(inputforhiraxoutput)
        self.kpar_all, self.kperp_all, self.kc_all = self.hirax_output.k_space_parameters()
        
        self.kpar       =   self.kpar_all['kpar']
        self.kperp      =   self.kperp_all['kperp']
        self.kcenter    =   self.kc_all['kcenter']
        
        
        try:
            if INPUT != None:
                assert INPUT['likelihood']['ps_rel_err']['override'] == 'no'
            self.covhirax = self.hirax_output.covhirax
            self.errs = self.hirax_output.rel_err
        except:
            if INPUT != None:
                assert INPUT['likelihood']['ps_rel_err']['override'] == 'yes'
            self.covhirax = np.loadtxt(os.path.join(self.modulelocation, 'inputfiles', INPUT['likelihood']['ps_rel_err']['filename']))
            self.errs  =  np.sqrt(abs(np.diag(self.covhirax))).reshape(self.kpar_all['kpar_size'],self.kperp_all['kperp_size'])
        
        self.kparstart = self.kpar_all['kpar_bands'][:-1]
        self.kparend = self.kpar_all['kpar_bands'][1:]
        self.kperpstart = self.kperp_all['kperp_bands'][:-1]
        self.kperpend = self.kperp_all['kperp_bands'][1:]
        
        self.psrel_estimatedfromhirax = self.hirax_output.ps_relative_estimated_from_hirax
        
        self.redshift = self.hirax_output.redshift
        
        # print("Inside the chi2_function instance, hiraxrundirname is ",hirax_output.hiraxrundir_name)
        
        """ select k-bins """
        
        
        
        self.kperp_limits_hunits =  {'l':self.kperpstart[2], 'u':self.kperpend[6]}  # {'l':0.025, 'u':0.1}
        self.kpar_limits_hunits =    {'l':self.kparstart[4], 'u':self.kparend[12]} # {'l':0.025, 'u':0.25}
        self.kcenter_limits_hunits = {'l':0.05, 'u':0.15}
        
        
        
        self.xsens =  (self.kperp > self.kperp_limits_hunits['l']) * (self.kperp < self.kperp_limits_hunits['u']) *  (
            self.kpar > self.kpar_limits_hunits['l']) * (self.kpar < self.kpar_limits_hunits['u']) * (
                abs(self.errs) < 1) *  (
                    self.kcenter > self.kcenter_limits_hunits['l']) * (self.kcenter < self.kcenter_limits_hunits['u']) #(k_center.flat > 0.1) * (k_center.flat < 0.15) * (np.diag(y['cov']) < 1)
        
        
        self.xsensflat = np.array(self.xsens.flat)
        self.ysens = np.outer(self.xsensflat,self.xsensflat)
        
        cov_masked_sens = self.covhirax[self.ysens]
        self.newshape_sens = int(cov_masked_sens.shape[0]**0.5)
        self.cov_masked_sens = cov_masked_sens.reshape(self.newshape_sens,self.newshape_sens);
        
        
        """ params_fixed """
        self.params_fixed = ParametersFixed()
        # self.currentparamsfixed = self.params_fixed.current_params_fixed
        self.allparams_fixed = self.params_fixed.current_allparams_fixed
        self.cosmoparams_fixed = self.params_fixed.cosmoparams_fixed
        
        
        """ createPS2D instances """
        
        self.ps2d_from_Pofk = Ps2dFromPofk(inputforhiraxoutput = self.inputforhiraxoutput)
        
        
        self.cp_sample = CreatePs2d(inputforhiraxoutput = inputforhiraxoutput, pspackage='class', pstype = 'sample')
        self.cp_params = CreatePs2d(inputforhiraxoutput = inputforhiraxoutput, pspackage='class', pstype = 'param')
        
        self.pk_z_estimated_class, self.pspackage_properties_class = self.cp_params.pofk_from_class(currentparams = self.params_fixed.cosmoparams_fixed)
        
        if self.cp_sample.pspackage == 'camb':
            self.pk_estimated_camb, self.pspackage_properties_camb = self.cp_sample.pofk_from_camb(currentparams = self.params_fixed.cosmoparams_fixed)
            self.dA_fid = self.pspackage_properties_camb.angular_diameter_distance(self.redshift)
            self.hz_fid = self.pspackage_properties_camb.h_of_z(self.redshift)
            self.pk_estimated = self.pk_estimated_camb
        elif self.cp_sample.pspackage == 'class':
            self.dA_fid = self.pspackage_properties_class.angular_distance(self.redshift)
            self.hz_fid = self.pspackage_properties_class.Hubble(self.redshift)
            self.pk_estimated = self.pk_z_estimated_class
        
        self.h_fiducial = self.params_fixed.cosmoparams_fixed['H0']/100
        self.f_growth_for_ps_estimated = self.pspackage_properties_class.scale_independent_growth_factor_f(self.redshift)
        
        self.q_perp_for_ps_estimated =  self.pspackage_properties_class.angular_distance(self.redshift) / self.dA_fid * (self.params_fixed.cosmoparams_fixed['H0']/100)/self.h_fiducial
        self.q_par_for_ps_estimated =  self.hz_fid / self.pspackage_properties_class.Hubble(self.redshift) * (self.params_fixed.cosmoparams_fixed['H0']/100)/self.h_fiducial
        
        
        self.ps_estimated = self.cp_sample.get_ps2d_from_pok(PK_k_zClass = self.pk_estimated,
                                                             q_perp_input = self.q_perp_for_ps_estimated, 
                                                             q_par_input = self.q_par_for_ps_estimated,
                                                             z=self.redshift,
                                                             currentparams_input = self.params_fixed.cosmoparams_fixed,
                                                             f_growth = self.f_growth_for_ps_estimated)
                                                             # currentparams = self.params_fixed.current_params_fixed
        
        
    
    def chi2_multiz(self, PK_k_z_currentstep, current_class_instance, z, currentparams, cosmoparams):  # currentparams,
        
        freqdep_paramstovary = checkconditionforlist(list(currentparams.keys()), allelements_have_subpart='(z)')
        
        try:
            assert not(freqdep_paramstovary)
            
            currentparamstemp = currentparams.copy()
            for i in cosmoparams.keys():
                try:
                    assert i in currentparamstemp.keys()
                except:
                    currentparamstemp[i] = cosmoparams[i]
            
            
            # temporarily commenting out the h/h_fid ratio rescaling of q_par and q_perp
            # only on the branch "q_no_rescale_with_h_ratio"
            
            q_perp = current_class_instance.angular_distance(z) / self.dA_fid   #* (currentparamstemp['H0']/100)/self.h_fiducial
            q_par = self.hz_fid / current_class_instance.Hubble(z)              #* (currentparamstemp['H0']/100)/self.h_fiducial
            f_growth = current_class_instance.scale_independent_growth_factor_f(z)
            currentparams_input_for_pscalc = currentparamstemp
        except:
            assert freqdep_paramstovary
            q_perp = currentparams['dA(z)'] / self.dA_fid     * (cosmoparams['H0']/100)/self.h_fiducial
            q_par = self.hz_fid / currentparams['h(z)']       * (cosmoparams['H0']/100)/self.h_fiducial
            f_growth = currentparams['f(z)']
            currentparams_input_for_pscalc = cosmoparams
        
        
        self.q_perp = q_perp
        self.q_par = q_par
        self.fz = f_growth
        
        pscalc = self.cp_params.get_ps2d_from_pok(PK_k_zClass = PK_k_z_currentstep,
                                                  q_perp_input = q_perp,
                                                  q_par_input = q_par,
                                                  z=z,
                                                  currentparams_input = currentparams_input_for_pscalc,
                                                  f_growth = f_growth)
        
        ps = (pscalc/self.ps_estimated - 1)
        
        ps_masked_sens_chi2 = ps.flat[:][self.xsensflat]
        
        self.ps_masked_sens_chi2 = ps_masked_sens_chi2
        
        chi2 = np.matmul(np.matmul(ps_masked_sens_chi2, la.inv(self.cov_masked_sens)), ps_masked_sens_chi2)
        
        return chi2
    
    
    # def chi2_multiz(self, PK_k_z_currentstep, current_class_instance, z, currentparams, cosmoparams=None):  # currentparams,
        
    #     q_perp = current_class_instance.angular_distance(z) / self.dA_fid   * (currentparams['H0']/100)/self.h_fiducial
    #     q_par = self.hz_fid / current_class_instance.Hubble(z)              * (currentparams['H0']/100)/self.h_fiducial
    #     f_growth = current_class_instance.scale_independent_growth_factor_f(z)
        
        
    #     self.q_perp = q_perp
    #     self.q_par = q_par
        
        
        
    #     self.fz = f_growth
        
    #     pscalc = self.cp_params.get_ps2d_from_pok(PK_k_zClass = PK_k_z_currentstep,
    #                                               q_perp_input = q_perp,
    #                                               q_par_input = q_par,
    #                                               z=z,
    #                                               currentparams_input = currentparams,
    #                                               f_growth = f_growth)
    #                                               # currentparams = currentparams,
        
    #     ps = (pscalc/self.ps_estimated - 1)
        
        
    #     ps_masked_sens_chi2 = ps.flat[:][self.x_sens]
        
    #     chi2 = np.matmul(np.matmul(ps_masked_sens_chi2, la.inv(self.cov_masked_sens)), ps_masked_sens_chi2)
        
    #     return chi2
        
    
    
    
    # def chi2_multiz_p2vfreqdep(self, PK_k_z_currentstep, current_class_instance, z, currentparams, cosmoparams):  # currentparams,
        
    #     q_perp = currentparams['dA(z)'][self.key_inputforhiraxoutput] / self.dA_fid       * (cosmoparams['H0']/100)/self.h_fiducial
    #     q_par = self.hz_fid / currentparams['h(z)'][self.key_inputforhiraxoutput]         * (cosmoparams['H0']/100)/self.h_fiducial
        
        
    #     self.q_perp = q_perp
    #     self.q_par = q_par
        
    #     f_growth = currentparams['f(z)'][self.key_inputforhiraxoutput]
        
    #     self.fz = f_growth
        
    #     pscalc = self.cp_params.get_ps2d_from_pok(PK_k_zClass = PK_k_z_currentstep,
    #                                               q_perp_input = q_perp,
    #                                               q_par_input = q_par,
    #                                               z=z,
    #                                               currentparams_input = cosmoparams,
    #                                               f_growth = f_growth)
        
        
    #     ps = (pscalc/self.ps_estimated - 1)
        
        
    #     ps_masked_sens_chi2 = ps.flat[:][self.x_sens]
        
    #     chi2 = np.matmul(np.matmul(ps_masked_sens_chi2, la.inv(self.cov_masked_sens)), ps_masked_sens_chi2)
        
    #     return chi2
        
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
    
    # @property
    # def ps_estimated(self):
    #     return self.cp_camb_sample.get_ps2d_from_pok_camb(function_ps2d_bandfunc = self.get_ps2d_bandfunc,
    #                                                       CLASS_instance = self.cp_class_param, 
    #                                                       CAMB_sample_instance = self.cp_camb_sample,
    #                                                       pk_interpolator_from_camb = self.cp_camb_sample.pofk_interpolator_from_camb())
        
    
    
    # @property
    # def ps_estimated(self):
    #     return self.cp_camb_sample.create_ps2d_from_camb(z=self.redshift)
        
    
    # @property
    # def ps_estimated_camb_sample(self):
    #     return self.ps_estimated(z=self.redshift) 
    
    # @property
    # def cp_class_param(self):
    #     return CreatePs2d(inputforhiraxoutput = self.inputforhiraxoutput, pspackage='class', pstype= 'param')
    
    
    # @property
    # def cp_camb_param(self):
    #     return CreatePs2d(inputforhiraxoutput = self.inputforhiraxoutput, pspackage='camb', pstype= 'param')
    
    # def ps_calc(self, current_params, z):
    #     pscalc = self.cp_class_param.create_ps2d_from_class(currentparams = current_params, z=self.redshift)
    #     return pscalc
    
    
    def chi2(self, current_params, z, get_pscalc_out = False):
        
        # t0 = time.time()
        
        pscalc = self.cp_class_param.create_ps2d_from_class(currentparams = current_params, z=z)
        
        # pscalc = self.cp_camb_param.create_ps2d_from_camb(currentparams = current_params, z=z)
        
        
        ps = (pscalc/self.ps_estimated - 1).T
        
        ps_masked_sens_chi2 = ps.flat[:][self.x_sens] 
        
        chi2 = np.matmul(np.matmul(ps_masked_sens_chi2,la.inv(self.cov_masked_sens)), ps_masked_sens_chi2)
        
        # print(time.time() - t0)
        
        if get_pscalc_out:
            return chi2, pscalc
        else:
            return chi2, None
    
    def chi2_from_hirax(self, current_params, z, get_pscalc_out = False):
        
        # t0 = time.time()
        
        pscalc = self.cp_class_param.create_ps2d_from_class(current_params, z = self.redshift)
        
        ps = (pscalc/self.psrel_estimatedfromhirax - 1)
        
        ps_masked_sens_chi2 = (ps.flat[:][self.x_sens]).T
        
        chi2 = np.matmul(np.matmul(ps_masked_sens_chi2,la.inv(self.cov_masked_sens)), ps_masked_sens_chi2)
        
        # print(time.time() - t0)
        
        if get_pscalc_out:
            return chi2, pscalc
        else:
            return chi2, None
        
    # def pofk_interpolator_for_pscalc(self, current_params):
    #     return self.cp_class_param.pofk_interpolator_from_class(current_params)
        
    
        
        
        
        
        
        

