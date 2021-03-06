#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,sys
import numpy as np
## import time


from configobj import ConfigObj


import hiraxmcmc
from hiraxmcmc.util.basicfunctions import *
from hiraxmcmc.core.hiraxoutput import HiraxOutput
from hiraxmcmc.util.cosmoparamvalues import ParametersFixed

# CLASS/CAMB import

# from camb.sources import GaussianSourceWindow, SplinedSourceWindow
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model #, initialpower

from classy import Class




# =============================================================================
# PS 1d --> PS 2d 
# =============================================================================


class Ps2dFromPofk:
    
    def __init__(self, inputforhiraxoutput):
        
        self.inputforhiraxoutput = inputforhiraxoutput
        
        
        self.parameters_fixed = ParametersFixed()
        # |__
        #    |
        #   \|/
        #    V
        # self.h_fix = self.parameters_fixed.h(self.parameters_fixed.H0_fix)
        self.h_fix = self.parameters_fixed._h_fix
        
        self.hirax_output = HiraxOutput(inputforhiraxoutput)    # inputforhiraxoutput = hiraxrundirname, psetype
        # |__
        #    |
        #   \|/
        #    V
        self.kpa , self.kpe , self.kc = self.hirax_output.k_space_parameters()
        self.kpar_center_1d = self.kpa['kpar_center_1d']
        self.kperp_center_1d = self.kpe['kperp_center_1d']
        # self.k_center = (self.kpa['kpar_center_flat'] ** 2 + self.kpe['kperp_center_flat'] ** 2) ** 0.5
        
        # self.kpa_al , self.kpe_al = self.hirax_output.k_space_parameters_al()
        # self.k_center1 = (self.kpa_al['kpar_center'] ** 2 + self.kpe_al['kperp_center'] ** 2) ** 0.5
        
        self.redshift_from_hiraxoutput = self.hirax_output.redshift
        
        
        def k_obs(k_fid, mu_fid, qpar, qperp):
            return k_fid * (mu_fid**2/qpar**2 + (1-mu_fid**2)/qperp**2)**(0.5)
        self.k_obs = k_obs
        
        def mu_obs(mu_fid, qpar, qperp):
            return mu_fid/qpar * (mu_fid**2/qpar**2 + (1-mu_fid**2)/qperp**2)**(-0.5)
        self.mu_obs = mu_obs
        
        
        """
        For band func method for 2D PS 
        """
        
        # self.bounds = list(zip(self.kpa_al['kpar_start'], self.kpa_al['kpar_end'], self.kpe_al['kperp_start'], self.kpe_al['kperp_end']))
        def bandfunc_2d_cart(kpar_s, kpar_e, kperp_s, kperp_e):
            def band(k, mu):
                kpar = k * mu
                kperp = k * (1.0 - mu ** 2) ** 0.5
                parb = (kpar >= kpar_s) * (kpar <= kpar_e)
                perpb = (kperp >= kperp_s) * (kperp < kperp_e)
                return (parb * perpb).astype(np.float64)
            return band
        self.bandfunc_2d_cart = bandfunc_2d_cart
        
        # bounds = list(zip(self.kpa['kpar_start_flat'].reshape(self.kpa['kpar_size'],self.kpe['kperp_size']).T.flatten(),
        #                   self.kpa['kpar_end_flat'].reshape(self.kpa['kpar_size'],self.kpe['kperp_size']).T.flatten(),
        #                   self.kpe['kperp_start_flat'].reshape(self.kpa['kpar_size'],self.kpe['kperp_size']).T.flatten(),
        #                   self.kpe['kperp_end_flat'].reshape(self.kpa['kpar_size'],self.kpe['kperp_size']).T.flatten()))
        bounds = list(zip(self.kpa['kpar_start_flat'],
                          self.kpa['kpar_end_flat'],
                          self.kpe['kperp_start_flat'],
                          self.kpe['kperp_end_flat']))
        self.bounds = bounds
        
        
        band_func = [self.bandfunc_2d_cart(*bound) for bound in self.bounds]
        self.band_func = band_func
        
    """
    Method 1: bandfunc
    """
    
    
    def get_ps2d_bandfunc(self, PK_k_zClass, pspackage, q_perp, q_par, currentparams, f_growth, bias=1, PKinterp=None):  #, currentparams
        
        # h = currentparams['H0']/100
        
        
        rescaling_factor = 1/(q_perp**2 * q_par)
        
        def P_kmu(k,mu):
            try:
                assert PKinterp == None
                if pspackage == 'class':
                    pofk_final = lambda k: PK_k_zClass(k,self.redshift_from_hiraxoutput)
                elif pspackage == 'camb':
                    pofk_final = lambda k: PK_k_zClass(k)
            except:
                print("PKinterp argument entered! Running exception")
                if pspackage == 'class':
                    pofk_final = lambda k: PKinterp(k, self.redshift_from_hiraxoutput)
                elif pspackage == 'camb':
                    pofk_final = lambda k: PKinterp.P(self.redshift_from_hiraxoutput , k)
                    
            return pofk_final(k) * (bias + f_growth * mu**2)**2
        
        """
        check comment here!
        """
        self.band_pk = [(lambda bandt: (lambda k, mu: 
                                        rescaling_factor 
                                        * P_kmu(self.k_obs(k,mu,
                                                           qpar=q_par,qperp=q_perp), 
                                                self.mu_obs(mu,
                                                            qpar=q_par,qperp=q_perp)) 
                                        * bandt(k, mu)
                                        # check if you need to enter kobs and muobs as bandt args
                                        )
                         )
                        (band)
                        for band in self.band_func]
        
        
        
        
        
        # print('i am here')
        
        psds1 = []
        iii = 0
        
        for k1 in self.kpar_center_1d:
            for k2 in self.kperp_center_1d:
                kpar1, kperp1 = k1 , k2
                k = np.sqrt(kpar1**2+kperp1**2)
                mu = kpar1/k
                psds1.append(self.band_pk[iii](k, mu))
                iii += 1
        
        # print('band_func used')
        
        
        # for k1 in self.kperp_center_1d.reshape(-1):
        #     for k2 in self.kpar_center_1d.reshape(-1):
        #         kperp1,kpar1 = k1/q_perp ,k2/q_par
        #         k = np.sqrt(kpar1**2+kperp1**2)
        #         mu = kpar1/k
        #         # psds1.append(rescaling_factor * band_pk1[iii](k_obs(k,mu),mu_obs(mu)))
        #         if pspackage == 'class':
        #             psds1.append(rescaling_factor * PK_k_zClass(k,self.redshift_from_hiraxoutput))
        #         else:
        #             psds1.append(rescaling_factor * PK_k_zClass(k))
        #         iii += 1
        # print('band_func not used')
        
        
        psds1 = np.array(psds1)
        # psds1 = psds1.reshape(self.kperp_c.shape[0],self.kpar_c.shape[0])
        psds1 = psds1.reshape(self.hirax_output.kpar_size,self.hirax_output.kperp_size)
        
        return psds1
        
        
            
    
    # def get_ps2d_bandfunc_old(self, PK, currentparams, redshift , pspackage):
    #     """
    #     Works only for bandfunc method
        
    #     Parameters
    #     ----------
    #     PK : function 
    #         DESCRIPTION. CLASS: function of k: PK(k * h0, redshift) *  h0**3 
    #                      CAMB : function of k: PK.P(redshift, k)
                         
    #     redshift : float
    #         DESCRIPTION. redshift at which the power spectrum has to be calculated.
    #     pspackage : string, optional
    #         DESCRIPTION. The default is 'class'.
            
    #     Returns
    #     -------
    #     psds1 : array
    #         array of shape of kpar x kperp (eg. 9x30). 
        
    #     """
        
    #     h0 = currentparams['H0']/100
        
    #     if pspackage == 'class':
    #         band_pk1 = [(lambda bandt: (lambda k, mu: PK(k*h0, redshift)*(h0)**3 * bandt(k, mu)))(band) for band in self.band_func]
    #     elif pspackage == 'camb':
    #         band_pk1 = [(lambda bandt: (lambda k, mu: PK.P(redshift , k) * bandt(k, mu)))(band) for band in self.band_func]
        
    #     psds1 = []
    #     iii = 0
    #     for k1 in self.kperp_c.reshape(-1):
    #         for k2 in self.kpar_c.reshape(-1):
    #             kperp1,kpar1 = k1,k2
    #             k = np.sqrt(kpar1**2+kperp1**2)
    #             mu = kpar1/k
    #             psds1.append(band_pk1[iii](k,mu))
    #             iii += 1
        
    #     psds1 = np.array(psds1)
    #     psds1 = psds1.reshape(self.kperp_c.shape[0],self.kpar_c.shape[0])
        
        
    #     return psds1
    
    
    
    """
    Method 2: Kaiser formula
    """
    
    def get_ps2d_kaiser(self, PK,kpar,kperp,z=0,b=1,f=1):
        """
        Get 2d-powerspectrum from the given input P(k) function as a function 
        of <kpar> and <kperp>.

        Parameters
        ----------
        PK : TYPE function
            DESCRIPTION. CLASS: function of k: PK(k * h0, redshift) *  h0**3 
                         CAMB : function of k: PK.P(redshift, k) 
        kpar : TYPE np.array
            DESCRIPTION. k_parallel modes (30 x 9 shape array)
        kperp : TYPE np.array
            DESCRIPTION. k_perpendicular modes (30 x 9 shape array)
            
        z : TYPE, float
            DESCRIPTION. The default redshift is 0.
        b : TYPE, float
            DESCRIPTION. The default is 1.
        f : TYPE, float
            DESCRIPTION. The default is 1.
            
        Returns
        -------
        TYPE np.array
            DESCRIPTION. numpy array of shape 

        """
        beta = f/b
        k = np.sqrt(kpar**2+kperp**2)
        mu = kpar/k
        return b**2*(1+beta*mu**2)**2*PK.P(z,k)
    



# =============================================================================
# CAMB/CLASS
# =============================================================================



class CreatePs2d:
    
    def __init__(self, inputforhiraxoutput, pspackage='class', pstype = 'param'):
        """
        
        Parameters
        ----------
        pspackage : STR, optional
            DESCRIPTION. The default is 'class'. Choose between ['class', 'camb']
        pstype : STR, optional
            DESCRIPTION. The default is 'param'. Choose between ['param', 'sample']

        Returns
        -------
        None.
        
        """
        
        self.modulelocation = os.path.dirname(hiraxmcmc.__file__)
        self.pspackage = pspackage
        self.pstype = pstype
        self.uname = os.uname()[1]
        self.parameters_fixed = ParametersFixed()
        self.cosmoparams_fixed = self.parameters_fixed.cosmoparams_fixed
        self.inputforhiraxoutput = inputforhiraxoutput
        
        # ====================================================================
        if pstype == 'sample':
            self.OmMh2 = self.parameters_fixed.OmM_fix * self.parameters_fixed._h_fix**2 #self.parameters_fixed.Om_to_omh2(self.parameters_fixed.OmM_fix, self.parameters_fixed.H0_fix)
        elif pstype == 'param':
            self.OmGv = self.parameters_fixed.OmG_fix
        
        # self.findcambinifile = find_file('planck_2018.ini','~')
        # ====================================================================
        if pspackage == 'camb':
            self.cambpars = camb.read_ini(os.path.join(self.modulelocation, 'planckfiles', 'planck_2018.ini'))
        elif pspackage == 'class':
            self.cambpars = camb.read_ini(os.path.join(self.modulelocation,'planckfiles', 'planck_2018.ini'))
            self.classpars = ConfigObj(os.path.join(self.modulelocation, 'planckfiles', 'base_2018_plikHM_TTTEEE_lowl_lowE_lensing.ini'))
            for key1, value1 in self.classpars.items():
                try:
                    self.classpars[key1] = float(value1)
                    if 'verbose' in key1: 
                        self.classpars[key1] = 0   # this is to switch of the verbose in class computation
                except:
                    pass
            # for key in self.classpars.keys():      
            #     if 'verbose' in key: 
            #         self.classpars[key] = 0
            self.pcl = Class()
        
        self.ps2d_from_Pofk = Ps2dFromPofk(inputforhiraxoutput = self.inputforhiraxoutput)
        
        self.redshift_from_hiraxoutput = self.ps2d_from_Pofk.redshift_from_hiraxoutput
        
        # self.classprecisionsettings = {'k_min_tau0':0.002,
        #                           'k_max_tau0_over_l_max':3.,
        #                           'k_step_sub':0.015,
        #                           'k_step_super':0.0001,
        #                           'k_step_super_reduction':0.1,
        #                           'start_small_k_at_tau_c_over_tau_h': 0.0004,
        #                           'start_large_k_at_tau_h_over_tau_k' : 0.05,
        #                           'tight_coupling_trigger_tau_c_over_tau_h':0.005,
        #                           'tight_coupling_trigger_tau_c_over_tau_k':0.008,
        #                           'start_sources_at_tau_c_over_tau_h': 0.006,
        #                           # 'tol_perturb_integration':1.e-6,
        #                           # 'perturb_sampling_stepsize':0.01
        #                           }
        
      
    # =========================================================================
    # =========================================================================
    # =========================================================================
    
    
    def pofk_from_camb(self, 
                       currentparams =  ParametersFixed().cosmoparams_fixed,
                       z = None,
                       output_CAMB_instance = True,
                       k_hunit_override = None,
                       hubble_units_override = None,
                       ):   # redshift should be overwritten by self.redshift_from_hiraxoutput
        """
        This function generates matter power spectrum P(k) at redshift <z> for
        given values of parameters using CAMB.

        Parameters
        ----------
        currentparamsfixed : DICT, optional
            DESCRIPTION. The default is ParametersFixed().current_params_fixed.
            Values can be entered as 
            {'H0': 67.8, 'Omk': 0.0, 'Oml': 0.684, 'w0': -1.0, 'wa': 0.0}
        z : float, optional
            DESCRIPTION. The default redshift is 1.5.
        
        Returns
        -------
        pk : function
            DESCRIPTION. function of k and redshift <z>
            pk = lambda kh: PK.P(z, kh)
            
            hubble_units   -->   power spectra 
            -----------------------------------
               True                 (Mpc/h)^3
               False                (Mpc)^3
              
            k_hunit        -->   power spectra
            -----------------------------------
               True                 P(k*h)
               False                P(k)
            
            
        """
        
        if z == None:
            zv = self.redshift_from_hiraxoutput
        else:
            zv = z
        
        currentparamstemp = currentparams.copy()
        for i in self.cosmoparams_fixed:
            try:
                assert i in currentparamstemp.keys()
            except:
                currentparamstemp[i] = self.cosmoparams_fixed[i]
        
        # H0v, Omkv, Omlv, w0v, wav = currentparams.values()
        
        try:
            h = currentparamstemp['h']
            H0 = h * 100
        except:
            H0 = currentparamstemp['H0']
            h = H0/100
            
        
        if self.pstype == 'sample':
            # ombh2v = self.OmMh2 - self.cambpars.omch2 - self.cambpars.omnuh2
            omch2v = self.OmMh2 - self.cambpars.ombh2 - self.cambpars.omnuh2
        elif self.pstype == 'param':
            omch2v = (1 - currentparamstemp['Oml'] - currentparamstemp['Omk'] - self.OmGv) * h**2 - self.cambpars.ombh2 - self.cambpars.omnuh2 
            # ombh2v = (1 - Omlv - Omkv - self.OmGv)*(H0v/100)**2 - self.cambpars.omch2 - self.cambpars.omnuh2 
        
        
        kmax = 20.0
        
        
        
        
        self.cambpars.set_cosmology(H0 = H0 , omk = currentparamstemp['Omk'], ombh2 = self.cambpars.ombh2, omch2 = omch2v)
        self.cambpars.NonLinear = model.NonLinear_both
        self.cambpars.DarkEnergy.set_params(w = currentparamstemp['w0'] , wa = currentparamstemp['wa'])
        
        self.cambresults = camb.get_results(self.cambpars)
        
        
        try:
            assert k_hunit_override != None
            k_hunit_val = k_hunit_override
        except:
            assert k_hunit_override == None
            k_hunit_val = True
        
        
        try:
            assert hubble_units_override != None
            hubble_units_val = hubble_units_override
        except:
            assert hubble_units_override == None
            hubble_units_val = False
        
        PK = camb.get_matter_power_interpolator(self.cambpars, 
                                                nonlinear=False, 
                                                kmax=kmax,
                                                zmax=250, 
                                                k_hunit=k_hunit_val,
                                                hubble_units=hubble_units_val)
        
        
        pk_kh = lambda kh: PK.P(zv, kh)
        # the input k from hirax mmode runs are in h units already (i.e., h/Mpc)
            # So, we don't need to use the k_hunit = True here, 
            # because using this multiplies the arg-input k's by h
            # and we don't want the mmode input k (h/Mpc) to be further multiplied by h
        
        
        if output_CAMB_instance:
            return pk_kh, self.cambresults
        else:
            return pk_kh
    
    # =========================================================================
    
    
    def pofk_from_class(self,
                        currentparams, #=  ParametersFixed().current_params_fixed,
                        z = None, 
                        output_CLASS_instance = True,
                        k_hunit_override = None,
                        hubble_units_override = None
                        ):   # redshift should be overwritten by self.redshift_from_hiraxoutput
        """
        This function generates matter power spectrum P(k) at redshift <z> for
        given values of parameters using CLASS.
        
        k-units: When the k_hunits is True, the input arguments (k values) need 
        to be in h-units so that the function multiplies inherently the input by
        the h value (input in currentparams).
        When the k_hunits is False, the input k should be in Mpc^-1 units and not 
        h/Mpc units.
        
        Parameters
        ----------
        currentparamsfixed : DICT, optional
            DESCRIPTION. The default is ParametersFixed().current_params_fixed.
            Values can be entered as 
            {'H0': 67.8, 'Omk': 0.0, 'Oml': 0.684, 'w0': -1.0, 'wa': 0.0}
        z : float, optional
            DESCRIPTION. The default redshift is ...

        Returns
        -------
        pk1 : function
            DESCRIPTION. function of k and redshift <z>
            pk1 = lambda kh: PK(kh * h0, z) *  h0**3
        
        """
        # if z == None:
        #     zv = self.redshift_from_hiraxoutput
        # else:
        #     zv = z
        
        currentparamstemp = currentparams.copy()
        for i in self.cosmoparams_fixed:
            try:
                assert i in currentparamstemp.keys()
            except:
                currentparamstemp[i] = self.cosmoparams_fixed[i]
        
        try:
            h = currentparamstemp['h']
            H0 = h * 100
        except:
            H0 = currentparamstemp['H0']
            h = H0/100
        
        # H0v, Omkv, Omlv, w0v, wav = currentparams.values()
        
        # h = currentparamstemp['H0']/100
        
        if self.pstype == 'param':
            omch2 = (1 - currentparamstemp['Oml'] - currentparamstemp['Omk'] - self.OmGv) * h**2 - self.classpars['omega_b'] - self.cambpars.omnuh2
            # ombh2 = (1 - Omlv - Omkv - self.OmGv) * (H0v/100)**2 - self.cambpars.omch2 - self.cambpars.omnuh2
        elif self.pstype == 'sample':
            omch2 = self.OmMh2 - self.classpars['omega_b'] - self.cambpars.omnuh2
            # ombh2 = self.OmMh2 - self.cambpars.omch2 - self.cambpars.omnuh2
        
        self.pcl.set(dict(self.classpars))
        
        self.pcl.set({'H0': H0,
                      'omega_b': float(self.classpars['omega_b']),
                      'omega_cdm': omch2,
                      'Omega_k': currentparamstemp['Omk'],
                      'Omega_Lambda': currentparamstemp['Oml'],
                      'w0_fld': currentparamstemp['w0'],
                      'wa_fld': currentparamstemp['wa'],
                      })
        
        self.pcl.set({'lensing':'no',
                      'output':'mPk',
                      'P_k_max_h/Mpc':20.0,
                      'z_max_pk':5,
                      'non linear':'none'
                      })
        
        # self.pcl.set(self.classprecisionsettings)
        
        
        self.pcl.compute()
        
        
        PK = self.pcl.pk
        
        
        try:
            assert k_hunit_override != None
            k_hunit_val = k_hunit_override
        except:
            assert k_hunit_override == None
            k_hunit_val = True
        
        
        try:
            assert hubble_units_override != None
            hubble_units_val = hubble_units_override
        except:
            assert hubble_units_override == None
            hubble_units_val = False
        
            
        if k_hunit_val == True and hubble_units_val==True:
            pk_k_z = lambda k,zv: PK(k*h, zv) * h**3
        elif k_hunit_val == True and hubble_units_val==False:
            pk_k_z = lambda k,zv: PK(k*h, zv)
        elif k_hunit_val == False and hubble_units_val==True:
            pk_k_z = lambda k,zv: PK(k, zv) * h**3
        elif k_hunit_val == False and hubble_units_val==False:
            pk_k_z = lambda k,zv: PK(k, zv)
        
        
        
        if output_CLASS_instance:
            return pk_k_z, self.pcl
        else:
            return pk_k_z
    
    # =========================================================================
    # =========================================================================
    # =========================================================================
    
    def get_ps2d_from_pok(self,                                       # *^*^*^*^*^*^*^*^*^*^*^*^*
                          PK_k_zClass,
                          q_perp_input, q_par_input,
                          currentparams_input,
                          f_growth, 
                          z=None):                          # currentparams,
        
        # if z == None:
        #     zv = self.redshift_from_hiraxoutput
        # else:
        #     zv = z
        
        psds = self.ps2d_from_Pofk.get_ps2d_bandfunc(PK_k_zClass, 
                                                     pspackage=self.pspackage, 
                                                     q_perp = q_perp_input, 
                                                     q_par = q_par_input,
                                                     currentparams = currentparams_input,
                                                     f_growth = f_growth) #currentparams, 
        
        return psds
    
    
    
    
    # =========================================================================
    # =========================================================================
    # =========================================================================
    
    
    # def create_ps2d_from_camb(self,
    #                           currentparams =  ParametersFixed().current_params_fixed,
    #                           z = None):     # redshift should be overwritten by self.redshift_from_hiraxoutput
        
        
    #     if z == None:
    #         zv = self.redshift_from_hiraxoutput
    #     else:
    #         zv = z
            
    #     H0v, Omkv, Omlv, w0v, wav = currentparams.values()
        
    #     if self.pstype == 'sample':
    #         omch2v = self.OmMh2 - self.cambpars.ombh2 - self.cambpars.omnuh2
    #         # ombh2v = self.OmMh2 - self.cambpars.omch2 - self.cambpars.omnuh2
    #     elif self.pstype == 'param':
    #         omch2v = (1 - Omlv - Omkv - self.OmGv)*(H0v/100)**2 - self.cambpars.ombh2 - self.cambpars.omnuh2 
    #         # ombh2v = (1 - Omlv - Omkv - self.OmGv)*(H0v/100)**2 - self.cambpars.omch2 - self.cambpars.omnuh2 
        
    #     kmax = 1.0
        
    #     self.cambpars.set_cosmology(H0 = H0v , omk = Omkv, ombh2 = self.cambpars.ombh2, omch2 = omch2v)
    
    #     self.cambpars.NonLinear = model.NonLinear_both
    #     self.cambpars.DarkEnergy.set_params(w = w0v , wa = wav)
        
    #     self.cambresults = camb.get_results(self.cambpars)
        
    #     PK = camb.get_matter_power_interpolator(self.cambpars, nonlinear=True, kmax=kmax, zmax=250)
        
        
    #     psds = self.ps2d_from_Pofk.get_ps2d_bandfunc(PK, redshift= zv, pspackage=self.pspackage)
        
        
        
    #     return psds
    
    # def create_ps2d_from_class(self,
    #                            currentparams , 
    #                            z):
    #     """
        
        
    #     Parameters
    #     ----------
    #     currentparams : dict
    #         dict object of the form 
    #     z : float, optional
    #         Redshift. The default is 1.5.

    #     Returns
    #     -------
    #     psds : np.array
    #         2d powerspectrum.

    #     """
    #     # t0 = time.time()
    #     H0v, Omkv, Omlv, w0v, wav = currentparams.values()
        
    #     if self.pstype == 'param':
    #         # ombh2 = (1 - Omlv - Omkv - self.OmGv) * (H0v/100)**2 - self.cambpars.omch2 - self.cambpars.omnuh2
    #         omch2 = (1 - Omlv - Omkv - self.OmGv) * (H0v/100)**2 - self.cambpars.ombh2 - self.cambpars.omnuh2
    #     elif self.pstype == 'sample':
    #         # ombh2 = self.OmMh2 - self.cambpars.omch2 - self.cambpars.omnuh2
    #         omch2 = self.OmMh2 - self.cambpars.ombh2 - self.cambpars.omnuh2
        
    #     self.pcl.set(dict(self.classpars))
        
    #     self.pcl.set({'H0': H0v, 
    #                   'omega_b': self.cambpars.ombh2,
    #                   'omega_cdm': omch2,
    #                   'Omega_k': Omkv,
    #                   'Omega_Lambda': Omlv,
    #                   'w0_fld': w0v,
    #                   'wa_fld': wav,
    #                   })
        
    #     self.pcl.set({'lensing':'no',
    #                   'output':'mPk',
    #                   'P_k_max_h/Mpc':2.0,
    #                   'z_max_pk':2.6
    #                   # 'non linear':'hmcode'
    #                   })
        
        
    #     self.pcl.compute()
        
        
    #     PK = self.pcl.pk
        
        
        
    #     psds = self.ps2d_from_Pofk.get_ps2d_bandfunc(PK, redshift= z, pspackage=self.pspackage)
        
    #     self.pcl.struct_cleanup()     
    #     self.pcl.empty()
    #     # THIS IS THE REASON I DON'T USE PofK FUNCTION IN ANOTHER FUNCTION TO GET PS2D
        
    #     # print(time.time()-t0)
    
    #     return psds 
    
    # =========================================================================
    # =========================================================================
    # =========================================================================
    
    # def get_ps2d_from_pok_camb(self,                                        # *^*^*^*^*^*^*^*^*^*^*^*^*
    #                             PK_k_zClass,
    #                             # currentparams,
    #                             q_perp_input, q_par_input,
    #                             z=None):
        
    #     # if z == None:
    #     #     zv = self.redshift_from_hiraxoutput
    #     # else:
    #     #     zv = z
        
        
    #     psds = self.ps2d_from_Pofk.get_ps2d_bandfunc(PK_k_zClass, 
    #                                                  pspackage=self.pspackage, 
    #                                                  q_perp=q_perp_input, 
    #                                                  q_par=q_par_input) #, currentparams
        
    #     return psds
        
        
    # def get_ps2d_from_pok_class(self,                                       # *^*^*^*^*^*^*^*^*^*^*^*^*
    #                             PK_k_zClass,
    #                             # currentparams,
    #                             q_perp_input, q_par_input,
    #                             z=None):
        
    #     # if z == None:
    #     #     zv = self.redshift_from_hiraxoutput
    #     # else:
    #     #     zv = z
        
    #     psds = self.ps2d_from_Pofk.get_ps2d_bandfunc(PK_k_zClass, 
    #                                                  pspackage=self.pspackage, 
    #                                                  q_perp=q_perp_input, 
    #                                                  q_par=q_par_input) #currentparams, 
        
    #     return psds
    
    #     # self.pcl.struct_cleanup()     |---> NOT DONE IN THIS FUNCTION!!!!
    #     # self.pcl.empty()              /
    
        
    
    
    # =========================================================================
    # =========================================================================
    # =========================================================================
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # def get_ps2dDict_from_class_multz(self,
    #                                   currentparams,
    #                                   z_arr):
        
    #     H0v, Omkv, Omlv, w0v, wav = currentparams.values()
        
    #     if self.pstype == 'param':
    #         # ombh2 = (1 - Omlv - Omkv - self.OmGv) * (H0v/100)**2 - self.cambpars.omch2 - self.cambpars.omnuh2
    #         omch2 = (1 - Omlv - Omkv - self.OmGv) * (H0v/100)**2 - self.cambpars.ombh2 - self.cambpars.omnuh2
    #     elif self.pstype == 'sample':
    #         # ombh2 = self.OmMh2 - self.cambpars.omch2 - self.cambpars.omnuh2
    #         omch2 = self.OmMh2 - self.cambpars.ombh2 - self.cambpars.omnuh2
        
    #     self.pcl.set(dict(self.classpars))
        
    #     self.pcl.set({'H0': H0v, 
    #                   'omega_b': self.cambpars.ombh2,
    #                   'omega_cdm': omch2,
    #                   'Omega_k': Omkv,
    #                   'Omega_Lambda': Omlv,
    #                   'w0_fld': w0v,
    #                   'wa_fld': wav,
    #                   })
        
    #     self.pcl.set({'lensing':'no',
    #                   'output':'mPk',
    #                   'P_k_max_h/Mpc':2.0,
    #                   'z_max_pk':2.6
    #                   # 'non linear':'hmcode'
    #                   })
        
        
    #     self.pcl.compute()
        
        
    #     PK = self.pcl.pk
        
        
    #     psdsdict = {}
        
    #     for z in z_arr:
    #         psdsdict[z] = self.ps2d_from_Pofk.get_ps2d_bandfunc(PK, redshift= z, pspackage=self.pspackage)
            
        
    #     self.pcl.struct_cleanup()     
    #     self.pcl.empty()
        
    #     # print(time.time()-t0)
        
    #     return psdsdict




















        
