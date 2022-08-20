#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.constants import speed_of_light as cc
import numpy as np

# =============================================================================
# Fixed param values
# =============================================================================

class ParametersFixed:
    
    # if there is no __init__ function and if the values are instead defined just 
    # within the class, then the values won't reinitialize each time the class 
    # object is generated
    
    
    # Gravitational constant G kg^-1 m^3 s^-2
    _G = 6.6742e-11
    # Stefan-Boltzmann (in W m^{-2} K^{-4})
    _stefan_boltzmann = 5.6705e-8
    # Radiation constant (in J m^{-3} K^{-4})
    _a_rad = 4*_stefan_boltzmann / cc
    # Boltzmann constant
    _k_B = 1.3806503e-23
    # CMB temperature
    _T0 = 2.726
    # Parsecs in m
    _parsec_in_m = 3.08568025e16
    # Parsecs in km
    _parsec_in_km = 3.08568025e13
    
    def __init__(self):
        
        self._h = 0.6777
        self._H0 = self._h * 100
        
        self._ombh2 = 0.022161
        self._omch2 = 0.11889
        
        self._Omb = self._ombh2/self._h**2
        self._Omc = self._omch2/self._h**2
        
        self._Omk = 0.0
        
        self._OmM = (self._ombh2 + self._omch2)/self._h**2
        
        # fixing Omega_gamma
        rhoc = 3.0 * self._H0**2 * cc**2 / (8.0 * np.pi * self._G) / (1e6 * self._parsec_in_km)**2
        rhorad = self._a_rad * self._T0**4
        self._Omg = rhorad / rhoc
        
        # fixing Omega_nu
        self.nnu = 3.046
        rhonu = self.nnu * rhorad * 7.0 / 8.0 * (4.0 / 11.0)**(4.0 / 3.0)
        self._Omnu = rhonu / rhoc
        self._omnuh2 = self._Omnu * self._h**2
        
        # Adding Omega_gamma and Omega_nu, temporarily set to 0 for simplicity
        self._Omr = 0# self._Omg + self._Omnu
                
        # Finally, evaluating Omega_Lambda from the other fixed parameters
        self._Oml = 1 - self._Omk - self._OmM - self._Omr
        
        
        self._w0 = -1.0
        self._wa = 0.0
        
        self._hz = {'400_500':0.0007327671367151243, 
                    '500_600':0.0005568269434413366, 
                    '600_700':0.0004491941045627553,
                    '700_800':0.0003790722414922297}
        
        self._Hz = {}
        for key,val in self._hz.items():
            self._Hz[key] = val*cc/1e3
        
        self._qpar = 1
        
        self._dA = {'400_500':1753.152138846927, 
                    '500_600':1794.9333333728628, 
                    '600_700':1757.6610835353,
                    '700_800':1655.0761074055235}
        
        self._qperp = 1
        
        self._fz = {'400_500':0.9628415413719129, 
                    '500_600':0.9354237272393772, 
                    '600_700':0.8994759926141177, 
                    '700_800':0.8561943222181662}
        
    
    ###### ###### ###### ######
    
    @property
    def H0_fid(self):
        return self._H0
    @H0_fid.setter
    def H0_fid(self, H0_fid_new):
        self._H0 = H0_fid_new
        
    @property
    def h_fid(self):
        return self._h
    @h_fid.setter
    def h_fid(self, h_fid_new):
        self._h = h_fid_new
    
    @property
    def Omk_fid(self):
        return self._Omk
    @Omk_fid.setter
    def Omk_fid(self, Omk_fid_new):
        self._Omk = Omk_fid_new
    
    @property
    def Oml_fid(self):
        return self._Oml
    @Oml_fid.setter
    def Oml_fid(self, Oml_fid_new):
        self._Oml = Oml_fid_new
    
    @property
    def w0_fid(self):
        return self._w0
    @w0_fid.setter
    def w0_fid(self, w0_fid_new):
        self._w0 = w0_fid_new
    
    @property
    def wa_fid(self):
        return self._wa
    @wa_fid.setter
    def wa_fid(self, wa_fid_new):
        self._wa = wa_fid_new
    
    ###### ###### ###### ######
    
    @property
    def omch2_fid(self):
        return self._omch2
    @omch2_fid.setter
    def omch2_fid(self, omch2_fid_new):
        self._omch2 = omch2_fid_new
    
    @property
    def ombh2_fid(self):
        return self._ombh2
    @ombh2_fid.setter
    def ombh2_fid(self, ombh2_fid_new):
        self._ombh2 = ombh2_fid_new
        
    @property
    def OmM_fid(self):
        return self._OmM
    @OmM_fid.setter
    def OmM_fid(self, OmM_fid_new):
        self._OmM = OmM_fid_new
        
    @property
    def Omg_fid(self):
        return self._Omg
    @Omg_fid.setter
    def Omg_fid(self, Omg_fid_new):
        self._Omg = Omg_fid_new
    
    @property
    def Omnu_fid(self):
        return self._Omnu
    @Omnu_fid.setter
    def Omnu_fid(self, Omnu_fid_new):
        self._Omnu = Omnu_fid_new
        
    @property
    def omnuh2_fid(self):
        return self._omnuh2
    @omnuh2_fid.setter
    def omnuh2_fid(self, omnuh2_fid_new):
        self._omnuh2 = omnuh2_fid_new
    
    @property
    def Omr_fid(self):
        return self._Omr
    @Omr_fid.setter
    def Omr_fid(self, Omr_fid_new):
        self._Omr = Omr_fid_new
    
    ###### ###### ###### ######
    
    @staticmethod
    def h(H0):
        return H0/100.
    
    @classmethod
    def Om_to_omh2(cls, OmX, H0):
        h_temp = cls.h(H0)
        return OmX * h_temp**2
    
    @classmethod
    def omh2_to_Om(cls, omXh2, H0):
        h_temp = cls.h(H0)
        return omXh2/h_temp**2
    
    @property
    def current_allparams_fixed(self):
        return {'h':self._h, 'Omk':self._Omk, 'Oml':self._Oml, 'w0':self._w0, 'wa':self._wa,
                'h(z)':self._hz, 'qpar(z)':self._qpar, 'dA(z)':self._dA, 'qperp(z)':self._qperp, 'f(z)':self._fz}
    
    
    @property
    def cosmoparams_fixed(self):
        temp = {}
        for key,val in self.current_allparams_fixed.items():
            if '(z)' not in key:
                temp[key] = val
        return temp
    

    def current_params_to_vary_fixed(self, params_to_vary, toggle_paramstovary_freqdep, fcbin=None, external_current_allparams_fixed=None):
        """
        Function to get the values of the currently varying params (in the respective single freq bin)
        
        Parameters
        ----------
        params_to_vary : TYPE
            DESCRIPTION.
        toggle_paramstovary_freqdep : TYPE
            DESCRIPTION.
        fclist : TYPE
            DESCRIPTION.
        external_current_allparams_fixed : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        truncdict : TYPE
            DESCRIPTION.

        """

        truncdict = {}
        # for cosmological parameters
        if not(toggle_paramstovary_freqdep):
            # fclist = None
            for pp in params_to_vary:
                if external_current_allparams_fixed == None:
                    truncdict[pp] = self.current_allparams_fixed[pp]
                else:
                    truncdict[pp] = external_current_allparams_fixed[pp]
        # for scaling parameters (freq dependent)
        elif toggle_paramstovary_freqdep:
            for pp in params_to_vary:
                assert fcbin!=None
                # assert len(fclist) == 1 
                # Using the default current_allparams_fixed from the class 
                # property to get the values
                if external_current_allparams_fixed == None:
                    try:
                        # for qpar(z) and qperp(z), no freq channel keys
                        assert type(self.current_allparams_fixed[pp]) != dict
                        truncdict[pp] = self.current_allparams_fixed[pp]
                    except:
                        # for f(z)
                        truncdict[pp] = self.current_allparams_fixed[pp][fcbin]
                # Using an external list of current_allparams_fixed  
                # to get the values
                else:
                    try:
                        # for qpar(z) and qperp(z), no freq channel keys
                        assert type(self.current_allparams_fixed[pp]) != dict
                        truncdict[pp] = external_current_allparams_fixed[pp]
                    except:
                        # for f(z)
                        truncdict[pp] = external_current_allparams_fixed[pp][fcbin]
        return truncdict
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
