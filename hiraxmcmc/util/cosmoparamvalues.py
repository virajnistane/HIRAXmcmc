#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.integrate import cumtrapz, trapz
from scipy.constants import speed_of_light as cc
import numpy as np

from hiraxmcmc.util.basicfunctions import *

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
    # Parsecs in m
    _parsec_in_m = 3.08568025e16
    # Parsecs in km
    _parsec_in_km = 3.08568025e13
    # CMB temperature
    _T0 = 2.7255
    
    def __init__(self):
        
        
        # 68% limit values in 
        # case 2.5 of 
        # https://wiki.cosmos.esa.int/planck-legacy-archive/images/b/be/Baseline_params_table_2018_68pc.pdf
        
        
        # Hubble parameter
        self._h = 0.6732117
        self._H0 = self._h * 100
        
        rhoc = 3.0 * self._H0**2 * cc**2 / (8.0 * np.pi * self._G) / (1e6 * self._parsec_in_km)**2
        
        # Omega_matter
        self._ombh2 = 0.02238280
        self._omch2 = 0.1201075
        self._Omb = self._ombh2/self._h**2
        self._Omc = self._omch2/self._h**2
        
        # self.N_ncdm = 1
        # self.m_ncdm = 0.06 
        # self.T_ncdm = 0.7137658555036082   # (4/11)^(1/3)
        # self.rho_ncdm = self.m_ncdm * self.N_ncdm
        # self._Omncdm = self.rho_ncdm / rhoc
        
        self._OmM = (self._ombh2 + self._omch2)/self._h**2
        # self._OmM = 0.3158 # (self._ombh2 + self._omch2)/self._h**2 + self._Omncdm
        # self._Omncdm = self._OmM - self._Omb - self._Omc
        
        # Omega_curvature
        self._Omk = 0.0
        
        
        # Omega_radiation
        
        # Omega_gamma
        rhog = self._a_rad * self._T0**4
        self._Omg = rhog / rhoc
        # Omega_nu
        self.N_ur = 3.046
        rhonu = self.N_ur * rhog * 7/8 * (4/11)**(4/3)
        self._Omnu = rhonu / rhoc
        self._omnuh2 = self._Omnu * self._h**2
            # Omega_r = Omega_nu + Omega_gamma
        self._Omr = self._Omg + self._Omnu
        
        
        # self._Omr = 1 - self._Oml - self._OmM - self._Omk
        # self._Omnu = self._Omr - self._Omg 
        # rhonu = self._Omnu * rhoc
        
        
        
        # Finally, evaluating Omega_Lambda from the other fixed parameters
        self._Oml = 1 - self._Omk - self._OmM - self._Omr
        # self._Omk = 1 - self._Oml - self._OmM  - self._Omr
        
        
        self._w0 = -1.0
        self._wa = 0.0
        
        
        self._hz = {}
        for key in ['400_500','500_600','600_700','700_800']:
            self._hz[key] = self.hz_fun(freq2z( (float(key.split('_')[0]) + float(key.split('_')[1]))/2))
        
        self._Hz = {}
        for key,val in self._hz.items():
            self._Hz[key] = val*cc/1e3
        self._qpar = 1
        
        # _dA = {}
        # for key in ['400_500','500_600','600_700','700_800']:
        #     _dA[key] = pkpr.angular_distance(freq2z( (float(key.split('_')[0]) + float(key.split('_')[1]))/2))
        self._dA = {'400_500': 1758.2926246185887,
                    '500_600': 1802.503416422851,
                    '600_700': 1767.395306138714,
                    '700_800': 1666.4200025225027}
        self._qperp = 1
        
        self._fz = {}
        for key in ['400_500','500_600','600_700','700_800']:
            self._fz[key] = self.fz_fun(freq2z( (float(key.split('_')[0]) + float(key.split('_')[1]))/2))
    
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
    def Omc_fid(self):
        return self._Omc
    @Omc_fid.setter
    def Omc_fid(self, Omc_fid_new):
        self._Omc = Omc_fid_new
    
    @property
    def ombh2_fid(self):
        return self._ombh2
    @ombh2_fid.setter
    def ombh2_fid(self, ombh2_fid_new):
        self._ombh2 = ombh2_fid_new
        
    @property
    def Omb_fid(self):
        return self._Omb
    @Omb_fid.setter
    def Omb_fid(self, Omb_fid_new):
        self._Omb = Omb_fid_new
        
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
        return {'h':self._h, 'Omb':self._Omb, 'Oml':self._Oml, 'w0':self._w0, 'wa':self._wa,
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
    
    
    ###### ###### ###### ######
    
    def hz_fun(self, z):   # in Mpc^-1
        ez = np.sqrt(self._OmM * (1+z)**3 
                     + self._Oml * (1+z)**(3*(1+self._w0+self._wa)) * np.exp(-3*self._wa * (z/(1+z)))
                     + self._Omk * (1+z)**2 + self._Omr * (1+z)**4
                     )
        return self._h * ez * 1e5/cc

    def dAz_fun(self, z):  # in Mpc
        omk = self._Omk
        aa_arr = np.linspace(1, 1e-4, 10000)
        # zarr = np.linspace(0,1e4, 100000)
        ea_fun = lambda a: np.sqrt(self._OmM * a**-3 
                                   + self._Oml * a**(-3*(1+self._w0+self._wa)) * np.exp(3*self._wa * (a-1))
                                   + self._Omk * a**-2 
                                   + self._Omr * a**-4)
        rz = cc/1e3 / self._H0 * trapz(ea_fun(aa_arr)**(-1) * aa_arr**(-2), aa_arr[::-1])#, initial=0)
        if omk < 0:
            return (1+z)**(-1) * cc/1e3/self._H0 * 1/np.sqrt(
                abs(omk)) * np.sin(np.sqrt(abs(omk))*self._H0/(cc/1e3) * rz)
        elif omk == 0:
            return (1+z)**(-1) * rz
        elif omk > 0:
            return (1+z)**(-1) * cc/1e3/self._H0 * 1/np.sqrt(
                omk) * np.sinh(np.sqrt(omk)*self._H0/(cc/1e3) * rz)
    
    def fz_fun(self,z):
        ez = np.sqrt(self._OmM * (1+z)**3 
                     + self._Oml * (1+z)**(3*(1+self._w0+self._wa)) * np.exp(-3*self._wa * (z/(1+z)))
                     + self._Omk * (1+z)**2 + self._Omr * (1+z)**4
                     )
        return (self._OmM * (1+z)**3/ez**2)**(0.55)
    
    
    
    
    
    
    
    
    
    
