#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.constants import speed_of_light as cc

# =============================================================================
# Fixed param values
# =============================================================================

class ParametersFixed:
    
    # if there is no __init__ function and if the values are instead defined just 
    # within the class, then the values won't reinitialize each time the class 
    # object is generated
    
    def __init__(self):
        self._H0_fix = 67.80
        self._h_fix = 0.678
        self._Omk_fix = 0.0
        self._Oml_fix = 0.684
        self._w0_fix = -1.0
        self._wa_fix = 0.0
        
        self._hz_fix = {'400_500':0.0007339864092978513, 
                        '500_600':0.0005577061879176558, 
                        '600_700':0.0004498518311851779, 
                        '700_800':0.000379575515395735 }
        
        self._Hz_fix = {'400_500':0.0007339864092978513 * cc/1e3, 
                        '500_600':0.0005577061879176558 * cc/1e3, 
                        '600_700':0.0004498518311851779 * cc/1e3, 
                        '700_800':0.000379575515395735  * cc/1e3}
        
        self._qpar_fix = 1
        
        self._dA_fix = {'400_500':1751.0952983505435, 
                        '500_600':1792.9837345626768, 
                        '600_700':1755.9106803363068, 
                        '700_800':1653.5773864998494}
        
        self._qperp_fix = 1
        
        self._fz_fix = {'400_500':0.9604491314463909, 
                        '500_600':0.9331817576718678, 
                        '600_700':0.8974290025235386, 
                        '700_800':0.8543701735721654}
        
        self._OmM_fix = 0.308
    
        # omch2_fix = 0.1201075
        # ombh2_fix = 0.0223828
    
        self.OmG_fix = 1 - self._OmM_fix - self._Omk_fix - self._Oml_fix
    
    ###### ###### ###### ######
    
    @property
    def H0_fix(self):
        return self._H0_fix
    @H0_fix.setter
    def H0_fix(self, H0_fix_new):
        self._H0_fix = H0_fix_new
        
    @property
    def h_fix(self):
        return self._h_fix
    @h_fix.setter
    def h_fix(self, h_fix_new):
        self._h_fix = h_fix_new
    
    @property
    def Omk_fix(self):
        return self._Omk_fix
    @Omk_fix.setter
    def Omk_fix(self, Omk_fix_new):
        self._Omk_fix = Omk_fix_new
    
    @property
    def Oml_fix(self):
        return self._Oml_fix
    @Oml_fix.setter
    def Oml_fix(self, Oml_fix_new):
        self._Oml_fix = Oml_fix_new
    
    @property
    def w0_fix(self):
        return self._w0_fix
    @w0_fix.setter
    def w0_fix(self, w0_fix_new):
        self._w0_fix = w0_fix_new
    
    @property
    def wa_fix(self):
        return self._wa_fix
    @wa_fix.setter
    def wa_fix(self, wa_fix_new):
        self._wa_fix = wa_fix_new
    
    ###### ###### ###### ######
    
    @property
    def OmM_fix(self):
        return self._OmM_fix
    @OmM_fix.setter
    def OmM_fix(self, OmM_fix_new):
        self._OmM_fix = OmM_fix_new
    
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
        return {'h':self._h_fix ,'Omk':self.Omk_fix, 'Oml':self.Oml_fix, 'w0':self.w0_fix, 'wa':self.wa_fix,
                'h(z)':self._hz_fix, 'qpar(z)':self._qpar_fix, 'dA(z)':self._dA_fix, 'qperp(z)':self._qperp_fix, 'f(z)':self._fz_fix }
    
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
