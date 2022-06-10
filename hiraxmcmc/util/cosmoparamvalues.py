#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.constants import speed_of_light as cc

# =============================================================================
# Fixed param values
# =============================================================================

class ParametersFixed:
    
    
    def __init__(self):
        self._H0_fix = 67.80
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
        
        self._dA_fix = {'400_500':1751.0952983505435, 
                        '500_600':1792.9837345626768, 
                        '600_700':1755.9106803363068, 
                        '700_800':1653.5773864998494}
        
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
    
    # ###### ###### ###### ######
    # allparamsfixdict = {'H0':H0_fix, 
    #                     'Omk':Omk_fix,
    #                     'Oml':Oml_fix, 
    #                     'w0':w0_fix, 
    #                     'wa':wa_fix,
    #                     'h(z)':_hz_fix,
    #                     'dA(z)':_dA_fix,
    #                     'f(z)':_fz_fix
    #                     }
    
    @property
    def current_allparams_fixed(self):
        return {'H0':self.H0_fix, 'Omk':self.Omk_fix, 'Oml':self.Oml_fix, 'w0':self.w0_fix, 'wa':self.wa_fix,
                'h(z)':self._hz_fix,'dA(z)':self._dA_fix, 'f(z)':self._fz_fix }
    
    @property
    def cosmoparams_fixed(self):
        temp = {}
        for key,val in self.current_allparams_fixed.items():
            if '(z)' not in key:
                temp[key] = val
        return temp
    
    
    def current_params_to_vary_fixed(self, params_to_vary, fclist, freqdep_paramstovary=False):
        truncdict = {}
        for pp in params_to_vary:
            try:
                assert not(freqdep_paramstovary)
                truncdict[pp] = self.current_allparams_fixed[pp]
            except:
                assert freqdep_paramstovary
                for fc in fclist:
                    truncdict[pp] = self.current_allparams_fixed[pp][fc]
        return truncdict
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
