#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,sys
from hiraxmcmc.util.basicfunctions import *

# =============================================================================
# Directory management
# =============================================================================


class DirManage:
    
    def __init__(self, mcmc_mainrun_dir_relpath, currentrunindex, testfilekw):
        self.testfilekw = testfilekw
        self.mcmc_mainrun_dir_relpath = mcmc_mainrun_dir_relpath
        self.currentrunindex = currentrunindex
    
    def outputdir_for_class(self):
        try:
            if not os.path.exists(os.path.join(self.mcmc_mainrun_dir_relpath, 'output')):
                os.mkdir('output')
            if not os.path.exists(os.path.join(self.mcmc_mainrun_dir_relpath, 'output')):
                os.mkdir(os.path.join(self.mcmc_mainrun_dir_relpath,'output'))
        except:
            print('<<output>> for class alredy exists'); sys.stdout.flush()    

    
    def chaindir(self):
        try:
            if not os.path.exists(os.path.join(self.mcmc_mainrun_dir_relpath, 'chains')):
                os.mkdir(os.path.join(self.mcmc_mainrun_dir_relpath, 'chains'))
        except:
            print('<<chains>> alredy exists'); sys.stdout.flush()
        
        try:
            if not os.path.exists(os.path.join(self.mcmc_mainrun_dir_relpath, 'chains/run%s'%(self.currentrunindex))):
                os.mkdir(os.path.join(self.mcmc_mainrun_dir_relpath, 'chains/run%s'%(self.currentrunindex)))
        except:
            print('<<chains/run??>> alredy exists'); sys.stdout.flush()
    
        try:
            if not os.path.exists(os.path.join(self.mcmc_mainrun_dir_relpath, 'chains/run%s/trulyAccepted'%(self.currentrunindex))):
                os.mkdir(os.path.join(self.mcmc_mainrun_dir_relpath, 'chains/run%s/trulyAccepted'%(self.currentrunindex)))
        except:
            print('<<chains/run??/trulyAccepted>> alredy exists'); sys.stdout.flush()
        
    def TRFdir(self):
        try:
            if not os.path.exists(os.path.join(self.mcmc_mainrun_dir_relpath, 'TRF')):
                os.mkdir(os.path.join(self.mcmc_mainrun_dir_relpath, 'TRF'))
        except:
            print('<<TRF>> alredy exists'); sys.stdout.flush()
            
        try:
            if not os.path.exists(os.path.join(self.mcmc_mainrun_dir_relpath, 'TRF%s/run%s'%(self.testfilekw,self.currentrunindex))):
                os.mkdir(os.path.join(self.mcmc_mainrun_dir_relpath, 'TRF%s/run%s'%(self.testfilekw,self.currentrunindex)))
        except:
            print('<<TRF/run>> alredy exists'); sys.stdout.flush()
            
    def ARdir(self):
        try:
            if not os.path.exists(os.path.join(self.mcmc_mainrun_dir_relpath, 'AR')):
                os.mkdir(os.path.join(self.mcmc_mainrun_dir_relpath, 'AR'))
        except:
            print('<<AR>> alredy exists'); sys.stdout.flush()
            
        try:
            if not os.path.exists(os.path.join(self.mcmc_mainrun_dir_relpath, 'AR%s/run%s'%(self.testfilekw,self.currentrunindex))):
                os.mkdir(os.path.join(self.mcmc_mainrun_dir_relpath, 'AR%s/run%s'%(self.testfilekw,self.currentrunindex)))
        except:
            print('<<AR/run>> alredy exists'); sys.stdout.flush()
    
    def extractsforMLdir(self):
        try:
            if not os.path.exists(os.path.join(self.mcmc_mainrun_dir_relpath, 'extracts_for_ML')):
                os.mkdir(os.path.join(self.mcmc_mainrun_dir_relpath,'extracts_for_ML'))
        except:
            print('<<extracts_for_ML>> alredy exists'); sys.stdout.flush()
            
        try:
            if not os.path.exists(os.path.join(self.mcmc_mainrun_dir_relpath, 'extracts_for_ML/run%s'%(self.currentrunindex))):
                os.mkdir(os.path.join(self.mcmc_mainrun_dir_relpath, 'extracts_for_ML/run%s'%(self.currentrunindex)))
        except:
            print('<<extracts_for_ML/run??>> alredy exists'); sys.stdout.flush()





            
            
