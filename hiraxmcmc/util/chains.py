#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,sys
from hiraxmcmc.util.basicfunctions import *


# =============================================================================
# CHAINS - saving and removing old files
# =============================================================================

class Chains:
    
    def __init__(self, currentrunindex, totalParams_inclChi2, rankmpi, comm, testfilekw, parameterssavetxt, mcmc_mainrun_dir_relpath, write_out_paramsTrulyAccepted = False, deletePrevChainFiles=False):
        
        self.columnsInFile = totalParams_inclChi2
        self.currentrunindex = currentrunindex
        
        self.testfilekw = testfilekw
        self.parameterssavetxt = parameterssavetxt
        
        self.uname = os.uname()[1]
        self.mcmc_mainrun_dir_relpath = mcmc_mainrun_dir_relpath
        
        self.deletePrevChainFiles = deletePrevChainFiles
        
        self.comm = comm
        
        # if 'MacbookPro' not in self.uname:
        #     self.changedirname = '../'
        # else:
        #     self.changedirname = '../mcmc_cc_xd'
        
        
        if rankmpi == 0:
            if deletePrevChainFiles == True:
                addsuffix = find_last_suffix(filenamepartToSearchFor='0_allparams_cc%s'%(parameterssavetxt),dirnameToSearchIn=os.path.join(mcmc_mainrun_dir_relpath,'chains/run%s'%(int(currentrunindex))))
                # self.addsuffix = addsuffix
            else:
                prevSuffix = find_last_suffix(filenamepartToSearchFor='0_allparams_cc%s'%(parameterssavetxt),dirnameToSearchIn=os.path.join(mcmc_mainrun_dir_relpath,'chains/run%s'%(int(currentrunindex))))
                xx = prevSuffix.split('_')[1]
                addsuffix = '_' + '%02d'%(int(xx)+1)
                # self.addsuffix = addsuffix
        else:
            addsuffix = None
        
        addsuffix = self.comm.bcast(addsuffix, root=0)
        
        self.addsuffix = addsuffix
        
        
        
        
        self.accepted_chains_file_name = os.path.join(mcmc_mainrun_dir_relpath, 
                                                      'chains',
                                                      'run%s'%(currentrunindex),
                                                      ##
                                                      '%s_allparams_cc'%(rankmpi) 
                                                      + parameterssavetxt 
                                                      + addsuffix 
                                                      + '%s.dat'%(testfilekw))
        
        self.truly_accepted_chains_file_name = os.path.join(mcmc_mainrun_dir_relpath,
                                                            'chains',
                                                            'run%s'%(currentrunindex),
                                                            'trulyAccepted',
                                                            ##
                                                            '%s_allparams_cc'%(rankmpi) 
                                                            + parameterssavetxt 
                                                            + addsuffix 
                                                            + '%s.dat'%(testfilekw))
        
        self.write_out_paramsTrulyAccepted = write_out_paramsTrulyAccepted
        
    
    # =============================================================================
    # WARNING: This function is not updated !!!
    # =============================================================================
    def remove_olderChainFiles_forThisRun(self):
        if self.deletePrevChainFiles == True:
            try:
                listofChainsNames = find_files_containing('_allparams_cc'
                                                          + self.parameterssavetxt 
                                                          + self.addsuffix 
                                                          + '%s.dat'%(self.testfilekw), 
                                                          ##
                                                          os.path.join(mcmc_mainrun_dir,
                                                                       'chains',
                                                                       'run%s'%(self.currentrunindex)))
                
                if len(listofChainsNames) != 0:
                    for filex in listofChainsNames:
                        os.remove(os.path.join(self.mcmc_mainrun_dir_relpath, 
                                               'chains',
                                               'run%s'%(self.currentrunindex),
                                               '%s'%(filex)))
                        
                    for filey in os.listdir(os.path.join(self.mcmc_mainrun_dir_relpath,
                                                         'chains',
                                                         'run%s'%(self.sysargs[1]),
                                                         'trulyAccepted')):
                        os.remove(os.path.join(self.mcmc_mainrun_dir_relpath, 
                                               'chains',
                                               'run%s'%(self.sysargs[1]),
                                               'trulyAccepted',
                                               '%s'%(filey)))
                print('Older files for this run removed'); sys.stdout.flush()
            except:
                print('No older files for this run'); sys.stdout.flush()
        else:
            print('NOTE: You have chosen to not delete the old chain files for this run (%s). If there exist any older files, then a suffix number will be added to the new files.'%(sys.argv[1])); sys.stdout.flush()
            
    
    
    def write_chains(self, stepindex, paramsAcceptedarg, paramsTrulyAcceptedarg):
        
        
        
        f1 = open(self.accepted_chains_file_name , 'a')
        f1.writelines('%s\t'*self.columnsInFile%(tuple(paramsAcceptedarg[:,stepindex]))+'\n')
        f1.close()
        
        if self.write_out_paramsTrulyAccepted:
            f2 = open(self.truly_accepted_chains_file_name , 'a')
            f2.writelines('%s\t'*self.columnsInFile%(tuple(paramsTrulyAcceptedarg[:,stepindex]))+'\n')
            f2.close()
        else:
            None






