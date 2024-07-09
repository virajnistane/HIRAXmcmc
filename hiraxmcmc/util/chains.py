#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,sys
from hiraxmcmc.util.basicfunctions import *


# =============================================================================
# CHAINS - saving and removing old files
# =============================================================================


class Chains:
    
    def __init__(self, currentrunindex, totalParams_inclChi2, rankmpi, comm, testfilekw, parameterssavetxt, mcmc_mainrun_dir_relpath, addsuffix_fromLoadOldRes, write_out_paramsTrulyAccepted = False, deletePrevChainFiles=False):
        
        self.columnsInFile = totalParams_inclChi2  # number of columns in the file
        self.currentrunindex = currentrunindex # current run index
        
        self.testfilekw = testfilekw # keyword for the test file
        self.parameterssavetxt = parameterssavetxt # keyword for the parameters file
        
        self.uname = os.uname()[1]  # name of the computer
        self.mcmc_mainrun_dir_relpath = mcmc_mainrun_dir_relpath # relative path to the main run directory
        
        self.deletePrevChainFiles = deletePrevChainFiles # delete previous chain files - boolean
        
        self.comm = comm # MPI communicator
        


        if rankmpi == 0: 
            if deletePrevChainFiles == True:
                # Description: If the user has chosen to delete the previous chain files, then no suffix number will be added to the new files.
                addsuffix = find_last_suffix(filenamepartToSearchFor='0_allparams_cc%s'%(parameterssavetxt),dirnameToSearchIn=os.path.join(mcmc_mainrun_dir_relpath,'chains/run%s'%(int(currentrunindex)))) # find the last suffix for the chain files
                # self.addsuffix = addsuffix
            else:
                # Description: If the user has chosen to not delete the previous chain files, then a suffix number will be added to the new files.
                if addsuffix_fromLoadOldRes == None:
                    prevSuffix = find_last_suffix(filenamepartToSearchFor='0_allparams_cc%s'%(parameterssavetxt),dirnameToSearchIn=os.path.join(mcmc_mainrun_dir_relpath,'chains/run%s'%(int(currentrunindex))))
                    xx = prevSuffix.split('_')[1]
                    addsuffix = '_' + '%02d'%(int(xx)+1)
                    # self.addsuffix = addsuffix
                elif addsuffix_fromLoadOldRes != None:
                    addsuffix = addsuffix_fromLoadOldRes
        else:
            addsuffix = None
        
        addsuffix = self.comm.bcast(addsuffix, root=0)
        
        self.addsuffix = addsuffix
        
        
        
        # Description: The chain files are saved in the chains directory in the main run directory.
        self.accepted_chains_file_name = os.path.join(mcmc_mainrun_dir_relpath, 
                                                      'chains',
                                                      'run%s'%(currentrunindex),
                                                      ##
                                                      '%s_allparams_cc'%(rankmpi) 
                                                      + parameterssavetxt 
                                                      + addsuffix 
                                                      + '%s.dat'%(testfilekw))
        
        # Description: The truly accepted chain files are saved in the trulyAccepted directory in the chains directory in the main run directory.
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






