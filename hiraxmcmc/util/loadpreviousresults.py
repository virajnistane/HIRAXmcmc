#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,sys
import numpy as np
import itertools

from hiraxmcmc.util.basicfunctions import *




# =============================================================================
# Load old MCMC results --> sample_cov and init_params
# =============================================================================

class LoadOlderResults:
    
    """
    The objective for this class is to include functions that can load the older/previous MCMC results (if any) in order to have some initial starting point and a thetacov matrix for the present run. 
    
    Case1: If a suffix is been added to the current run, then we would simple want the previous results with the highest suffix number because that would be (ideally) the run which is the most correct of them all. 
    
    """
    
    def __init__(self, currentrunindex, paramstovary, currentparams, ordered_params_list, 
                 priors, rankmpi, mcmc_mainrun_dir_relpath, 
                 override_filenamepart = None, testfilekw = ''):   # inputforhiraxoutput (not needed)
        self.p2v = paramstovary
        self.testfilekw = testfilekw
        self.rankmpi = rankmpi
        self.currentparams = currentparams
        self.priors = priors
        # self.inputforhiraxoutput = inputforhiraxoutput
        self.mcmc_mainrun_dir_relpath = mcmc_mainrun_dir_relpath
        
        self.ordered_params_list = ordered_params_list
        self.uname = os.uname()[1]
        
        if sys.argv != ['']:
            # self.changedirname = '../'
            self.currentrunindex = currentrunindex
            self.prevrunindex = int(self.currentrunindex - 1)
        elif sys.argv == ['']:
            # self.changedirname = '../mcmc_cc_xd'
            self.currentrunindex = 1         # *******
            self.prevrunindex = int(self.currentrunindex - 1)
            
        self.override_filenamepart = override_filenamepart
        
        comb_parametersvaried_prev = []
        
        for rr in range(len(ordered_params_list)+1):
            comb_obj = itertools.combinations(ordered_params_list, rr)
            comb_list = list(comb_obj)
            comb_parametersvaried_prev += comb_list
        
        #remove the empty tuple:
        comb_parametersvaried_prev.remove(())
        #remove the full length tuple:
        comb_parametersvaried_prev.remove(tuple(ordered_params_list))
        
        # remove the current params-to-vary tuple                            # WHY????
        comb_parametersvaried_prev.remove(tuple(paramstovary))
        
        self.comb_parametersvaried_prev = comb_parametersvaried_prev
        self.comb_parameterssavetxt_prev = ['_'+'_'.join(rr) for rr in comb_parametersvaried_prev]
        
        
        self.prevRunLastSuffix_forChains = find_last_suffix(filenamepartToSearchFor='0_allparams_cc',dirnameToSearchIn=os.path.join(mcmc_mainrun_dir_relpath,'chains/run%s'%(self.prevrunindex)))
        self.prevRunLastSuffix_forFinal = find_last_suffix(filenamepartToSearchFor='%s_paramsAcceptedFinal_cambcamb'%(self.prevrunindex),dirnameToSearchIn=mcmc_mainrun_dir_relpath, filetype='final_allparams')
    
    """ Make text combinations of params possibly used in previous run """
    # comb_parameterssavetxt_prev = []
    # for i1, j1 in enumerate(ordered_params_list):
    #     for i2, j2 in enumerate(ordered_params_list):
    #         if i2 > i1:
    #             comb_parameterssavetxt_prev.append('_' + j1 + '_' + j2)
    #             # comb_parametersvaried_prev['_' + j1 + '_' + j2] = [j1,j2]
    #             for i3, j3 in enumerate(ordered_params_list):
    #                 if i3 > i2 > i1:
    #                     comb_parameterssavetxt_prev.append('_' + j1 + '_' + j2 + '_' + j3)
    #                     # comb_parametersvaried_prev['_' + j1 + '_' + j2 + '_' + j3] = [j1,j2,j3]
    #                     for i4, j4 in enumerate(ordered_params_list):
    #                         if i4 > i3 > i2 > i1:
    #                             comb_parameterssavetxt_prev.append('_'+j1 + '_' + j2 + '_' + j3 + '_' + j4)
    #                             # comb_parametersvaried_prev['_' + j1 + '_' + j2 + '_' + j3 + '_' + j4] = [j1,j2,j3,j4]
    
    
    
    
    
    
    
    
    
    def check_parameterssavetxt_prev(self, filesavingstyle = 'allparams'):
        """
        Sets the text prompt <prev_combination_text> for finding files based on 
        the parameters varied in the last run. If the same parameters were varied in the previous index
        run, then the text prompt would be the same as the current run's 'params-
        to-vary' in text format. 
        
        This function also sets the variable <prev_params_varied>.

        Parameters
        ----------
        filesavingstyle : TYPE, optional
            DESCRIPTION. The default is 'allparams'.
        
        Returns
        -------
        None. Sets the variables <prev_params_varied> and <prev_combination_text> 
        within the class for further use

        """
        
        if os.path.exists(os.path.join(self.mcmc_mainrun_dir_relpath, 'chains', 'run%s'%(self.prevrunindex), '0_allparams_cc' + '_'+'_'.join(self.p2v) + self.prevRunLastSuffix_forChains + '%s.dat'%(self.testfilekw))):
            prev_combination_text = '_'+'_'.join(self.p2v)
            prev_params_varied = self.p2v
            if self.rankmpi == 0:
                print('same params combination as the current one found'); sys.stdout.flush()
        else:
            for comb in self.comb_parameterssavetxt_prev:
                if os.path.exists(os.path.join(self.mcmc_mainrun_dir_relpath, 'chains', 'run%s'%(self.prevrunindex), '0_allparams_cc' + comb + self.prevRunLastSuffix_forChains + '%s.dat'%(self.testfilekw))):
                # if os.path.exists(os.path.join(mcmc_mainrun_dir, '%s_paramsAcceptedFinal_cambcamb'%(self.prevrunindex)+ comb + '%s.dat'%(self.testfilekw))):
                    # loadallparamfile = np.loadtxt('%s_paramsAcceptedFinal_cambcamb'%(self.previndex)+ comb + '%s.dat'%(self.testfilekw))
                    if len(comb.split('_')[1:]) != len(self.p2v):
                        prev_params_varied = comb.split('_')[1:]
                        prev_combination_text = comb
                        if self.rankmpi == 0:
                            print("Previous set of params varied --> %s --> file found and can be loaded "%(prev_combination_text)); sys.stdout.flush()
                        break
                else:
                    prev_params_varied = self.ordered_params_list
                    prev_combination_text = ''
            
        self.prev_params_varied = prev_params_varied
        self.prev_combination_text = prev_combination_text
        
        
    
    def load_allparams_file_and_chains(self, totalParams_inclChi2, burnin_length_for_each_chain=0):
        """
        This function is to be used when all the final files (final output and 
        chains) for the previous run exist in the working directory.

        Parameters
        ----------
        totalParams_inclChi2 : TYPE
            DESCRIPTION.

        Raises
        ------
        IOError
            DESCRIPTION.

        Returns
        -------
        None. Sets thetacovauto and initial points.

        """
        
        
        if self.override_filenamepart != None:
            loadallparamfile = np.loadtxt(os.path.join(self.mcmc_mainrun_dir_relpath,
                                                       self.prevrunindex + '_paramsAcceptedFinal_cambcamb' + self.override_filenamepart))
        else:
            loadallparamfile = np.loadtxt(os.path.join(self.mcmc_mainrun_dir_relpath, 
                                                       '%s_paramsAcceptedFinal_cambcamb'%(self.prevrunindex) 
                                                       + self.prev_combination_text 
                                                       + self.prevRunLastSuffix_forFinal 
                                                       +'%s.dat'%(self.testfilekw)))
            
            if self.rankmpi == 0:
                print('Final file loaded: ', os.path.join(self.mcmc_mainrun_dir_relpath,
                                                          '%s_paramsAcceptedFinal_cambcamb'%(self.prevrunindex)
                                                          + self.prev_combination_text 
                                                          + self.prevRunLastSuffix_forFinal 
                                                          + '%s.dat'%(self.testfilekw)
                                                          )
                      ); sys.stdout.flush()
        
        prev_params_val = loadallparamfile[:,1:].T
        self.prev_params_val = prev_params_val
        # H0prev, omkprev, omlprev, w0prev, waprev = loadallparamfile[:,1:].T
        
        # print(prev_params_val)
        # self.thetacovauto = np.cov([H0prev, omkprev, omlprev, w0prev, waprev])
        self.thetacovauto = np.cov(prev_params_val)
        
        
        try:
            ###### from corresponding chains
            
            # 1. Make a list of chain files names in the directory corresponding to the file loaded above.
            if self.override_filenamepart != None:
                listofChainsNames = find_files_containing('_allparams_cc' + self.override_filenamepart + '%s.dat'%(self.testfilekw), 
                                                          os.path.join(self.mcmc_mainrun_dir_relpath,'chains/run%s'%(self.prevrunindex)))
            else:
                listofChainsNames = find_files_containing('_allparams_cc' + self.prev_combination_text + self.prevRunLastSuffix_forChains + '%s.dat'%(self.testfilekw), 
                                                          os.path.join(self.mcmc_mainrun_dir_relpath, 'chains/run%s'%(self.prevrunindex)))
                
                if self.rankmpi == 0:
                    print('Chain files to be loaded: ', 
                          '_allparams_cc' + self.prev_combination_text + self.prevRunLastSuffix_forChains + '%s.dat'%(self.testfilekw),
                          ' <-- FROM -- ', 
                          'chains/run%s'%(self.prevrunindex)
                          ); sys.stdout.flush()
        
            
            
            # 2. Find the number of such files present.
            nchains = len(listofChainsNames)
            
            # 3. Store the sizes of all the chains in an array
            listOfAllChainsLengths = np.array([])
            for chainname in listofChainsNames:
                listOfAllChainsLengths = np.append(listOfAllChainsLengths, 
                                                   len(np.loadtxt(
                                                       os.path.join(
                                                           self.mcmc_mainrun_dir_relpath, 
                                                           'chains/run%s'%(self.prevrunindex),chainname))[:,0]
                                                       )
                                                   )
            
            # 4. Find the chain with the smallest size
            shortestChainLength = int(np.min(listOfAllChainsLengths))
            
            if self.rankmpi == 0:
                print('Burn in region selected: %s for files loaded for previous run %s'%(burnin_length_for_each_chain, self.prevrunindex))
            
            
            # 5. Make all the chains of the same size as the shortest chain and 'Gather' them into a 2d/3d array of size <nchains>
            
            chains = np.zeros((nchains, totalParams_inclChi2, int(shortestChainLength - burnin_length_for_each_chain)))
            for chainNumberIndex, chainname in enumerate(listofChainsNames):
                chains[chainNumberIndex] = np.loadtxt(os.path.join(self.mcmc_mainrun_dir_relpath, 
                                                                   'chains/run%s'%(self.prevrunindex), 
                                                                   chainname)).T[:,int(burnin_length_for_each_chain):int(shortestChainLength)]
            
            # 5.A Find thetacovauto if burnin_length_for_each_chain != 0
            
            # self.chains = chains
            
            if self.prevrunindex == 1:# and burnin_length_for_each_chain != 0:
                # H0prev = chains[:,1,:].flatten()
                # omkprev = chains[:,2,:].flatten()
                # omlprev = chains[:,3,:].flatten()
                # w0prev = chains[:,4,:].flatten()
                # waprev = chains[:,5,:].flatten()
                prevparamsval_flatten_BIrem = []
                for i in np.arange(1, totalParams_inclChi2):
                    prevparamsval_flatten_BIrem.append(chains[:,i,:].flatten())
                
                # self.thetacovauto = np.cov([H0prev, omkprev, omlprev, w0prev, waprev])
                self.thetacovauto = np.cov(np.array(prevparamsval_flatten_BIrem))
            
            # 6. Fine the last element in each chain to have the initial set of params for a chosen rankmpi
            
            for pi, pv in enumerate(self.p2v):
                self.currentparams[pv] = chains[self.rankmpi, int(pi+1), -1]
            
            
            if self.rankmpi == 0:
                print("Initial parameters chosen from the previous chain (FOR THE SAME PARAMETERS AS HERE)"); sys.stdout.flush()
            
            
        except:
            
            # raise IOError("No final all_params file")
            ###### randomly
            
            for index1, paramname1 in enumerate(self.p2v):
                self.currentparams[paramname1] = np.random.uniform(np.mean(prev_params_val[index1]) - np.std(prev_params_val[index1]) , 
                                                              np.mean(prev_params_val[index1]) + np.std(prev_params_val[index1]))
                
                
            # self.currentparams['H0'] = np.random.uniform( np.mean(H0prev) - np.std(H0prev) , np.mean(H0prev) + np.std(H0prev))
            # self.currentparams['Omk'] = np.random.uniform( np.mean(omkprev) - np.std(omkprev) , np.mean(omkprev) + np.std(omkprev))
            # self.currentparams['Oml'] = np.random.uniform( np.mean(omlprev) - np.std(omlprev) , np.mean(omlprev) + np.std(omlprev))
            # self.currentparams['w0'] = np.random.uniform( np.mean(w0prev) - np.std(w0prev) , np.mean(w0prev) + np.std(w0prev))
            # self.currentparams['wa'] = np.random.uniform( np.mean(waprev) - np.std(waprev) , np.mean(waprev) + np.std(waprev))
            
            if self.rankmpi == 0:
                print("Initial parameters chosen randomly"); sys.stdout.flush()
        
            
    def load_allParams_chains_only(self, totalParams_inclChi2):
        """
        This function is to be used when only chains for the previous run exist 
        in the working directory.

        Parameters
        ----------
        totalParams_inclChi2 : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.override_filenamepart != None:
            listofChainsNames = find_files_containing('_allparams_cc' + self.override_filenamepart + '%s.dat'%(self.testfilekw), 
                                                      os.path.join(self.mcmc_mainrun_dir_relpath, 'chains/run%s'%(self.prevrunindex)))
        else:
            listofChainsNames = find_files_containing('_allparams_cc' + self.prev_combination_text + self.prevRunLastSuffix_forChains + '%s.dat'%(self.testfilekw), 
                                                      os.path.join(self.mcmc_mainrun_dir_relpath, 'chains/run%s'%(self.prevrunindex)))
            
            if self.rankmpi == 0:
                print('Chain files to be loaded: ', 
                      '_allparams_cc' + self.prev_combination_text + self.prevRunLastSuffix_forChains + '%s.dat'%(self.testfilekw),
                      ' <-- FROM -- ', 
                      'chains/run%s'%(self.prevrunindex)
                      ); sys.stdout.flush()
            
        nchains = np.size(listofChainsNames)
        
        lengthsListOfAllChains = np.array([])
        for chainname in listofChainsNames:
            lengthsListOfAllChains = np.append(lengthsListOfAllChains, 
                                               len(np.loadtxt(
                                                   os.path.join(self.mcmc_mainrun_dir_relpath, 
                                                                'chains/run%s'%(self.prevrunindex), 
                                                                chainname))[:,0]
                                                   )
                                               )
        
        shortestChainLength = int(np.min(lengthsListOfAllChains))
        
        if self.prevrunindex == 1:
            burnin_length_for_each_chain = int(shortestChainLength/2)
        else:
            burnin_length_for_each_chain = 0
        
        
        if self.rankmpi == 0:
            print('Burn in region selected: %s for files loaded for previous run %s'%(burnin_length_for_each_chain, self.prevrunindex))
            
        
        chains = np.zeros((nchains, totalParams_inclChi2, int(shortestChainLength - burnin_length_for_each_chain)))
        
        for chainNumberIndex, chainname in enumerate(listofChainsNames):    
            chains[chainNumberIndex] = np.loadtxt(os.path.join(self.mcmc_mainrun_dir_relpath, 
                                                               'chains/run%s'%(self.prevrunindex),
                                                               chainname)).T[:,int(burnin_length_for_each_chain):int(shortestChainLength)]
        
        
        prevparamsval_flatten_BIrem = []
        for i in np.arange(1, totalParams_inclChi2):
            prevparamsval_flatten_BIrem.append(chains[:,i,:].flatten())
        
        self.thetacovauto = np.cov(np.array(prevparamsval_flatten_BIrem))
        
        # H0prev = chains[:,1,:].flatten()
        # omkprev = chains[:,2,:].flatten()
        # omlprev = chains[:,3,:].flatten()
        # w0prev = chains[:,4,:].flatten()
        # waprev = chains[:,5,:].flatten()
        
        # self.thetacovauto = np.cov([H0prev, omkprev, omlprev, w0prev, waprev])
        
        for pi, pv in enumerate(self.p2v):
            self.currentparams[pv] = chains[self.rankmpi, int(pi+1), -1]
        
        
        if self.rankmpi == 0:
            print("Final data file ISN'T available for the previous run, so loading both thetacov and initial params from the previous chains"); sys.stdout.flush()
        
    
    def firstrunparams(self,thetacov0):
        """
        This function is to be used when the current run index is 1. 

        Returns
        -------
        None.

        """
        
        if self.rankmpi == 0:
            print("It seems to be the first run, thus using a random initial proposal covariance matrix"); sys.stdout.flush()
        
        # thetacov0 = np.diag([0.5,5e-6,5e-6,5e-6,5e-6])
        
        self.thetacovauto = thetacov0
        # self.currentparams['H0'] = np.random.uniform( 50, 85 )
        # self.currentparams['Omk'] = np.random.uniform(  -0.1 , 0.1 )
        # self.currentparams['Oml'] = np.random.uniform(  0.5, 0.88 )
        # self.currentparams['w0'] = np.random.uniform( -2, 0 )
        # self.currentparams['wa'] = np.random.uniform( -4, 4  )
        
        for paramname in self.p2v:
            self.currentparams[paramname] = np.random.uniform(self.priors[paramname][0], self.priors[paramname][1])
        
        
        
        
        
        
        
        
        
        
        
        
   
