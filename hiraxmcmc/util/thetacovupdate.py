#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,sys
import numpy as np
from scipy.interpolate import interp1d
import random


from hiraxmcmc.util.basicfunctions import *
from hiraxmcmc.util.loadpreviousresults import LoadOlderResults



# =============================================================================
# Thetacov update function
# =============================================================================


larr = np.arange(0,1,0.01)

# accRate_vs_TRF_default = np.array([[0.        , 0.5       ],
#                                    [0.05      , 0.45      ],
#                                    [0.08      , 0.4       ],
#                                    [0.1       , 0.35      ],
#                                    [0.15      , 0.30      ],    
#                                    [0.2       , 0.28      ],
#                                    [0.3       , 0.25      ],      
#                                    [0.5       , 0.20      ],
#                                    [0.6       , 0.15      ],
#                                    [0.7       , 0.10      ],
#                                    [0.8       , 0.09      ],
#                                    [0.9       , 0.085     ],
#                                    [1.        , 0.08      ]])


# accRate_vs_TRF_default = np.array([[0.        , 0.010       ],
#                                    [0.05      , 0.025       ],
#                                    [0.08      , 0.060       ],
#                                    [0.1       , 0.090       ],
#                                    [0.15      , 0.120       ],
#                                    [0.2       , 0.250       ],
#                                    [0.3       , 0.500       ],      
#                                    [0.5       , 0.700       ],
#                                    [0.6       , 1.000       ],
#                                    [0.7       , 2.000       ],
#                                    [0.8       , 2.500       ],
#                                    [0.9       , 4.800       ],
#                                    [1.        , 5.000       ]])

# accRate_vs_TRF_default = np.array([larr, 1/500+1/500*np.exp(7.2*larr)]).T


accRate_vs_TRF_default = np.array([[0.        , 0.2     ],
                                   [0.1       , 0.35    ],
                                   [0.2       , 0.4     ],
                                   [0.3       , 0.5     ],
                                   [0.4       , 0.9     ],
                                   [0.5       , 1.5     ],
                                   [0.6       , 2.0     ],
                                   [0.7       , 2.5     ],
                                   [0.8       , 5.0     ],
                                   [0.9       , 10.0    ],
                                   [1.        , 20.0    ]])


class ThetaCovUpdate:
    
    def __init__(self, INPUT, comm, rankmpi, sizempi, 
                 currentparams, chaininstance, mcmc_mainrun_dir_relpath,   # inputforhiraxoutput
                 accRate_vs_TRF = accRate_vs_TRF_default, testfilekw = ''):
        
        
        self.do_update_thetacov = INPUT['mcmc']['do_update_thetacov']
        self.thetacovold_until = INPUT['mcmc']['thetacovold_until']
        self.TRFold_until = INPUT['mcmc']['TRFold_until']
        self.dothetacovupdateafterevery = INPUT['mcmc']['dothetacovupdateafterevery']
        
        self.ordered_params_list = list(INPUT['PARAMS'].keys())
        self.params_to_vary = params_to_vary_list_from_input(INPUT['params_to_vary'], self.ordered_params_list)
        self.priors = {}
        for pp in self.params_to_vary:
            self.priors[pp] = INPUT['PARAMS'][pp]['prior']
            
            
        # self.inputforhiraxoutput = inputforhiraxoutput
        
        self.comm = comm
        self.rankmpi = rankmpi
        self.sizempi = sizempi
        
        
        
        self.totalParams = len(self.params_to_vary)
        # self.totalParams_inclChi2 = int(totalParams+1)
        
        
        self.currentrunindex = INPUT['current_run_index']
        self.mcmc_mainrun_dir_relpath = mcmc_mainrun_dir_relpath
        
        self.accRate_vs_TRF = accRate_vs_TRF
        self.testfilekw = testfilekw
        
        self.uname = os.uname()[1]
        
        self.parameterssavetxt = '_'+'_'.join(self.params_to_vary)
        
        # if 'MacbookPro' not in self.uname:
        #     self.changedirname = '../'
        # else:
        #     self.changedirname = '../mcmc_cc_xd/'
        
        
        load_older_results = LoadOlderResults(self.currentrunindex, self.params_to_vary, currentparams, self.ordered_params_list, self.priors, rankmpi, mcmc_mainrun_dir_relpath) #inputforhiraxoutput
        load_older_results.check_parameterssavetxt_prev()
        self.prev_combination_text = load_older_results.prev_combination_text
        
        
        if int(self.currentrunindex) != 1:
            self.last_suffix = find_last_suffix('_TRF%s'%(self.prev_combination_text), 
                                                os.path.join(mcmc_mainrun_dir_relpath,'TRF%s/run%s'%(testfilekw,int(int(self.currentrunindex)-1))), 
                                                filetype='TRF')
        else:
            self.last_suffix = None
            
        
        # if self.last_suffix != None:
        #     self.addsuffix = '_' + '%02d'%(int(self.last_suffix.split('_')[1])+1)
        # else:
        #     self.addsuffix = '_00'
        self.addsuffix = chaininstance.addsuffix

        
            
        
        
        if int(self.currentrunindex) == 1:
            self.thetacov_reductionfactor_initial = 20
        else:
            self.thetacov_reductionfactor_initial = np.load(os.path.join(mcmc_mainrun_dir_relpath, 'TRF%s/run%s/rank%s_TRF%s%s.npy'
                                                                         %(self.testfilekw,
                                                                           int(int(self.currentrunindex)-1), 
                                                                           self.rankmpi,
                                                                           self.prev_combination_text, 
                                                                           self.last_suffix))).item()
            
        
# check this before next run
        if int(self.currentrunindex) > 2:
            self.acceptance_rate = np.load(os.path.join(mcmc_mainrun_dir_relpath, 'AR%s/run%s/rank%s_AR%s%s.npy'
                                                             %(self.testfilekw, 
                                                               int(int(self.currentrunindex)-1), 
                                                               self.rankmpi, 
                                                               self.prev_combination_text, 
                                                               self.last_suffix))).item()
        else:
            self.acceptance_rate = interp1d(accRate_vs_TRF_default[:,1],accRate_vs_TRF_default[:,0])(1).item()
        
        
        if self.do_update_thetacov in ['yes','true','1']:
            self.thetacovUpdate = self.thetacovYesUpdate
        elif self.do_update_thetacov in ['no','false','0']:
            self.thetacovUpdate = self.thetacovNoUpdate
    
        def covfuninput(onlyParamsAccepted_together_input, 
                        lasthowmany_for_cov=None, 
                        startingfrom_for_cov=None):
            if lasthowmany_for_cov == None and startingfrom_for_cov == None:
                lasthowmany_for_cov = onlyParamsAccepted_together_input.shape[-1]
            
                
            if startingfrom_for_cov != None:
                lasthowmany_for_cov = int(onlyParamsAccepted_together_input.shape[-1] - startingfrom_for_cov)
                covfuninputarr = np.zeros((onlyParamsAccepted_together_input.shape[1], 
                                           int(onlyParamsAccepted_together_input.shape[0]*lasthowmany_for_cov)))
                for i in np.arange(np.shape(onlyParamsAccepted_together_input)[1]):
                    covfuninputarr[i] = onlyParamsAccepted_together_input[:,i,startingfrom_for_cov:].flatten()
                return covfuninputarr
            else:
                covfuninputarr = np.zeros((onlyParamsAccepted_together_input.shape[1], 
                                           int(onlyParamsAccepted_together_input.shape[0]*lasthowmany_for_cov)))
                for i in np.arange(np.shape(onlyParamsAccepted_together_input)[1]):
                    covfuninputarr[i] = onlyParamsAccepted_together_input[:,i,-lasthowmany_for_cov:].flatten()
                return covfuninputarr
            
        self.covfuninput = covfuninput
        
        def AR_TRF(currentstep_ii, 
                   acceptance_rate, previous_accept_rate, 
                   thetacov_reductionfactor_old):
            tck = interp1d(self.accRate_vs_TRF[:,0],self.accRate_vs_TRF[:,1])
            
            ar = random.choices([acceptance_rate, np.random.uniform(acceptance_rate,previous_accept_rate)], weights=[0.5,0.5])[0]
            
            new_TRF = tck(np.array([ar]))
            
            # if int(self.currentrunindex) < 10:
            #     if new_TRF >= thetacov_reductionfactor:
            #         return new_TRF
            #     else:
            #         return thetacov_reductionfactor
            if int(self.currentrunindex) == 1 and currentstep_ii<int(self.TRFold_until):   #4000
                return [thetacov_reductionfactor_old]
            else:
                return new_TRF
        
        self.AR_TRF = AR_TRF
        
    def thetacovYesUpdate(self, thetacovold, thetacov_reductionfactor, currentstep, paramsAccepted_excdChi2_CurrentFullArray, paramsTrulyAccepted_excdChi2_CurrentFullArray):
        """
        
        Parameters
        ----------
        thetacovold : array
            The thetacov being used until the point of usage of this function. 
        thetacov_reductionfactor : float
            The thetacov_reductionfactor being used until the point of usage of this function.
        currentstep : integer
            The current iteration/step of the for-loop.
        allAcceptedCurrentFullArray : array
            paramsAccepted. Array of size (totalParams_inclChi2 x total_iterations) 
            Eg.: (6 x 10000) array where the rows correspond to (chi2, H0, Omk, Oml, w0, wa) in that order
        H0trulyAcceptedCurrentFullArray : array
            The column corresponding to the param 'H0' of the paramsTrulyAccepted. Array of size (1x10000)
            Any other parameters also works: Eg.: paramsTrulyAccepted[4]
        
        Returns
        -------
        
        thetacov, thetacov_reductionfactor
        array, float
            The updated thetacov according to the sample covariance and the new thetacov_reductionfactor.
            The updated thetacov_reductionfactor according to the acceptance rate
            
        """
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        
        nstepstotal = paramsAccepted_excdChi2_CurrentFullArray.shape[-1]
        
        # nonzeroIndices_onlyParamsAccepted = np.where(np.ma.masked_not_equal(allAcceptedUntilNow[1],0).mask)[0]  
        FilledIndices_onlyParamsAccepted = np.arange(1, int(currentstep+1))
        onlyParamsAccepted = paramsAccepted_excdChi2_CurrentFullArray[:,FilledIndices_onlyParamsAccepted]   # exclude chi2, and the zero elements not yet filled with accepted values of params
        
        # FilledIndices_H0trulyaccepted = np.where(np.ma.masked_not_equal(H0trulyAcceptedUntilNow,0).mask)[0]
        FilledIndices_firstparam_trulyaccepted = np.arange(1, int(currentstep+1))
        firstparam_trulyaccepted = paramsTrulyAccepted_excdChi2_CurrentFullArray[0, FilledIndices_firstparam_trulyaccepted]
        
        # ^^^ basically cuts out the zero initialized part of the arrays ^^^
        
        if int(self.currentrunindex) == 1 and currentstep<int(self.thetacovold_until/2): #4000
            lastHowManyValues_forAcceptanceRate = paramsAccepted_excdChi2_CurrentFullArray.shape[1]
        else:
            lastHowManyValues_forAcceptanceRate = 500 #allAcceptedCurrentFullArray.shape[1]
        
        
        
        acceptance_rate = len(firstparam_trulyaccepted[-lastHowManyValues_forAcceptanceRate:]
                              [np.where(np.ma.masked_not_equal(firstparam_trulyaccepted[-lastHowManyValues_forAcceptanceRate:],0).mask)[0]]
                              )/len(onlyParamsAccepted[0][-lastHowManyValues_forAcceptanceRate:])
        
        
        
        
        
        print("\n rank %s is at ii= %s\n"%(self.rankmpi, currentstep)); sys.stdout.flush()
        
        ######################################################################
        ######################################################################
        
        """
        Method 1 using comm.Gather
        """
        
        onlyParamsAccepted_part_Reshaped = onlyParamsAccepted.reshape(int(self.totalParams*np.shape(onlyParamsAccepted)[-1]))
        
        onlyParamsAccepted_together_Reshaped = None
        
        if self.rankmpi == 0:
            onlyParamsAccepted_together_Reshaped = np.empty([self.sizempi,len(onlyParamsAccepted_part_Reshaped)])
            
        self.comm.Gather(onlyParamsAccepted_part_Reshaped, onlyParamsAccepted_together_Reshaped, root=0)
        
        if self.rankmpi == 0:
            onlyParamsAccepted_together = onlyParamsAccepted_together_Reshaped.reshape(self.sizempi, self.totalParams, np.shape(onlyParamsAccepted)[-1])
            # print("onlyParamsAccepted_part: \n%s"%(onlyParamsAccepted_part))
        
        
        """
        Method 2 using comm.Reduce 
        """
        
        # onlyParamsAcceptedTemp_part = np.zeros([self.sizempi, self.totalParams, np.shape(onlyParamsAccepted)[-1] ])
        # onlyParamsAcceptedTemp_part[self.rankmpi] = onlyParamsAccepted
        
        # onlyParamsAccepted_part = np.zeros_like(onlyParamsAcceptedTemp_part)
        
        
        # self.comm.Barrier()
        
        # if self.rankmpi == 0:
        #     print('sync achieved, transferring data to root'); sys.stdout.flush()
        
        # self.comm.Reduce(onlyParamsAcceptedTemp_part, onlyParamsAccepted_part, op=MPI.SUM, root=0)
        
        
        ######################################################################
        ######################################################################
        
        if self.rankmpi == 0:
            print("\n Last 5 H0 values gathered:\n %s\n"%(onlyParamsAccepted_together[:,0,-5:]))
        
        previous_accept_rate = self.acceptance_rate
        self.acceptance_rate = acceptance_rate
        
        
        
        
        print("\nAcceptance rate until now at rank %s = %s = %.1f%%"%(self.rankmpi,acceptance_rate,acceptance_rate*100)); sys.stdout.flush()
        thetacov_reductionfactor = self.AR_TRF(currentstep, acceptance_rate, previous_accept_rate, thetacov_reductionfactor)[0]
        print("thetacov_reductionfactor updated to: ",thetacov_reductionfactor); sys.stdout.flush()
        
        
        
        
        if self.rankmpi == 0:
            # if it is the first run and the currentstep is less than 4000, thetacov does not change
            if int(self.currentrunindex) == 1 and currentstep<int(self.thetacovold_until):
                thetacov = thetacovold
                print('I am here')
            # if it is the first run and the currentstep is greater than 4000,
            # OR
            # if it is not the first run, 
            else:
                # thetacov changes after every 1000 steps
                if np.mod(currentstep,1000) == 0:
                    if int(self.currentrunindex==1):
                    # if it is the first run and the currentstep is greater than 4000, 
                    # thetacov sample is calculated as below (not over all the previous samples)
                        
                        # if currentstep <= 4001:
                        #     thetacov_part = np.cov(self.covfuninput(onlyParamsAccepted_together, lasthowmany_for_cov=999))
                        # elif 4002 <= currentstep <= 6001:
                        #     thetacov_part = np.cov(self.covfuninput(onlyParamsAccepted_together, lasthowmany_for_cov=2000))
                        # else: 
                        #     thetacov_part = np.cov(self.covfuninput(onlyParamsAccepted_together, startingfrom_for_cov=6000))
                            
                        if currentstep <= int(self.thetacovold_until + 1):
                            thetacov_part = np.cov(self.covfuninput(onlyParamsAccepted_together, lasthowmany_for_cov=999))
                        elif int(self.thetacovold_until  + 2) <= currentstep <= int(self.thetacovold_until*3/2 + 1):
                            thetacov_part = np.cov(self.covfuninput(onlyParamsAccepted_together, lasthowmany_for_cov=2000))
                        else: 
                            thetacov_part = np.cov(self.covfuninput(onlyParamsAccepted_together, startingfrom_for_cov=int(self.thetacovold_until*5/4+1)))
                    else:
                    # if it is not the first run, thetacov sample is calculated 
                    # over the entire previous sample
                        thetacov_part = np.cov(self.covfuninput(onlyParamsAccepted_together))
                    
                    thetacov = thetacov_part * thetacov_reductionfactor
                    
                    
                    print("Updated thetacov: \n", thetacov,"\n"); sys.stdout.flush()
                else:
                    thetacov = thetacovold
                    
        else:
            thetacov = np.empty((len(self.params_to_vary),len(self.params_to_vary)))
        
        # thetacov = newcovmatrix(thetacov, self.params_to_vary, self.ordered_params_list)
        
        
        self.comm.Bcast(thetacov, root=0)
        
        return thetacov, thetacov_reductionfactor
    
    
    def thetacovNoUpdate(self, thetacovold, thetacov_reductionfactor, currentstep, allAcceptedCurrentFullArray, H0trulyAcceptedCurrentFullArray):
        
        return thetacovold, thetacov_reductionfactor
    
    
    # def updateOrnoupdate(self):
    #     if self.DoUpdateThetacov in ['yes','true','1']:
    #         self.thetacovUpdate = self.thetacovYesUpdate
    #     elif self.DoUpdateThetacov in ['no','false','0']:
    #         self.thetacovUpdate = self.thetacovNoUpdate
    
    def saveAR(self,acceptance_rate):
        np.save(os.path.join(self.mcmc_mainrun_dir_relpath,'AR%s/run%s/rank%s_AR%s%s'%(self.testfilekw,self.currentrunindex,self.rankmpi, self.parameterssavetxt, self.addsuffix)),acceptance_rate)
        
    def saveTRF(self,thetacov_reductionfactor, fcname=None):
        try: 
            assert fcname == None
            fcname2 = ''
        except: 
            fcname != None
            fcname2 = '_'+fcname
        np.save(os.path.join(self.mcmc_mainrun_dir_relpath,'TRF%s/run%s/rank%s%s_TRF%s%s'%(self.testfilekw,self.currentrunindex,self.rankmpi, fcname2, self.parameterssavetxt, self.addsuffix)),thetacov_reductionfactor)
    
    
