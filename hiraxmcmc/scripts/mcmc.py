#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HIRAX - MCMC 


If you want to output ps2d, modify the following lines:
    1. return values in "chi2" function
    2. setting chi2old from the initial parameters (line 827 - can change) in "Thetacov production and initial points selection"

"""


import sys,os
import numpy as np
import numpy.linalg as la
from numpy.random import multivariate_normal as mvn
from scipy.constants import speed_of_light as cc

from mpi4py import MPI
import time
from datetime import date


import json

""" Personal modules """


import hiraxmcmc
from hiraxmcmc.util.basicfunctions import *
from hiraxmcmc.core.hiraxoutput import HiraxOutput
from hiraxmcmc.util.cosmoparamvalues import ParametersFixed
from hiraxmcmc.core.chi2function import Chi2Func
from hiraxmcmc.util.loadpreviousresults import LoadOlderResults
from hiraxmcmc.util.directorymanagement import DirManage
from hiraxmcmc.util.thetacovupdate import ThetaCovUpdate
from hiraxmcmc.util.chains import Chains
## from extractforML import extractForML


""" CAMB """

# from camb.sources import GaussianSourceWindow, SplinedSourceWindow
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower

""" CLASS """

from classy import Class


# =============================================================================
# Initialize MPI
# =============================================================================




comm     = MPI.COMM_WORLD
size_mpi = comm.Get_size()
rank_mpi = comm.Get_rank()



if rank_mpi == 0:
    print('MPI successfully initialised, number of processes: '+str(size_mpi))
    sys.stdout.flush()
    
    
    

# =============================================================================
# IIIIIIIIIII    N      N    PPPPPPP     U      U   TTTTTTTTTTT
#      I         NN     N    P      P    U      U        T
#      I         N N    N    P      P    U      U        T
#      I         N  N   N    P      P    U      U        T
#      I         N   N  N    PPPPPPP     U      U        T
#      I         N    N N    P           U      U        T
#      I         N     NN    P           U      U        T
# IIIIIIIIIIII   N      N    P             UUUU          T
# =============================================================================

input_override = False

try:
    assert not(input_override)
    
    inputparamsload_filename = os.path.basename(sys.argv[1])  #'input.json'
    
    # if rank_mpi == 0:
    #     print(find_file(inputparamsload_filename, os.path.join(os.getcwd())))
    
    try:
        inputparamsload_file_relpath = sys.argv[1]     # in case full path to the input file is given as an argument
    except:
        inputparamsload_file_relpath = os.path.relpath(find_file(inputparamsload_filename, os.path.join(os.getcwd())))
    
    
    with open(inputparamsload_file_relpath, 'r') as inputloadingfile:
        INPUT = json.load(inputloadingfile)
    
except:
    
    assert input_override
    # False:-> input is taken from json file
    # True:-> input is taken from below
    
    
    if input_override:
        
        # INPUT = {'current_run_index': 1,
        #           'params_to_vary': ['qpar(z)', 'qperp(z)', 'f(z)'],
        #           'mmode_output': {'freq_channel': {'start': 400, 'end': 500},
        #                             'klmode': 'kl_5thresh_nofg',
        #                             'power_spectrum_estimator_type': 'minvar'},
        #           'mcmc': {'nsteps': 100000,
        #                   'do_update_thetacov': 'yes',
        #                   'dothetacovupdateafterevery': 100,
        #                   'thetacovold_until': 10000,
        #                   'TRFold_until': 16000,
        #                   'thetacov0': {'do_override': 'no',
        #                                 'manual_input_variance': {'H0': 25,
        #                                                           'Omk': 0.002,
        #                                                           'Oml': 0.01,
        #                                                           'w0': 0.2,
        #                                                           'wa': 1.5}}},
        #           'likelihood': {'PS_cov': {'override': 'no', 
        #                                     'filename_fullpath': ''}},
        #           'PARAMS': {'H0': {'prior': [50, 90]},
        #                       'h': {'prior': [0.5, 0.9]},
        #                       'Omk': {'prior': [-0.2, 0.2]},
        #                       'Oml': {'prior': [0.5, 0.9]},
        #                       'w0': {'prior': [-3, 1]},
        #                       'wa': {'prior': [-5, 5]},
        #                       'qpar(z)': {'prior': [0.3, 1.7], 'freqdep': True},
        #                       'h(z)': {'prior': [0.0001, 0.001], 'freqdep': True},
        #                       'qperp(z)': {'prior': [0.3, 1.7], 'freqdep': True},
        #                       'dA(z)': {'prior': [800, 3000], 'freqdep': True},
        #                       'f(z)': {'prior': [0.2, 1.2], 'freqdep': True}}}
        
        INPUT = {'current_run_index': 1,
                  'params_to_vary': ['h', 'Omk', 'Oml'],
                  'mmode_output': {'freq_channel': {'start': 400, 'end': 800},
                                  'klmode': 'dk_5thresh_fg_1000thresh',
                                  'power_spectrum_estimator_type': 'minvar'},
                  'mcmc': {'nsteps': 20000,
                          'do_update_thetacov': 'yes',
                          'dothetacovupdateafterevery': 100,
                          'thetacovold_until': 4000,
                          'TRFold_until': 6000,
                          'thetacov0': {'do_override': 'yes',
                                        'manual_input_variance': {'H0': 25,
                                                                  'Omk': 0.002,
                                                                  'Oml': 0.01,
                                                                  'w0': 0.2,
                                                                  'wa': 1.5}}},
                  'likelihood': {'PS_cov': {'override': 'no', 
                                            'filename_fullpath': ''}},
                  'PARAMS': {'H0': {'prior': [50, 90]},
                              'h': {'prior': [0.5, 0.9]},
                              'Omk': {'prior': [-0.2, 0.2]},
                              'Oml': {'prior': [0.5, 0.9]},
                              'w0': {'prior': [-3, 1]},
                              'wa': {'prior': [-5, 5]},
                              'qpar(z)': {'prior': [0.3, 1.7], 'freqdep': True},
                              'h(z)': {'prior': [0.0001, 0.001], 'freqdep': True},
                              'qperp(z)': {'prior': [0.3, 1.7], 'freqdep': True},
                              'dA(z)': {'prior': [800, 3000], 'freqdep': True},
                              'f(z)': {'prior': [0.2, 1.2], 'freqdep': True}}}
        
        
        
        # (optional, change to True if you want to save the input params to a file)
        if 0:
            with open('../inputfiles/input_example_fcdep.json','w') as f:
                json.dump(INPUT, f, indent=4)
            raise ValueError
        
        if 0:
            with open('../inputfiles/input_example_cosmo.json','w') as f:
                json.dump(INPUT, f, indent=4)
            raise ValueError
        




# =============================================================================
# 
# =============================================================================

uname = os.uname()[1]
sysargv = sys.argv
mainrunfilename = os.path.basename(sysargv[0])

currentrunindex = INPUT['current_run_index']
previousrunindex = int(currentrunindex - 1)
previoudrunindicesall = np.arange(currentrunindex,dtype=int)+1
doupdatethetacovindex = INPUT['mcmc']['do_update_thetacov']

datestamp = date.today().strftime("%Y%m%d")[2:]




mcmc_mainrun_dir = 'mcmc_output_' + os.path.splitext(inputparamsload_filename)[0]

mcmc_mainrun_dir_relpath = os.path.relpath(os.path.join( os.path.dirname(inputparamsload_file_relpath),mcmc_mainrun_dir ))

if rank_mpi == 0:
    if not(os.path.exists(mcmc_mainrun_dir_relpath)):
        os.mkdir(mcmc_mainrun_dir_relpath)
        print('Output stored at location: %s'%(mcmc_mainrun_dir_relpath)); sys.stdout.flush()
        with open(os.path.join(mcmc_mainrun_dir_relpath, 'input.json'),'w') as ff:
            json.dump(INPUT, ff, indent=4)
    else:
        print('Output stored at (already existing) location: %s'%(mcmc_mainrun_dir_relpath)); sys.stdout.flush()
        if len(find_files_containing('input', mcmc_mainrun_dir_relpath)) == 0:            
            with open(os.path.join(mcmc_mainrun_dir_relpath, 'input.json'),'w') as ff:
                json.dump(INPUT, ff, indent=4)
        else:
            lastinputjsonsuffix = find_last_suffix('input', '', filetype='inputjson')
            xx = lastinputjsonsuffix.split('_')[1]
            addsuffix = '_' + '%02d'%(int(xx)+1)
            
            #first compare the two inputs: current input dict and the previous file present
            
            INPUT_wo_cri = INPUT.copy()
            del INPUT_wo_cri['current_run_index']               # current input dict without 'current_run_index' key
            
            
            currentInput_isDoneBefore = 0
            for previnputfile in find_files_containing('input', mcmc_mainrun_dir_relpath):
                with open(os.path.join(mcmc_mainrun_dir_relpath, previnputfile),'r') as ftemp:
                    inputdicttemp = json.load(ftemp)
                # prev_input_dictlist.append(inputdicttemp)
                
                inputdicttemp_wo_cri = inputdicttemp.copy()     # prev input dict without 'current_run_index' key
                del inputdicttemp_wo_cri['current_run_index']
                
                
                if inputdicttemp_wo_cri == INPUT_wo_cri:
                    currentInput_isDoneBefore += 1
                else:
                    currentInput_isDoneBefore += 0
            
            # Now, if this input is not present already as 'input.json' file, then create a new one with a different suffix
            
            if currentInput_isDoneBefore == 0:
                with open(os.path.join(mcmc_mainrun_dir_relpath, 'input%s.json'%(addsuffix)),'w') as ff:
                    json.dump(INPUT, ff, indent=4)
            
            


MCMCmodulespath = os.path.dirname(hiraxmcmc.__file__)

if rank_mpi==0:
    print('MCMCmodulespath: ',MCMCmodulespath)


# if 'MacbookPro' not in os.uname()[1]:

#     if not(os.path.exists(os.path.join(os.getcwd(), '..',  mcmc_mainrun_dir ))):


#     MCMCmodulespath = os.path.abspath(os.path.join(mcmc_mainrun_dir, 'MCMCmodules2'))
# else:
#     try:
#         mcmc_mainrun_dir = find_dir(mcmc_mainrundir_name, os.path.expanduser('~'))
#         MCMCmodulespath = os.path.abspath(os.path.join(mcmc_mainrun_dir,'..', 'MCMCmodules2'))
#     except: 
#         # os.mkdir(os.path.join(os.getcwd(),mcmc_mainrundir_name))
#         mcmc_mainrun_dir = None #find_dir(mcmc_mainrundir_name, os.path.expanduser('~'))
#         MCMCmodulespath = None
#     # MCMCmodulespath = os.path.abspath(os.path.join(mcmc_mainrun_dir,'..', 'MCMCmodules2'))




# =============================================================================
# Take names of the parameters to vary from user input (sys.argv) and make a list
# =============================================================================



ordered_params_list = list(INPUT['PARAMS'].keys())

params_to_vary = params_to_vary_list_from_input(INPUT['params_to_vary'], ordered_params_list)


if not(checkconditionforlist(params_to_vary, allelements_have_subpart='(z)')):
    freqdep_paramstovary = False
elif checkconditionforlist(params_to_vary, allelements_have_subpart='(z)'):
    freqdep_paramstovary = True

if rank_mpi == 0:
    print('Params selected to vary: ',params_to_vary); sys.stdout.flush()



# =============================================================================
# Load mmode sims output and corresponding parameters
# =============================================================================


auto_hiraxoutput_kw = str(INPUT['mmode_output']['freq_channel']['start']) + '_' + str(INPUT['mmode_output']['freq_channel']['end'])

try:
    # freqstart = int(auto_hiraxoutput_kw.split('_')[0])
    # freqend   = int(auto_hiraxoutput_kw.split('_')[1])
    auto_hiraxoutput_kw_list = []
    redshiftlist = []
    auto_hiraxoutput_selection_dir = []
    for freqval in np.arange(INPUT['mmode_output']['freq_channel']['start'], INPUT['mmode_output']['freq_channel']['end'], 100):
        kw = str(freqval)+'_'+str(freqval+100)
        auto_hiraxoutput_kw_list.append(kw)
        fqmid = freqval+50
        redshiftlist.append(freq2z(fqmid))
        auto_hiraxoutput_selection_dir.append(find_subdirs_containing(kw, 
                                                                      os.path.abspath(os.path.join(MCMCmodulespath,'mmoderesults')),
                                                                      fullpathoutput=True)[0])
except:
    raise('Invalid frequency channel input')



# freqmid = [(float(automatichiraxoutputkeywordlist_element.split('_')[0]) 
#             + float(automatichiraxoutputkeywordlist_element.split('_')[1]))/2 
#            for automatichiraxoutputkeywordlist_element in automatichiraxoutputkeywordlist]




# inputforhiraxoutput = [[auto_hiraxoutput_selection_dir[index] , 'minvar', redshift[index]] for index in range(len(redshift))]
inputforhiraxoutput = {}
for index, freqlimits1 in enumerate(auto_hiraxoutput_kw_list):
    assert freqlimits1 in auto_hiraxoutput_selection_dir[index]
    inputforhiraxoutput[freqlimits1] = {}
    inputforhiraxoutput[freqlimits1]['result_dir_name'] = auto_hiraxoutput_selection_dir[index]
    inputforhiraxoutput[freqlimits1]['estimator_type'] = INPUT['mmode_output']['power_spectrum_estimator_type']
    inputforhiraxoutput[freqlimits1]['redshift'] = redshiftlist[index]
    
    btdir = os.path.join(find_subdirs_containing('drift_prod', auto_hiraxoutput_selection_dir[index], fullpathoutput=True)[0],'bt')
    inputforhiraxoutput[freqlimits1]['klmode'] = INPUT['mmode_output']['klmode']
    

numfc = len(list(inputforhiraxoutput.keys()))





hirax_output = {}
covhirax = {}
errs = {}

for index1, freqlimits in enumerate(auto_hiraxoutput_kw_list):
    hirax_output[freqlimits] = HiraxOutput(inputforhiraxoutput[freqlimits])
    covhirax[freqlimits] = hirax_output[freqlimits].covhirax
    errs[freqlimits] = hirax_output[freqlimits].rel_err



if rank_mpi == 0:
    print('\n',hirax_output,'\n'); sys.stdout.flush()



# =============================================================================
# Parameter fixed
# =============================================================================

params_fixed = ParametersFixed()


allparams_fixed = params_fixed.current_allparams_fixed


cosmoparams_fixed = {}
for paramname, paramval in allparams_fixed.items():
    if '(z)' not in paramname:
        cosmoparams_fixed[paramname] = paramval
        



currentparams_fixed = params_fixed.current_params_to_vary_fixed(params_to_vary, toggle_paramstovary_freqdep=freqdep_paramstovary, fclist=inputforhiraxoutput.keys())

# Initialization of currentparams
currentparams = params_fixed.current_params_to_vary_fixed(params_to_vary, toggle_paramstovary_freqdep=freqdep_paramstovary, fclist=inputforhiraxoutput.keys())




# =============================================================================
# mcmc essentials
# =============================================================================



extract_ps2d = False


# priors = {'H0':[50,90],
#           'Omk':[-0.2,0.2],
#           'Oml':[0.5,0.9],
#           'w0':[-3,1],
#           'wa':[-5,5]}

priors = {}
for pp in params_to_vary:
    priors[pp] = INPUT['PARAMS'][pp]['prior']



if 'test' in sys.argv[0]:
    niterations = 1000
else:
    niterations = INPUT['mcmc']['nsteps']

dothetacovupdateafterevery = INPUT['mcmc']['dothetacovupdateafterevery']  #int(niterations/1000)


if rank_mpi == 0:
    print('number of iterations:',niterations); sys.stdout.flush()


if 'test' not in sys.argv[0]:
    testfilekw = ''
elif 'test' in sys.argv[0]:
    testfilekw = '_test'


if len(params_to_vary) != len(ordered_params_list):
    parameterssavetxt = '_%s'*len(params_to_vary)%(tuple(params_to_vary))
elif len(params_to_vary) == len(ordered_params_list):
    parameterssavetxt = ''



"""
totalParams_inclChi2 = totalParams (H0, Omk, Oml, w0, wa) + 1 (chi2)
"""

######### NUMBER OF PARAMETERS ###########

totalParams = len(params_to_vary)

# try:
#     assert not(freqdep_paramstovary)
#     totalParams = len(params_to_vary)
# except:
#     assert freqdep_paramstovary
#     totalParams = len(params_to_vary) * numfc

totalParams_inclChi2 = totalParams + 1

paramsAccepted = np.zeros((totalParams_inclChi2,niterations+1))         # chi2, H0, omk, oml, w0, wa

paramsTrulyAccepted = np.zeros((totalParams_inclChi2,niterations+1))    # chi2, H0, omk, oml, w0, wa


''' 

Final results are saved for all params together in the following format:
    

    # chi2                  # H0                        # Omk               # Oml                       # w0                    # wa
    
22.5791642131668        67.56551868943583       0.0016030778614703027   0.6789796839891553      -1.1443290246621787     0.21478113673805288
22.5791642131668        67.56551868943583       0.0016030778614703027   0.6789796839891553      -1.1443290246621787     0.21478113673805288
19.340476513610334      67.74742201956526       0.004017522382267764    0.678077532994925       -1.1672784196833554     0.30588771800251147
19.340476513610334      67.74742201956526       0.004017522382267764    0.678077532994925       -1.1672784196833554     0.30588771800251147
19.340476513610334      67.74742201956526       0.004017522382267764    0.678077532994925       -1.1672784196833554     0.30588771800251147
       .                        .                         .                     .                         .                       .
       .                        .                         .                     .                         .                       .
       .                        .                         .                     .                         .                       .
       
'''







# =============================================================================
# Directory management
# =============================================================================

# if rank_mpi == 0:
#     print('waiting for directory management ... ') ; sys.stdout.flush()

# time.sleep(10)


if rank_mpi == 0:
    dm = DirManage(mcmc_mainrun_dir_relpath, currentrunindex, testfilekw)
    dm.outputdir_for_class()
    dm.chaindir()
    dm.TRFdir()
    dm.ARdir()
    
# time.sleep(10)


# if rank_mpi == 0:
#     print('... completed directory management') ; sys.stdout.flush()



# =============================================================================
# Load old MCMC results (if any)
# =============================================================================

# 





    
# if INPUT['']

if INPUT['mcmc']['thetacov0']['do_override'] in ['YES','Yes','yes','1',1,'True',True,'true']:
    thetacov0 = np.diag(np.zeros(len(params_to_vary)))
    for i, j in enumerate(params_to_vary):
        thetacov0[i,i] = INPUT['mcmc']['thetacov0']['manual_input_variance'][j]
    
elif INPUT['mcmc']['thetacov0']['do_override'] in ['NO','No','no','0',0,'False',False,'false']:
    thetacov0 = np.diag(np.zeros(len(params_to_vary)))
    for i, j in enumerate(params_to_vary):
        thetacov0[i,i] = priors[j][-1] - priors[j][0]


# if not(freqdep_paramstovary):
#     thetacov0 = np.diag(np.zeros(len(params_to_vary)))
#     for i, j in enumerate(params_to_vary):
#         thetacov0[i,i] = priors[j][-1] - priors[j][0]
#     # thetacov0 = np.diag([25,0.002,0.01,.2,1.5])                                 # Manual input
# elif freqdep_paramstovary:
#     thetacov0 = {}
#     for freqc in list(inputforhiraxoutput.keys()):
#         thetacov0[freqc] = np.diag(np.zeros(len(params_to_vary)))
#         for i, j in enumerate(params_to_vary):
#             thetacov0[freqc][i,i] = priors[j][-1] - priors[j][0]







load_old_res = LoadOlderResults(currentrunindex, params_to_vary, currentparams, ordered_params_list, priors, rank_mpi, mcmc_mainrun_dir_relpath)  # inputforhiraxoutput


load_old_res.check_parameterssavetxt_prev()

burnin_length_for_each_chain_input = 4000

try:
    try:
        if int(currentrunindex) == 2:
            load_old_res.load_allparams_file_and_chains(totalParams_inclChi2, burnin_length_for_each_chain = burnin_length_for_each_chain_input ) 
        else:
            load_old_res.load_allparams_file_and_chains(totalParams_inclChi2)
    except:
        load_old_res.load_allParams_chains_only(totalParams_inclChi2)
        
    # load_old_res.load_allParams_chains_only(totalParams_inclChi2)
except:
    assert int(currentrunindex) == 1
    load_old_res.firstrunparams(thetacov0)
    if rank_mpi == 0:
        print("First run parameters loaded, check the files if you think this is an error.")






# =============================================================================
# currentparams and thetacovauto FOR this particular run (set of params)
# =============================================================================

# if currentrunindex == '1':
#     for pi, pv in enumerate(ordered_params_list):
#         if pv not in params_to_vary:
#             load_old_res.currentparams[pv] = currentparams_fixed[pv]

currentparams = load_old_res.currentparams

# thetacovauto = newcovmatrix(load_old_res.thetacovauto, params_to_vary)
thetacovauto = load_old_res.thetacovauto




# =============================================================================
# For the runs adding a varying param to the previous run params set
# =============================================================================


if len(params_to_vary) > len(load_old_res.prev_params_varied):
    
    """ make list of new varying params in this run as compared to the last run """
    newparamlist_thistime = []
    for par in params_to_vary:
        if par not in load_old_res.prev_params_varied:  # if the varying param for the current run is not in the last run's params varied
            newparamlist_thistime.append(par)
    
    """ Make list of index of the new params in the ordered_params_list """
    list_newParindexThisTime = []
    for pi, pv  in enumerate(ordered_params_list):
        for newparthistime in newparamlist_thistime:
            if newparthistime == pv:
                list_newParindexThisTime.append(pi)
    
    for newparindexthistime in list_newParindexThisTime:
        thetacovauto[newparindexthistime,newparindexthistime] = thetacov0[newparindexthistime,newparindexthistime]
    
    for newparthistime in newparamlist_thistime:
        currentparams[newparthistime] = np.random.uniform( priors[newparthistime][0] , priors[newparthistime][1] )


chi2_func = {}
for freqc,value_inputforhiraxoutput in inputforhiraxoutput.items():
    chi2_func[freqc] = Chi2Func(value_inputforhiraxoutput, INPUT)
    




key0 = list(inputforhiraxoutput.keys())[0]


try:
    assert not(freqdep_paramstovary)
    initial_success = 0
    while initial_success == 0:
        try:
            PK_k_z_current , CLASS_instance_current = chi2_func[key0].cp_params.pofk_from_class(currentparams=currentparams)
            initial_success = 1
        except:
            assert int(currentrunindex) == 1
            load_old_res.firstrunparams(thetacov0)
            currentparams = load_old_res.currentparams
except:
    assert freqdep_paramstovary
    initial_success = 0
    while initial_success == 0:
        try:
            PK_k_z_current , CLASS_instance_current = chi2_func[key0].cp_params.pofk_from_class(currentparams=cosmoparams_fixed)
            initial_success = 1
        except:
            assert int(currentrunindex) == 1
            load_old_res.firstrunparams(thetacov0)
            currentparams = load_old_res.currentparams



# for param in list(currentparams.keys()):
#     if '(z)' not in param:
#         PK_k_z_current , CLASS_instance_current = chi2_func[key0].cp_params.pofk_from_class(currentparams=currentparams)
#     else:
#         PK_k_z_current , CLASS_instance_current = chi2_func[key0].cp_params.pofk_from_class(currentparams=cosmoparams_fixed)


chi2old1 = {}
for freqc,val in inputforhiraxoutput.items():
    chi2old1[freqc] = chi2_func[freqc].chi2_multiz(PK_k_z_currentstep=PK_k_z_current, 
                                                   current_class_instance=CLASS_instance_current, 
                                                   z=val['redshift'],
                                                   currentparams=currentparams,
                                                   cosmoparams=cosmoparams_fixed)




chi2old = np.sum(list(chi2old1.values()))


if rank_mpi==0:
    print('Initial proposal cov matrix: \n',thetacovauto); sys.stdout.flush()




# =============================================================================
# param array initialization per rank
# =============================================================================

paramsAccepted[0,0]         = chi2old
paramsTrulyAccepted[0,0]    = chi2old

for parcol, parname in enumerate(params_to_vary):
    paramsAccepted      [int(parcol+1)  : int(parcol+2), 0] = currentparams[parname]
    paramsTrulyAccepted [int(parcol+1)  : int(parcol+2), 0] = currentparams[parname]


# try:
#     assert not(freqdep_paramstovary)

#     paramsAccepted[:,0] = [chi2old, currentparams['H0'], currentparams['Omk'], currentparams['Oml'], currentparams['w0'], currentparams['wa']]
    
#     paramsTrulyAccepted[:,0] = [chi2old, currentparams['H0'], currentparams['Omk'], currentparams['Oml'], currentparams['w0'], currentparams['wa']]
    
# except:
#     assert freqdep_paramstovary
    
    
#     paramsAccepted[0,0]         = chi2old
#     paramsTrulyAccepted[0,0]    = chi2old
    
#     for parcol, parname in enumerate(params_to_vary):
#         paramsAccepted      [int(parcol*numfc+1)  : int(parcol*numfc+1+numfc), 0] = currentparams[parname]
#         paramsTrulyAccepted [int(parcol*numfc+1)  : int(parcol*numfc+1+numfc), 0] = currentparams[parname]



# =============================================================================
# Chains
# =============================================================================

""" 

Chains are saved for all params together for each chain (rank_mpi) separately in the following format:
    

    # chi2                  # H0                        # Omk               # Oml                       # w0                    # wa
    
22.5791642131668        67.56551868943583       0.0016030778614703027   0.6789796839891553      -1.1443290246621787     0.21478113673805288
22.5791642131668        67.56551868943583       0.0016030778614703027   0.6789796839891553      -1.1443290246621787     0.21478113673805288
19.340476513610334      67.74742201956526       0.004017522382267764    0.678077532994925       -1.1672784196833554     0.30588771800251147
19.340476513610334      67.74742201956526       0.004017522382267764    0.678077532994925       -1.1672784196833554     0.30588771800251147
19.340476513610334      67.74742201956526       0.004017522382267764    0.678077532994925       -1.1672784196833554     0.30588771800251147
       .                        .                         .                     .                         .                       .
       .                        .                         .                     .                         .                       .
       .                        .                         .                     .                         .                       .
       

"""


chainf = Chains(currentrunindex= currentrunindex, totalParams_inclChi2=totalParams_inclChi2, sysargs=sys.argv, 
                rankmpi=rank_mpi, comm=comm, testfilekw=testfilekw, 
                parameterssavetxt=parameterssavetxt, write_out_paramsTrulyAccepted=True, mcmc_mainrun_dir_relpath=mcmc_mainrun_dir_relpath)
# chainf = Chains(totalParams_inclChi2, sys.argv, rank_mpi, testfilekw, parameterssavetxt, write_out_paramsTrulyAccepted=True)

# if rank_mpi == 0:
#     chainf.remove_olderChainFiles_forThisRun()

print('\n')

time.sleep(5)


print('addsuffix -- rank %s: %s'%(rank_mpi,chainf.addsuffix))


time.sleep(5)

print('\n')


chainf.write_chains(stepindex=0, paramsAcceptedarg=paramsAccepted, paramsTrulyAcceptedarg=paramsTrulyAccepted)



# =============================================================================
# thetacov update function
# =============================================================================

time.sleep(20)


tcu = ThetaCovUpdate(INPUT, comm, rank_mpi, size_mpi, currentparams, 
                     chainf, mcmc_mainrun_dir_relpath)   # inputforhiraxoutput


thetacov_reductionfactor_initial = tcu.thetacov_reductionfactor_initial

if rank_mpi == 0:
    print('\n')

time.sleep(2)

print('Rank %s: initial thetacov reduction factor: %s'%(rank_mpi,thetacov_reductionfactor_initial)); sys.stdout.flush()

time.sleep(5)

if rank_mpi == 0:
    print('\n')

# =============================================================================
# ps2d array (for ML Alireza)
# =============================================================================


# if int(currentrunindex) == 1:
#     includeOlderPs2darray = False
# elif int(currentrunindex) != 1:
#     includeOlderPs2darray = True


# if rank_mpi==0:
#     if includeOlderPs2darray:
#         if os.path.exists('ps2d_for_Alireza_run%s%s.dat'%(int(int(currentrunindex)-1) , testfilekw)):
            
#             loadps2d = np.loadtxt('ps2d_for_Alireza_run%s%s.dat'%(int(int(currentrunindex)-1),testfilekw))
            
#             nonZeroElementsLen = int(len(np.where(loadps2d != np.zeros((270)))[0])/270)
#             loadps2d_masked = loadps2d[np.where(loadps2d != 0)].reshape(nonZeroElementsLen , 270)
            
#             loadps2d_masked_appended = np.concatenate((loadps2d_masked, np.zeros((niterations,270))))
            
#             ps2darray = loadps2d_masked_appended.reshape(loadps2d_masked_appended.shape[0],9,30)
            
#         else:
#             nonZeroElementsLen = 0
#             ps2darray = np.zeros((niterations,9,30))

#     else:
#         nonZeroElementsLen = 0
#         ps2darray = np.zeros((niterations,9,30))


doExtractForML = False

if doExtractForML:
    
    if rank_mpi == 0:
        dm.extractsforMLdir()
    
    do_get_pscalc_out_for_chi2 = True

    extractforml = extractForML(savetextadd='cc', 
                                parameterssavetxt=parameterssavetxt, 
                                addsuffix=chainf.addsuffix, 
                                sysargs=sys.argv, 
                                niterations=niterations, 
                                totalParams_inclChi2=totalParams_inclChi2,
                                rankmpi = rank_mpi,
                                pscalc_initial=pscalcold)
    
    
    extractforml.append_array_ps2d_all(pscalc=pscalcold, 
                                       currentstep=0)
    
    extractforml.append_chain_all(chi2=chi2old, 
                                  currentparamslist=list(currentparams.values()), 
                                  currentstep=0)
    
    
    
else:
    do_get_pscalc_out_for_chi2 = False
    


# =============================================================================
# Printing things
# =============================================================================


print("Rank %s starting with"%(rank_mpi),"χ2_old =",chi2old,"and",currentparams); sys.stdout.flush()

time.sleep(5)

if rank_mpi == 0:
    print('\n')

if rank_mpi == 0:
    print('waiting to start MCMC ... ') ; sys.stdout.flush()
    

time.sleep(10)

if rank_mpi == 0:
    print('... now starting MCMC :)') ; sys.stdout.flush()

time.sleep(5)


# =============================================================================
# =============================================================================
#      __        __       ______       __        __       ______  
#     |  \      /  |     / ____ \     |  \      /  |     / ____ \ 
#     |   \    /   |    / /    \_\    |   \    /   |    / /    \_\
#     |    \  /    |   | |            |    \  /    |   | |               
#     |  |\ \/ /|  |   | |            |  |\ \/ /|  |   | |  
#     |  | \__/ |  |   | |            |  | \__/ |  |   | |
#     |  |      |  |   | |      __    |  |      |  |   | |  
#     |  |      |  |    \ \____/ /    |  |      |  |    \ \____/ /  
#     |__|      |__|     \______/     |__|      |__|     \______/
# 
# =============================================================================
# =============================================================================

#                                ____
#                               |    |
#                               |    |
#                               |    |
#                             __|    |__
#                             \        /
#                              \      /
#                               \    /
#                                \  /
#                                 \/


for ii in np.arange(1,int(niterations+1)):
    
    def u():
        return np.random.uniform(0,1)
        
    # if ii == 1:
    #     if not(freqdep_paramstovary):
    #         try:
    #             assert int(currentrunindex) != 1
    #             thetacov_reductionfactor = thetacov_reductionfactor_initial
    #             thetacov = thetacovauto * thetacov_reductionfactor
    #         except:
    #             print('first run, using thetacov0'); sys.stdout.flush()
    #             thetacov_reductionfactor = thetacov_reductionfactor_initial
    #             thetacov = thetacovauto
    #     elif freqdep_paramstovary:
    #         thetacov = {}
    #         for fcname_init in inputforhiraxoutput.keys():
    #             try:
    #                 assert int(currentrunindex) != 1
    #                 thetacov_reductionfactor = thetacov_reductionfactor_initial
    #                 thetacov[fcname_init] = thetacovauto[fcname_init] * thetacov_reductionfactor[fcname_init]
    #             except:
    #                 print('first run, using thetacov0'); sys.stdout.flush()
    #                 thetacov_reductionfactor = thetacov_reductionfactor_initial
    #                 thetacov[fcname_init] = thetacovauto[fcname_init]
    
    if ii == 1:
        thetacov_reductionfactor = thetacov_reductionfactor_initial
        thetacov = thetacovauto
    
    
    # =============================================================================
    # =============================================================================
    
    try:
        
        # any of "chi2_func_*" works for pofk_interpolator_for_pscalc
        try:
            assert not(freqdep_paramstovary)
            PK_k_z_current , CLASS_instance_current = chi2_func[key0].cp_params.pofk_from_class(currentparams=currentparams)
        except:
            assert freqdep_paramstovary
            
        
        chi2new1 = {}
        for freqc,val in inputforhiraxoutput.items():
            chi2new1[freqc] = chi2_func[freqc].chi2_multiz(PK_k_z_currentstep=PK_k_z_current, 
                                                           current_class_instance=CLASS_instance_current, 
                                                           z=val['redshift'],
                                                           currentparams=currentparams,
                                                           cosmoparams=cosmoparams_fixed)
        
        # try:
        #     assert not(freqdep_paramstovary)
        #     chi2new1 = {}
        #     for freqc,val in inputforhiraxoutput.items():
        #         chi2new1[freqc] = chi2_func[freqc].chi2_multiz(PK_k_z_currentstep=PK_k_z_current, 
        #                                                       current_class_instance=CLASS_instance_current, 
        #                                                       z=val[2],
        #                                                       currentparams=currentparams)
        # except:
        #     assert freqdep_paramstovary
        #     chi2new1 = {}
        #     for freqc,val in inputforhiraxoutput.items():
        #         chi2new1[freqc] = chi2_func[freqc].chi2_multiz_p2vfreqdep(PK_k_z_currentstep=PK_k_z_current, 
        #                                                                  current_class_instance=CLASS_instance_current, 
        #                                                                  z=val[2],
        #                                                                  currentparams=currentparams,
        #                                                                  cosmoparams=cosmoparams_fixed)
        
        
        
        chi2new = np.sum(list(chi2new1.values()))
        
        if not paramv_within_priors(priors, currentparams):
            chi2new = 1e99
        
        
        ### for ml ###
        if doExtractForML:
            extractforml.append_array_ps2d_all(pscalc=pscalcnew, currentstep=ii)
            extractforml.append_chain_all(chi2=chi2new, currentparamslist=list(currentparams.values()), currentstep=ii)
            
        ##############
        
        
        
        
        if chi2new < chi2old or np.exp(-0.5*(chi2new-chi2old)) > u():          # MAIN SELECTION LINE
            # paramsAccepted[:,ii] = [chi2new, currentparams['H0'], currentparams['Omk'], currentparams['Oml'], currentparams['w0'], currentparams['wa']]            
            
            paramsAccepted[0,ii]         = chi2new
            paramsTrulyAccepted[0,ii]    = chi2new
            
            for parcol, parname in enumerate(params_to_vary):
                paramsAccepted      [int(parcol+1)  : int(parcol+2), ii] = currentparams[parname]
                paramsTrulyAccepted [int(parcol+1)  : int(parcol+2), ii] = currentparams[parname]
            
            
            chi2old = chi2new
            print("Step:",ii,"\n χ2 =",chi2new,"@rank =",rank_mpi); sys.stdout.flush()
            
            
            # paramsTrulyAccepted[:,ii] = [chi2new, currentparams['H0'], currentparams['Omk'], currentparams['Oml'], currentparams['w0'], currentparams['wa']]
            
            
            # paramsTrulyAccepted[0,ii]     = chi2new
            # paramsTrulyAccepted[1:5,ii]   = list(  list(currentparams.values())[0].values()  )
            # paramsTrulyAccepted[5:9,ii]   = list(  list(currentparams.values())[1].values()  )
            # paramsTrulyAccepted[9:13,ii]  = list(  list(currentparams.values())[2].values()  )
            
            
            # if rank_mpi==0:
            #     ps2darr[int(nonZeroElementsLen + ii - 1)] = pscalcnew
            if doExtractForML:
                extractforml.accepted_or_rejected_param(accepted=1, currentstep=ii)
            
        else:
            print("Step:",ii,"\n χ2 =",chi2new," <-- XXXXXX @rank =",rank_mpi); sys.stdout.flush()
 
            paramsAccepted[:,ii] = paramsAccepted[:,int(ii-1)]
            
            if doExtractForML:
                extractforml.accepted_or_rejected_param(accepted=0, currentstep=ii)
                
            # if rank_mpi==0:
            #     ps2darray[int(nonZeroElementsLen + ii - 1)] = ps2darray[int(nonZeroElementsLen + ii - 2)]
        
    except:
        
        if doExtractForML:
            extractforml.append_array_ps2d_all(pscalc=extractforml.ps2darrall[int(ii-1)] , currentstep=ii)
            extractforml.append_chain_all(chi2=extractforml.allparams[0,int(ii-1)], currentparamslist=extractforml.allparams[1:5,int(ii-1)], currentstep=ii)
            extractforml.accepted_or_rejected_param(accepted=0, currentstep=ii)
            
        
        paramsAccepted[:,ii] = paramsAccepted[:,int(ii-1)]
        
        print("Step:",ii,"\n chi2 calculation failed: exception @ rank =",rank_mpi); sys.stdout.flush()
        
        # if rank_mpi==0:
        #     ps2darray[int(nonZeroElementsLen + ii - 1)] = ps2darray[int(nonZeroElementsLen + ii - 2)]
        
    # =============================================================================
    # =============================================================================
    
    thetanew = paramsAccepted[1:,ii]
    
    # currentparams['H0'], currentparams['Omk'], currentparams['Oml'], currentparams['w0'], currentparams['wa'] = mvn(thetanew, thetacov)
    
    newpoint_temp = mvn(thetanew , thetacov)
    
    for i, j in enumerate(params_to_vary):
        currentparams[j] = newpoint_temp[i]
    

    # try:
    #     assert not(freqdep_paramstovary)
    #     currentparams['H0'], currentparams['Omk'], currentparams['Oml'], currentparams['w0'], currentparams['wa'] = mvn(thetanew, thetacov)
    # except:
    #     assert freqdep_paramstovary
    #     for fc_i, fc_j in enumerate(inputforhiraxoutput.keys()):
    #         currentparams['h(z)'], currentparams['dA(z)'], currentparams['f(z)'] = mvn(thetanew[fc_i::], thetacov)
    
    
    
    # =============================================================================
    # =============================================================================
    
    
    chainf.write_chains(stepindex = ii, paramsAcceptedarg = paramsAccepted, paramsTrulyAcceptedarg = paramsTrulyAccepted)
    
    if doExtractForML and np.mod(ii,1000) == 0:
        extractforml.save_final_array_ps2d_all(extractforml.ps2darrall)
        extractforml.save_final_chain_all(extractforml.allparams)
    
    
    # =============================================================================
    # =============================================================================
    
# check this before running
    # if ii <= 20000:
    #     dothetacovupdateafterevery = 4000
    # else:
    #     dothetacovupdateafterevery = dothetacovupdateafterevery
    
    
    
    # try:
    #     assert not(freqdep_paramstovary)
        
    if np.mod(ii,dothetacovupdateafterevery) == 0 and ii != 1: #int(niterations/10)
        thetacov, thetacov_reductionfactor = tcu.thetacovUpdate(thetacovold = thetacov,
                                                                thetacov_reductionfactor = thetacov_reductionfactor, 
                                                                currentstep = ii, 
                                                                paramsAccepted_excdChi2_CurrentFullArray = paramsAccepted[1:], 
                                                                paramsTrulyAccepted_excdChi2_CurrentFullArray = paramsTrulyAccepted[1:]
                                                                )
        
        # print('thetacov at rank %s is \n%s'%(rank_mpi,thetacov)); sys.stdout.flush()    
        
        # np.save('TRF%s/run%s/rank%s_TRF'%(testfilekw,currentrunindex,rank_mpi),thetacov_reductionfactor)
        
        tcu.saveAR(tcu.acceptance_rate)
        tcu.saveTRF(thetacov_reductionfactor)
    
    # except:
    #     assert freqdep_paramstovary
        
    #     if np.mod(ii,dothetacovupdateafterevery) == 0 and ii != 1: #int(niterations/10)
    #         for fc_name_index, fc_name in enumerate(inputforhiraxoutput.keys()):
    #             thetacov[fc_name], thetacov_reductionfactor[fc_name] = tcu.thetacovUpdate(thetacovold = thetacov[fc_name],
    #                                                                                       thetacov_reductionfactor = thetacov_reductionfactor[fc_name], 
    #                                                                                       currentstep = ii, 
    #                                                                                       paramsAccepted_excdChi2_CurrentFullArray = paramsAccepted[int(fc_name_index+1)::numfc], 
    #                                                                                       paramsTrulyAccepted_excdChi2_CurrentFullArray = paramsTrulyAccepted[int(fc_name_index+1)::numfc]
    #                                                                                       )
                
    #             # print('thetacov at rank %s is \n%s'%(rank_mpi,thetacov)); sys.stdout.flush()    
                
    #             # np.save('TRF%s/run%s/rank%s_TRF'%(testfilekw,currentrunindex,rank_mpi),thetacov_reductionfactor)
                
    #             tcu.saveAR(tcu.acceptance_rate)
    #             tcu.saveTRF(thetacov_reductionfactor[fc_name],fcname=fc_name)
        
        
        
    # if rank_mpi==0 and np.mod(ii,100) == 0:
    #     np.savetxt('ps2d_for_Alireza_run%s%s.dat'%(currentrunindex,testfilekw),ps2darray.reshape(ps2darray.shape[0],270))
    
    
    ###########################################################################
    ############################# END OF FOR LOOP #############################
    ###########################################################################


#                                 /\
#                                /  \
#                               /    \
#                              /      \
#                             /        \
#                            /__      __\
#                               |    |
#                               |    |
#                               |    |
#                               |____|













paramsAcceptedTempReshaped = None
    
if rank_mpi==0:
    paramsAcceptedTempReshaped = np.empty([size_mpi,totalParams_inclChi2*np.shape(paramsAccepted)[-1]])

paramsAcceptedReshaped = paramsAccepted.reshape(totalParams_inclChi2 * np.shape(paramsAccepted)[-1])






# comm.Barrier()
"""
https://stackoverflow.com/questions/43987304/mpi-barrier-with-mpi-gather-using-small-vs-large-data-set-sizes
https://stackoverflow.com/questions/13305814/when-do-i-need-to-use-mpi-barrier
"""


comm.Gather(paramsAcceptedReshaped,paramsAcceptedTempReshaped,root=0)





if (rank_mpi==0):
    
    addsuffix = chainf.addsuffix
    
    paramsAcceptedTemp = paramsAcceptedTempReshaped.reshape(size_mpi,totalParams_inclChi2,np.shape(paramsAccepted)[-1])
    
    
    
    print('Gathering done, now saving ...')
    sys.stdout.flush()
    
    
    
    # chi2acceptedFinal = paramsAcceptedTemp[:,0,:].flatten()
    # H0acceptedFinal = paramsAcceptedTemp[:,1,:].flatten()
    # omkacceptedFinal = paramsAcceptedTemp[:,2,:].flatten()
    # omlacceptedFinal = paramsAcceptedTemp[:,3,:].flatten()
    # w0acceptedFinal = paramsAcceptedTemp[:,4,:].flatten()
    # waacceptedFinal = paramsAcceptedTemp[:,5,:].flatten()
    
    # paramsAcceptedFinal = np.zeros((totalParams_inclChi2,len(chi2acceptedFinal)))
    # paramsAcceptedFinal[0] = chi2acceptedFinal
    # paramsAcceptedFinal[1] = H0acceptedFinal
    # paramsAcceptedFinal[2] = omkacceptedFinal
    # paramsAcceptedFinal[3] = omlacceptedFinal
    # paramsAcceptedFinal[4] = w0acceptedFinal
    # paramsAcceptedFinal[5] = waacceptedFinal
    
    
    
    chi2acceptedFinal = paramsAcceptedTemp[:,0,:].flatten()
    
    paracceptedFinal = {}
    
    for newparcol, newparname in enumerate(params_to_vary):
        paracceptedFinal[newparname] = paramsAcceptedTemp[:,int(newparcol+1)  : int(newparcol+2),:].flatten()
        
    
    
    paramsAcceptedFinal = np.zeros((totalParams_inclChi2,len(chi2acceptedFinal)))
    paramsAcceptedFinal[0] = chi2acceptedFinal
    for newparcol, newparname in enumerate(params_to_vary):
        paramsAcceptedFinal[int(newparcol+1)] = paracceptedFinal[newparname]
    
    
    
    
    # np.savetxt(os.path.join(mcmc_mainrun_dir_relpath, 'paramsAcceptedFinal_backup1.dat'),paramsAcceptedFinal.T)#.flatten('F'))
    np.savetxt(os.path.join(mcmc_mainrun_dir_relpath, '%s_paramsAcceptedFinal_cambcamb'%(currentrunindex)+parameterssavetxt+addsuffix+'%s.dat'%(testfilekw)),paramsAcceptedFinal.T)#.flatten('F'))





# if int(currentrunindex) == 1:
#     save_with_burnin_removed = False
# else:
#     save_with_burnin_removed = False
    
# if save_with_burnin_removed:
#     paramsAccepted_BIrem = paramsAccepted[burnin_length_for_each_chain_input:]
#     paramsAcceptedTempReshaped_BIrem = None
#     if rank_mpi==0:
#         paramsAcceptedTempReshaped_BIrem = np.empty([size_mpi,totalParams_inclChi2*np.shape(paramsAccepted_BIrem)[-1]])
    
#     paramsAcceptedReshaped_BIrem = paramsAccepted_BIrem.reshape(totalParams_inclChi2 * np.shape(paramsAccepted_BIrem)[-1])
    
#     comm.Gather(paramsAcceptedReshaped_BIrem,paramsAcceptedTempReshaped_BIrem,root=0)

#     if rank_mpi == 0:
#         addsuffix = chainf.addsuffix
#         paramsAcceptedTemp_BIrem = paramsAcceptedTempReshaped_BIrem.reshape(size_mpi,totalParams_inclChi2,np.shape(paramsAccepted_BIrem)[-1])
        
#         print('Also for saving with burn-in removed, gathering done, now saving ...'); sys.stdout.flush()
        
#         chi2acceptedFinal_BIrem = paramsAcceptedTemp_BIrem[:,0,:].flatten()
#         H0acceptedFinal_BIrem = paramsAcceptedTemp_BIrem[:,1,:].flatten()
#         omkacceptedFinal_BIrem = paramsAcceptedTemp_BIrem[:,2,:].flatten()
#         omlacceptedFinal_BIrem = paramsAcceptedTemp_BIrem[:,3,:].flatten()
#         w0acceptedFinal_BIrem = paramsAcceptedTemp_BIrem[:,4,:].flatten()
#         waacceptedFinal_BIrem = paramsAcceptedTemp_BIrem[:,5,:].flatten()
        
#         paramsAcceptedFinal_BIrem = np.zeros((totalParams_inclChi2,len(chi2acceptedFinal_BIrem)))
#         paramsAcceptedFinal_BIrem[0] = chi2acceptedFinal_BIrem
#         paramsAcceptedFinal_BIrem[1] = H0acceptedFinal_BIrem
#         paramsAcceptedFinal_BIrem[2] = omkacceptedFinal_BIrem
#         paramsAcceptedFinal_BIrem[3] = omlacceptedFinal_BIrem
#         paramsAcceptedFinal_BIrem[4] = w0acceptedFinal_BIrem
#         paramsAcceptedFinal_BIrem[5] = waacceptedFinal_BIrem
        
#         np.savetxt('%s_paramsAcceptedFinal_cambcamb'%(currentrunindex)+parameterssavetxt+addsuffix+'%s_BIrem.dat'%(testfilekw),paramsAcceptedFinal.T)#.flatten('F'))
        


if rank_mpi == 0:
    print('DONE at last')

#



