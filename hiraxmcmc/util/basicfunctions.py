#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,sys
import numpy as np

from scipy.constants import c as speedoflight




def freq2lambda(freq):
    """
    Convert freq in MHz to wavelength in cm
    Parameters
    ----------
    freq : float
        frequency value in MHz.
    Returns
    -------
    float
        wavelength in cm.
    """
    return speedoflight/(freq*1e6) * 100

def lambda2freq(lambda1):
    """
    Convert wavelength in cm to freq in MHz
    Parameters
    ----------
    lambda1 : float
        wavelength value in cm
    Returns
    -------
    float
        frequency value in MHz.
    """
    return speedoflight/(lambda1*0.01)*1e-6

def z2freq(z):
    """
    Convert 21cm redshift to freq in MHz
    Parameters
    ----------
    z : float
        redshift (unitless).
    Returns
    -------
    float
        frequency in MHz.

    """
    return lambda2freq(21)/(1+z)

def z2lambda(z):
    """
    Convert 21cm redshift to wavelength in cm
    Parameters
    ----------
    z : float
        redshift (unitless).
    Returns
    -------
    float
        wavelengh in cm.
    """
    return (1+z) * 21

def freq2z(freq):
    """
    Convert 21cm freq in MHz to redshift
    Parameters
    ----------
    freq : float
        frequency value in MHz.
    Returns
    -------
    float
        redshift (unitless).
    """
    return lambda2freq(21)/freq - 1

def lambda2z(lambda1):
    """
    Convert 21cm wavelength (in cm) to redshift
    Parameters
    ----------
    lambda1 : float
        wavelengh in cm.
    Returns
    -------
    float
        redshift (unitless).
    """
    return lambda1/21 - 1

def find_freqchannel_for_redshift(z, fc_all = ['400_500', '500_600', '600_700', '700_800']):
    for i in fc_all:
        if float(i.split('_')[0]) <= z2freq(z) <= float(i.split('_')[1]):
            return i


# =============================================================================
# Prior limiting function
# =============================================================================

def paramv_within_priors(priors, currentparams):
    """
    This function determines whether the currentparams are within prior or not

    Parameters
    ----------
    priors : type: DICT
        Eg. priors = {'H0':[60,75],
                      'Omk':[-0.2,0.2],
                      'Oml':[0.6,0.8],
                      'w0':[-2,0],
                      'wa':[-1,1]}
    currentparams : type: DICT
        Eg. {'H0': 67.8, 'Omk': 0.0, 'Oml': 0.684, 'w0': -1.0, 'wa': 0.0}

    Returns True/False
    -------
    type: BOOL
    - True: if ALL parameter values are within set priors
    - False: if even SOME parameter values are outside the set priors
    """
    templist = []
    for elem in list(priors.keys()):
        templist.append(bool(priors[elem][0] < currentparams[elem] < priors[elem][1]))
    return bool(np.product(templist))
    
# =============================================================================
# search for files/subdirs
# =============================================================================


def find_files_begin_with(str1,dirname,fullpathoutput=False):
    """
    Find files beginning with a given string argument in the given directory name argument

    Parameters
    ----------
    str1 : str
        string to search for in the beginning of the files.
    dirname : str
        full path of directory to search in.
    fullpathoutput : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    entries : numpy array 
        List of files present in the given dir.

    """
    entries = []
    listfiles = [name for name in os.listdir(dirname) if os.path.isfile(os.path.join(dirname,name))]
    for entry in listfiles:
        if entry[:len(str1)]==str1:
            entries.append(entry)
    if fullpathoutput:
        entries = [os.path.abspath(os.path.join(dirname,entry)) for entry in entries]
    return entries



def find_files_containing(str1,dirname, fullpathoutput=False):
    """
    This function is used to list all the files containing <str1> in the 
    path/dir <dirname>

    Parameters
    ----------
    str1 : string
        Any part/full of filename to search for
    dirname : string
        full path of directory to search in.

    Returns
    -------
    entries : numpy array 
        List of the file names found 
    """
    entries = []
    listfiles = [name for name in os.listdir(dirname) if os.path.isfile(os.path.join(dirname,name))]
    for entry in listfiles:
        try:
            assert type(str1) == str
            if str1 in entry:
                entries.append(entry)
        except:
            assert type(str1) == list
            if all([elem in entry for elem in str1]):
                entries.append(entry)
    if fullpathoutput:
        entries = [os.path.abspath(os.path.join(dirname,entry)) for entry in entries]
    return entries



def find_subdirs_begin_with(str1,dirname, fullpathoutput=False):
    """
    This function is used to list all the subdirectories beginning with <str1> in 
    the given path/dir <dirname>

    Parameters
    ----------
    str1 : string
        string to search for in the beginning of the subdirs.
    dirname : string
        full path of directory to search in.
    fullpathoutput : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    entries : Numpy array
        List of the subdir names found (not full paths).

    """
    entries = []
    listsubdirs = [subdir for subdir in os.listdir(os.path.abspath(dirname)) if os.path.isdir(os.path.join(dirname,subdir))]   # the list only includes directories
    for entry in listsubdirs:    
        if entry[:len(str1)]==str1:
            if fullpathoutput:
                entries.append(os.path.abspath(os.path.join(dirname,entry))) # os.path.join(dirname, entry))
            else:
                entries.append(entry)
    return entries

def find_subdirs_containing(str1,dirname,fullpathoutput=False):
    """
    This function is used to list all the subdirectories containing <str1> in 
    the path/dir <dirname>

    Parameters
    ----------
    str1 : string
        Any part/full of the subdir name to search for
    dirname : string
        DESCRIPTION. Full path of the immediate parent directory to be searched in

    Returns
    -------
    entries : Numpy array
        List of the subdir names found (not full paths)
    """
    entries = []
    listsubdirs = [name for name in os.listdir(os.path.abspath(dirname)) if os.path.isdir(os.path.join(dirname,name))]
    for entry in listsubdirs:
        try:
            assert type(str1) == str
            if str1 in entry:
                entries.append(entry)
        except:
            assert type(str1) == list# the list only includes directories 
            if all([elem in entry for elem in str1]):
                entries.append(entry)
    if fullpathoutput:
        entries = [os.path.abspath(os.path.join(dirname,entry)) for entry in entries]
    return entries
            
# =============================================================================
# Matrix sections
# =============================================================================

def extract_matrix_section(matrix, givenparams, 
                           ordered_params_list=['H0','Omk','Oml','w0','wa']):
    """
    For the ordered_params_list (Eg. ['H0', 'Omk', 'Oml', 'w0', 'wa']), this 
    function finds a section corresponding to the <givenparams>

    Eg.      H0   Omk  Oml  w0   wa
        H0   xa1  xa2  xa3  xa4  xa5   <givenparams>
        Omk  xb1  xb2  xb3  xb4  xb5    ['H0, w0']          H0    w0
        Oml  xc1  xc2  xc3  xc4  xc5   ----------->   H0 [[ xa1  xa4 ]
        w0   xd1  xd2  xd3  xd4  xd5                  w0  [ xd1  xd4 ]]
        wa   xe1  xe2  xe3  xe4  xe5
        
    Parameters
    ----------
    matrix : TYPE. numpy array
        DESCRIPTION. 5x5 matrix (Eg. cov matrix) corresponding to all the 
        parameters in ordered_params_list
    givenparams : TYPE. list
        DESCRIPTION. list of section/all of parameter names in ordered_params_list

    Returns
    -------
    newmatrix : numpy array
        DESCRIPTION. Give in the descriptive example above
    """
    # ordered_params_list = ['H0','Omk','Oml','w0','wa']
    givenparamsindexlist = []
    for i,j in enumerate(ordered_params_list):
        if j in givenparams:
            givenparamsindexlist.append(i)
    # print(givenparamsindexlist)
    newmatrix = np.zeros([len(givenparamsindexlist), len(givenparamsindexlist)])
    combinations = []
    for ii,jj in enumerate(givenparamsindexlist):
        for ii2, jj2 in enumerate(givenparamsindexlist):
            combinations.append([[ii,ii2],[jj,jj2]])
    for ii3 in combinations:
        newmatrix[ii3[0][0],ii3[0][1]] = matrix[ii3[1][0],ii3[1][1]]
    return newmatrix


def checkconditionforlist(inputlist, allelements_have_subpart=None, allelements_equalto=None, allelements_are_of_type=None):
    if allelements_have_subpart != None:
        templist = []
        for elem in inputlist:
            if allelements_have_subpart in elem:
                templist.append(True)
            else:
                templist.append(False)
        return bool(np.product(templist))
    elif allelements_equalto != None:
        templist = []
        for elem in inputlist:
            if allelements_equalto == elem:
                templist.append(True)
            else:
                templist.append(False)
        return bool(np.product(templist))
    elif allelements_are_of_type != None:
        templist = []
        for elem in inputlist:
            if type(elem) == allelements_are_of_type:
                templist.append(True)
            else:
                templist.append(False)
        return bool(np.product(templist))


def newcovmatrix(oldcovmatrix, givenparams, ordered_params_list):
    """
    For the ordered_params_list (Eg. ['H0', 'Omk', 'Oml', 'w0', 'wa']), this 
    function finds a section corresponding to the <givenparams>

    Eg.      H0   Omk  Oml  w0   wa                        H0   Omk  Oml  w0   wa 
        H0   xa1  xa2  xa3  xa4  xa5   <givenparams>  H0   xa1  0    0    xa4  0  
        Omk  xb1  xb2  xb3  xb4  xb5    ['H0, w0']    Omk  0    0    0    0    0
        Oml  xc1  xc2  xc3  xc4  xc5   ----------->   Oml  0    0    0    0    0
        w0   xd1  xd2  xd3  xd4  xd5                  w0   xd1  0    0    xd4  0
        wa   xe1  xe2  xe3  xe4  xe5                  wa   0    0    0    0    0
        
    Parameters
    ----------
    matrix : TYPE. numpy array
        DESCRIPTION. 5x5 matrix (Eg. cov matrix) corresponding to all the 
        parameters in ordered_params_list
    givenparams : TYPE. list
        DESCRIPTION. list of section/all of parameter names in ordered_params_list

    Returns
    -------
    newmatrix : numpy array
        DESCRIPTION. Give in the descriptive example above
    """
    notgivenparamsindexlist = []
    for i,j in enumerate(ordered_params_list):
        if j not in givenparams:
            notgivenparamsindexlist.append(i)
    # print(givenparamsindexlist)
    newmatrix = np.copy(oldcovmatrix)
    for i2 in notgivenparamsindexlist:
        newmatrix[i2] = np.zeros(3)
        newmatrix[:,i2] = np.zeros(3)
    return newmatrix
    
    
def newcovmatrix_advanced(oldcovmatrix, params2vary, ordered_params_list):
    """
    For the ordered_params_list (Eg. ['H0', 'Omk', 'Oml', 'w0', 'wa']), this 
    function finds a section corresponding to the <givenparams>

    Eg.      H0   Omk  Oml  w0   wa                        H0   Omk  Oml  w0   wa 
        H0   xa1  xa2  xa3  xa4  xa5   <givenparams>  H0   xa1  0    0    xa4  0  
        Omk  xb1  xb2  xb3  xb4  xb5    ['H0, w0']    Omk  0    0    0    0    0
        Oml  xc1  xc2  xc3  xc4  xc5   ----------->   Oml  0    0    0    0    0
        w0   xd1  xd2  xd3  xd4  xd5                  w0   xd1  0    0    xd4  0
        wa   xe1  xe2  xe3  xe4  xe5                  wa   0    0    0    0    0
        
    Parameters
    ----------
    matrix : TYPE. numpy array
        DESCRIPTION. 5x5 matrix (Eg. cov matrix) corresponding to all the 
        parameters in ordered_params_list
    givenparams : TYPE. list
        DESCRIPTION. list of section/all of parameter names in ordered_params_list

    Returns
    -------
    newmatrix : numpy array
        DESCRIPTION. Give in the descriptive example above
    """
    nm = np.zeros(np.shape(np.diag(params2vary)))
    
    for p2vi,p2viv in enumerate(params2vary):
        for p2vj,p2vjv in enumerate(params2vary):
            if (p2viv in ordered_params_list) and (p2vjv in ordered_params_list):
                for opli1, oplj1 in enumerate(ordered_params_list):
                    for opli2,oplj2 in enumerate(ordered_params_list):
                        if (oplj1 == p2viv) and (oplj2 == p2vjv):
                            nm[p2vi,p2vj] = oldcovmatrix[opli1,opli2]
    
    return nm


def find_dir(name, path):
    """
    Find directory <name> in the path <path>, with <path> not being of the 
    immediate parent directory necessarily, i.e., this function can work 
    recursively.

    Parameters
    ----------
    name : TYPE str
        Name of dir to find.
    path : TYPE str
        Full path of (non-)immediate parent directory.

    Returns
    -------
    TYPE str
        full path of the directory <name> if it is present in the given <path> input.
        
    """
    for root, dirs, files in os.walk(path):
        if name in dirs:
            return os.path.join(root, name)
        
def find_file(name, path):
    """
    Find file <name> in the path <path>, with <path> not being of the 
    immediate parent directory necessarily, i.e., this function can work 
    recursively.

    Parameters
    ----------
    name : TYPE str
        Name of file to find.
    path : TYPE str
        Full path of (non-)immediate parent directory.

    Returns
    -------
    TYPE str
        full path of the file <name> if it is present in the given <path> input.
        
    """
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

        


def find_last_suffix(filenamepartToSearchFor ,dirnameToSearchIn , filetype='chains'):
    try:
        if filetype.lower() == 'chains':
            listofPrevChainsNames = find_files_containing(filenamepartToSearchFor , dirnameToSearchIn)
            if len(listofPrevChainsNames) != 0:
                prevfiles0 = []
                for ff in listofPrevChainsNames:
                    if ff[0] == '0':              # <--- 0th letter for chain file names
                        prevfiles0.append(ff)
                    else:
                        pass
                suffixList = [int(pfile.split('.')[0][-2:]) for pfile in prevfiles0]
                lastsuffnumber = np.max(suffixList)
                for ffy in prevfiles0:
                    if int(ffy.split('.')[0][-2:]) == lastsuffnumber:
                        prevfiles0_latest = ffy
                lastsuffix = prevfiles0_latest.split('.')[0][-3:]
                return lastsuffix
            else:
                return '_-1'
        elif filetype.lower() == 'trf':
            listofPrevChainsNames = find_files_containing(filenamepartToSearchFor , dirnameToSearchIn)
            if len(listofPrevChainsNames) != 0:
                prevfiles0 = []
                for ff in listofPrevChainsNames:
                    if ff[4] == '0':            # <--- 4th letter for TRF file names
                        prevfiles0.append(ff)
                    else:
                        pass
                suffixList = [int(pfile.split('.')[0][-2:]) for pfile in prevfiles0]
                lastsuffnumber = np.max(suffixList)
                for ffy in prevfiles0:
                    if int(ffy.split('.')[0][-2:]) == lastsuffnumber:
                        prevfiles0_latest = ffy
                lastsuffix = prevfiles0_latest.split('.')[0][-3:]
                return lastsuffix
            else:
                return '_-1'
        elif filetype.lower() == 'final_allparams':
            listofPrevChainsNames = find_files_containing(filenamepartToSearchFor , dirnameToSearchIn)
            if len(listofPrevChainsNames) != 0:
                prevfiles0 = listofPrevChainsNames
                suffixList = [int(pfile.split('.')[0][-2:]) for pfile in prevfiles0]
                lastsuffnumber = np.max(suffixList)
                for ffy in prevfiles0:
                    if int(ffy.split('.')[0][-2:]) == lastsuffnumber:
                        prevfiles0_latest = ffy
                lastsuffix = prevfiles0_latest.split('.')[0][-3:]
                return lastsuffix
            else:
                return '_-1'
        elif filetype.lower() == 'inputjson':
            listofPrevChainsNames = find_files_containing(filenamepartToSearchFor , dirnameToSearchIn)
            if len(listofPrevChainsNames) != 0:
                prevfiles0 = listofPrevChainsNames
                suffixList = []
                for pfile in prevfiles0:
                    try:
                        suffixList.append(int(pfile.split('.')[0][-2:]))
                    except:
                        suffixList.append(0)
                lastsuffnumber = np.max(suffixList)
                for ffy in prevfiles0:
                    try:
                        if int(ffy.split('.')[0][-2:]) == lastsuffnumber:
                            prevfiles0_latest = ffy
                        lastsuffix = prevfiles0_latest.split('.')[0][-3:]
                    except:
                        if 0 == lastsuffnumber:
                            prevfiles0_latest = ffy
                        lastsuffix = '_00'
                return lastsuffix
            else:
                return '_-1'
        else:
            raise Exception("Invalid filetype given. Check again.")
    except: 
        return '_-1'
    
    
    
    
    
# =============================================================================
# Param to vary selection
# =============================================================================


def params_to_vary_list_from_input(sysargs, ordered_params_list):
    """
    This function gives a list of names of the parameters to be varied in the current run of
    MCMC based on the input arguments in the submission script

    Parameters
    ----------
    sysargs : TYPE: list
        DESCRIPTION. List of the system arguments for the run in string format.
    ordered_params_list : TYPE: list
        DESCRIPTION. Eg. ['H0', 'Omk', 'Oml', 'w0', 'wa']

    Raises
    ------
    ValueError
        Parameter names in the sys.args not understood.

    Returns
    -------
    params_to_vary : TYPE: list
        DESCRIPTION. Ordered list of the arguments selected for varying in the 
        same order as <ordered_params_list>

    """
    params_to_vary_unsorted = []
    
    
    for p in sysargs:
        try:
            if p.lower() in ['h0','hubble']:
                p = 'H0'
            elif p.lower() in ['omk','ok','omega_k','omegak']:
                p = 'Omk'
            elif p.lower() in ['oml','ol','omega_l','omegal']:
                p = 'Oml'
            elif p.lower() in ['w0']:
                p = 'w0'
            elif p.lower() in ['wa']:
                p = 'wa'
            elif p.lower() in ['hofz','hz','h(z)','hofz','h_of_z','h_ofz']:
                p = 'h(z)'
            elif p.lower() in ['angular_diameter_distance','angular_distance','da','daz','da(z)','daofz','da_of_z','da_ofz']:
                p = 'dA(z)'
            elif p.lower() in ['scale_independent_growth_factor_f','growth_factor','f(z)','fz','fofz','f_of_z','f_ofz']:
                p = 'f(z)'
        except:
            raise ValueError('Valid parameter name(s) not provided in the arguments. Aborting!')
            
        if p in ordered_params_list:
            params_to_vary_unsorted.append(p)
    
    srt = {b: i for i, b in enumerate(ordered_params_list)}
    
    params_to_vary = sorted(params_to_vary_unsorted, key=lambda x: srt[x])
    
    return params_to_vary


