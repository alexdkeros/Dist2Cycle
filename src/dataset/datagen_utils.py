import sys
sys.path.append('..')

import os
import random
import time
import itertools
import warnings
import numpy as np
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla
import tadasets
import gudhi as gd

from src.utils.converters import extract_simplices, build_boundaries, build_laplacians




def compute_interesting_f_vals(st_original, intervals, f_val_slices, trivial_coin):
    '''
    compute interesting filtration values (i.e. non-trivial homology) from a simplex tree 
    and given intervals
    
    args:
        st_original: simplex tree
        intervals: intervals from which to select interesting values
        f_val_slices: number of f values to consider
        trivial_coin: ratio of trivial homology instances to accept
        
    returns:
        list of interesting values to consider snapshot of filtration
    '''
    
    #ignore 0 dimensional stuff
    intervals_=[tup for tup in intervals if (tup[0]>0 and not np.isinf(tup[1][1]))]
    
    intervals_.sort(key=lambda tup:tup[1][1]-tup[1][0], reverse=True) #sort according to interval length
    
    
    interval_indx=0
    interesting_f_vals=[]

    while len(interesting_f_vals)<f_val_slices and interval_indx<len(intervals_):
        birth=intervals_[interval_indx][1][0]
        death=intervals_[interval_indx][1][1]
        if birth>0:

            b_birth=st_original.persistent_betti_numbers(birth-0.002,birth-0.001)

            if b_birth[1]==0 and (random.random()<trivial_coin):
                interesting_f_vals.append(birth)
            elif b_birth[1]>0:
                interesting_f_vals.append(birth)

        
        b_death=st_original.persistent_betti_numbers(death-0.002,death-0.001)

        if b_death[1]==0 and random.random()<trivial_coin:
            interesting_f_vals.append(death)
        elif b_death[1]>0:
            interesting_f_vals.append(death)
        
        interval_indx+=1
            
    interesting_f_vals.sort(reverse=True)
    
    return np.unique(interesting_f_vals)




def laplacians_to_features(laplacians, k):
    '''
    spectral embedding of nodes
    '''
    
    l_feats_LM=[]
    l_feats_SM=[]
    l_s_LM=[]
    l_s_SM=[]
    
    skip_cmplx=False
    
    for i,l in enumerate(laplacians):
        print('===== L_%d'%i)
        print(l.shape)
        
        #approximating Laplacian, keeping only k of them
        print('=== approximating %d eigenvecs'%k)
        try:
            print('---- trying sparse ------')
            s,uLM=spspla.eigsh(l.tocsr(), k=min(k,l.shape[1]-1), which='LM')
            s_i=np.argsort(s)
            s_i=s_i[::-1]
            uLM=uLM[:,s_i] #sorted descending order
            sLM=s[s_i]
            
            s,uSM=spspla.eigsh(l.tocsr(), k=min(k,l.shape[1]-1), which='LM', sigma=0.0) 
            # compute SM with shift-invert
            s_i=np.argsort(s)
            s_i=s_i[::-1]
            uSM=uSM[:,s_i] #sorted descending order
            sSM=s[s_i]
            
            l_feats_LM.append(uLM)
            l_feats_SM.append(uSM)
            l_s_LM.append(sLM)
            l_s_SM.append(sSM)
            
            continue
        except Exception as inst:
            #nope,wasn't it
            print('==== SPARSE ERROR ====')
            print(type(inst))
            print(inst.args)
            print(inst)
        try:
            if l.shape[1]>40000:
                raise Exception('Too big of a matrix')
            print('---- trying dense ------')
            u,s,_=np.linalg.svd(l.todense(), hermitian=True) #sorted in descending order
            l_feats_LM.append(u[:, :k])
            l_feats_SM.append(u[:,-k:])
            l_s_LM.append(s[:k])
            l_s_SM.append(s[-k:])
            
            continue
        except Exception as inst:
            #nope,wasn't it
            print('==== DENSE ERROR ====')
            print(type(inst))
            print(inst.args)
            print(inst)
            #better skip that complex altogether
            print('******** SKIPPING COMPLEX ********')
            skip_cmplx=True
            break
            


    return l_feats_LM, l_s_LM, l_feats_SM, l_s_SM, skip_cmplx


def coface_degrees(simplices):
    '''
    compute degrees of simplices as number of cofaces
    '''
    counts=[len(sc) for sc in simplices]
    no_simplices=sum(counts)

    boundaries=build_boundaries(simplices)

    cf_d=[]
    for b in boundaries:
        cf_d.append(np.asarray(np.sum(np.abs(b),axis=1)).flatten().astype(int))

    #top dimensional simplices have no boundary, so
    cf_d.append(np.zeros(len(simplices[-1])))

    return cf_d
    