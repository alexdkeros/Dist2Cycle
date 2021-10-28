import numpy as np
import numpy.linalg as npla
import phat as pm
import gudhi as gd
import itertools
import scipy.sparse as spsp
from scipy.sparse import coo_matrix

#=============== stars and links simplicial complex simplex tree operations ===========================
def get_full_subcomplex(subc):
    fc=set()
    for sc in subc:
        if type(sc) is int:
            fc.add(frozenset([sc]))
        else:
            fc.add(frozenset(sc))
            for i in list(range(len(sc)-1,0,-1)):
                for b in itertools.combinations(sc,i):
                    fc.add(frozenset(b))
    return list(fc)
        
def get_subcomplex_star(st, subc):
    '''
    args:
        st: simplex tree
        subc: subcomplex as a list of complexes
    '''
    subc_star=set()
    for s in subc:
        if st.find(s):
            for ss in st.get_star(s):
                subc_star.add(frozenset(ss[0]))
    return list(subc_star)

def get_subcomplex_closed_star(st, subc):
    
    star=get_subcomplex_star(st, subc)
    return get_full_subcomplex(star)
        
def get_subcomplex_link(st, subc):
    
    star=get_subcomplex_star(st, subc)
    clstar=get_subcomplex_closed_star(st, subc)
    star=set(star)
    clstar=set(clstar)
    return list(clstar-star)
#======================================================================================================
