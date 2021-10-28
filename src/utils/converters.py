import numpy as np
import numpy.linalg as npla
import phat as pm
import gudhi as gd
import itertools
import scipy.sparse as spsp
from scipy.sparse import coo_matrix



def compute_Ltilde_pinv(L, ltype='func', power=20):
    '''
    compute laplacian shifted pseudoinverse (true or approximation)
    and return the laplacian, along with eigenvectors and eigenvalues
    
    normdiag computes I-aD^-1/2 L D^-1/2, which approximates the inverted spectrum while preserving
    laplacian sparsity, so that no summant information is lost at message passing
    
    args:
        L: original laplacian
        ltype: laplacian approximation type ['func', 'approx', 'norm']
        power: in case of 'approx', which is Neumann approximation, power of series
        
    returns:
        L_tilde_pinv: L tilde inverse (the shifted inverted laplacian)
        V: eigenvectors of L_tilde_pinv
        S: eigenvalues of L_tilde_pinv
    '''
    #consider using npla.sparse.svds for pinv computation if this takes too long/or too much space
    try:
        
        if ltype=='norm':
            D=np.diag(np.diag(L)**(-1/2))
            Ltilde_norm=np.eye(L.shape[0])-(1/npla.norm(D@L@D,2))*D@L@D
            L_tilde_pinv=Ltilde_norm
        elif ltype=='approx':
                L_norm=L/npla.norm(L)
                L_tilde_pinv=np.eye(L_norm.shape[0])
                for i in range(1,power+1):
                    L_tilde_pinv+=npla.matrix_power(-L_norm,i)
        elif ltype=='func':
                L_tilde_pinv=npla.pinv(L+np.eye(L.shape[0]))
        
        
        V, S, U = npla.svd(L_tilde_pinv)
    
        return L_tilde_pinv, V, S
    

    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        
        return None, None, None
        
    

def build_laplacians(boundaries):
    """Build the Laplacian operators from the boundary operators.

    args:
        boundaries: list of sparse matrices
           List of boundary operators, one per dimension.


    returns:
        laplacians: list of sparse matrices
           List of Laplacian operators, one per dimension: laplacian of degree i is in the i-th position
    """
    laplacians = list()
    up = coo_matrix(boundaries[0] @ boundaries[0].T)
    laplacians.append(up)
    for d in range(len(boundaries)-1):
        down = boundaries[d].T @ boundaries[d]
        up = boundaries[d+1] @ boundaries[d+1].T
        laplacians.append(coo_matrix(down + up))
    down = boundaries[-1].T @ boundaries[-1]
    laplacians.append(coo_matrix(down))
    return laplacians



def build_boundaries(simplices):
    """Build the boundary operators from a list of simplices.

    args:
        simplices: list of dictionaries
            List of dictionaries, one per dimension d. The size of the dictionary
            is the number of d-simplices. The dictionary's keys are sets (of size d
            + 1) of the 0-simplices that constitute the d-simplices. The
            dictionary's values are the indexes of the simplices in the boundary
            and Laplacian matrices.

    returns:
       boundaries: list of sparse matrices
           List of boundary operators, one per dimension: i-th boundary is in i-th position
    """
    boundaries = list()
    for d in range(1, len(simplices)): #for each dimension (above the 0-th)
        idx_simplices, idx_faces, values = [], [], []
        for simplex, idx_simplex in simplices[d].items(): #([v1,v2,v3, ..., vk], indx)
            for i, left_out in enumerate(np.sort(list(simplex))): #go through sorted vertices of simplex
                idx_simplices.append(idx_simplex)
                values.append((-1)**i) #sign comes from this (its the coefficient of the boundary op)
                face = simplex.difference({left_out}) # [v1,v2,...,\hat{vi}, ...] to produce faces
                idx_faces.append(simplices[d-1][face])
        assert len(values) == (d+1) * len(simplices[d])
        boundary = coo_matrix((values, (idx_faces, idx_simplices)),
                                     dtype=np.float32,
                                     shape=(len(simplices[d-1]), len(simplices[d])))
        boundaries.append(boundary)
    return boundaries




def extract_simplices(simplex_tree):

    """Create a list of simplices from a gudhi simplex tree.

    args:
        simplex_tree: gudhi simplex tree
        
    returns:
        list of simplices
    """
    simplices = [dict() for _ in range(simplex_tree.dimension()+1)]
    for simplex, _ in simplex_tree.get_skeleton(simplex_tree.dimension()):
        k = len(simplex)
        simplices[k-1][frozenset(simplex)] = len(simplices[k-1])
    return simplices


