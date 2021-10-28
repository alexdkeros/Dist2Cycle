import numpy as np
import numpy.linalg as npla
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla



def neighborhood_smoothing(boundarymat, logits, power=1):
    '''
    smooth logits by neighborhood aggregation
    
    args:
        boundarymat: boundary matrix (dense format)
        logits: signal/logits to smooth
        power: matrix power of graph laplacian
        
    returns:
        smoothed logits
    '''
    
    B0=boundarymat
    
    A=np.abs(B0.T@B0-np.diag(np.diag(B0.T@B0)))
    D=np.diag(np.array(np.sum(A,axis=1)).flatten())
    D_n=np.diag(np.diag(D)**(-1/2))
    Lt=D-A
    Lt=D_n@Lt@D_n
    Lt
    logits_smoothed=logits-npla.matrix_power(Lt,1)@logits
    logits_smoothed=np.array(logits_smoothed).flatten()
    
    return logits_smoothed