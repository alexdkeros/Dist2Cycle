import math
import random
import itertools
import gudhi as gd
import tadasets
import scipy as sp
import scipy.spatial
import numpy as np



def pinched_torus_grid(npts, R=4, r=2, slack=0.3):
    ''' 
    Grid data from a pinched torus
    args:
        npts
        R: radius of inner hole
        r: radius of cylinder
        slack: size of gap at pinch point
    '''
    rad=np.linspace(0, 2*np.pi,npts)
    Phi,Theta=np.meshgrid(rad,rad)
    X=(R+(np.sin(Phi/2)+slack)*r*np.cos(Theta))*np.cos(Phi)
    Y=(R+(np.sin(Phi/2)+slack)*r*np.cos(Theta))*np.sin(Phi)
    Z=(np.sin(Phi/2)+slack)*r*np.sin(Theta)
    
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    pts=pts.T
    return pts

def pinched_torus(npts, R=4, r=2, slack=0.3):
    ''' 
    Random data from a pinched torus
    args:
        npts
        R: radius of inner hole
        r: radius of cylinder
        slack: size of gap at pinch point
    '''
    phi, theta=np.random.uniform(0,2*np.pi, npts), np.random.uniform(0,2*np.pi, npts)
    Phi,Theta=phi, theta
    X=(R+(np.sin(Phi/2)+slack)*r*np.cos(Theta))*np.cos(Phi)
    Y=(R+(np.sin(Phi/2)+slack)*r*np.cos(Theta))*np.sin(Phi)
    Z=(np.sin(Phi/2)+slack)*r*np.sin(Theta)
    
    pts = np.vstack([X, Y, Z])
    pts=pts.T
    
    return pts
    
    
def multi_holed_tori(n, npts, R=-1, r=-1, pinched=True, cuts=True, noise=0.0):
    '''
    sample from n 2-tori (might be multiple connected components)
    with centers randomly assembed into the 2d plane 
    
    args:
        n : number of holes
        npts: number of points to sample *from each torus*
        R: hole radius (if <0 set to half the distance to the closest point)
        r: tunnel radius (if <0 set to 0.3 the half-dist to the closest point [also ensure R>=r]
        pinched: pinch torus for clear minimizer of cycle
        cuts: bool, cut intersecting tori
        noise: sampling noise
    '''
    
    #sample hole centers
    hole_centers=np.random.uniform(size=(n, 2))
    hole_centers=np.hstack([hole_centers, np.zeros((hole_centers.shape[0],1))])
    D=sp.spatial.distance_matrix(hole_centers, hole_centers)
    D=D+np.diag([np.inf]*D.shape[0])
    #compute min distances
    min_d_indx=np.argmin(D, axis=0)
    min_d=np.min(D,axis=0)
    
    pts=np.array([])
    for c in range(len(hole_centers)):
#         print(f'----center {c}: {hole_centers[c,:]}')
        if R<0.0:
            R=0.9*(min_d[c]/2.0)
        if r<0.0:
            r_c=0.3*(min_d[c]/2.0)
            r_c=min(r_c,R)
        else:
            r_c=min(r,R)
        if pinched:
            pts_t=pinched_torus(npts, R=R, r=r_c, slack=r_c/2)
        else:
            pts_t=tadasets.torus(npts, c=R, a=r_c, noise=noise)
        pts_t=np.asarray(pts_t+np.ones((pts_t.shape[0],1))@np.matrix(hole_centers[c,:]))
        if cuts:
            halfway=(hole_centers[c,:]+hole_centers[min_d_indx[c],:])/2.0
            slope=hole_centers[c,:]-hole_centers[min_d_indx[c],:]
            slope=-1/(slope[1]/slope[0])
            b=halfway[1]-slope*halfway[0]
        #     print(f'line y={slope}x+{b}')
            line=lambda x:slope*x+b
            lp=np.array([0.0, line(0.0), 0.0])    
            outerprod=lambda p: np.cross((halfway-lp), (halfway-p))
            y=np.array([outerprod(pts_t[i,:])[2] for i in range(pts_t.shape[0])])
            
        #     print(f'{outerprod(hole_centers[c,:])}, {outerprod(hole_centers[min_d_indx[c],:])}')
            keepindx=np.argwhere(y*outerprod(hole_centers[c,:])[2]>0)
            pts_t=np.squeeze(pts_t[keepindx,:])
        
        if pts.shape[0]==0:
            pts=pts_t
        else:
            pts=np.vstack([pts, pts_t])
            
    return pts
            
    