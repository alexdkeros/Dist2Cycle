import sys
sys.path.append('..')
import os
import itertools
import time
import subprocess
import numpy as np
import gudhi as gd
from src.utils.complex import get_full_subcomplex, get_subcomplex_star, get_subcomplex_closed_star, get_subcomplex_link

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
        
      
      
      
def find_optimal_cycles(pts, simplices, shortlooppath, filtfolder , delete_files=True):
    '''
    find tight 1-cycles in complex via shortloop 
    '''
    
    counts=[len(sc) for sc in simplices]
    no_simplices=sum(counts)
    no_feats=sum(counts[:-1])
    
    pts=np.array(pts)
    
    skip_cmplx=False
    trials=3
    
    if pts.shape[1]<=3: #ShortLoop cannot handle higher dimensions
        #===========================================
        #ShortLoop
        #===========================================

        pts3D=pts
        while pts3D.shape[1]<3:
            pts3D=np.hstack((np.zeros((pts3D.shape[0],1)), pts3D))
            
        while trials>0:
            try:
                print('Running SHORTloop.')
                cyclesSimplices, cyclesIndices=shortloop_optimal_cycles(pts3D,
                                                    simplices,
                                                    shortlooppath,
                                                    filtfolder,
                                                    verbose=False, delete_files=delete_files)
                trials=0
                continue
            except Exception as inst:
                    #nope,wasn't it
                    print('==== SHORTLOOP ERROR ====')
                    print(type(inst))
                    print(inst.args)
                    print(inst)
                    trials-=1
                    if trials==0:
                        print('******** SKIPPING COMPLEX ********')
                        skip_cmplx=True
                        return [],[], skip_cmplx
    
                    
    return cyclesSimplices, cyclesIndices, skip_cmplx

  

def shortloop_optimal_cycles(pts,
                            simplices,
                            shortlooppath,
                            filtfolder,
                            filtname='f',
                            delete_files=True,
                            verbose=False):
    
    '''
    args:
        pts: coordinates of points (N,D) array
        simplices: list of dictionaries, one dict per dimension, {frozenset({v1, v2, ..., vk}):<index_in_boundary_mat_col>, ...}
        perslooppath: path of persloop executable
        filtfolder: folder for filtration file to be written
        filtname: name of filtration file to be written 
              (filtration file is read by shortloop)
        delete_names: bool, delete filtration files

    returns:
        cyclesP: list of lists of edges [ [ frozenset({v1,v2}), frozenset({v2, v3}), ..., frozenset({vk, v1} ], ... ] one list per cycle
        cyclesEindx: list of lists of indices of edge simplices, according to provided "simplices" list of dicts
    '''
    folder=filtfolder
    
    #if the default filtname, make it unique
    if filtname=='f':
        filtname=filtname+str(time.monotonic_ns())
    
    filtfile=os.path.abspath(folder+'/'+filtname+'.txt')
    loopsfile=os.path.abspath(folder+'/'+filtname+'_loops.txt')
    off_loopsfile=os.path.abspath(folder+'/'+filtname+'_loops.off')
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    #write filtration to file
    ptsdim=len(pts[0])
    pts_count=len(pts)
    
    
    fff=[]
    if len(simplices)>2:
        for sc in simplices[2].keys():
            for fsc in itertools.combinations(sc, 2):
                fff.append(frozenset(fsc))
        fff=list(set(fff))
    
    with open(filtfile,'w') as filt:
        
        #dims npts
        filt.write('OFF\n')
        
        if len(simplices)>2:
            n_faces=len(simplices[2])+(len(simplices[1])-len(fff))
        elif len(simplices)>1:
            n_faces=len(simplices[1])
        else:
            n_faces=0
            
        filt.write(f'{pts_count} {n_faces} 0\n')

        #coords
        for i in range(pts_count):
            s=' '.join([str(c) for c in pts[i]])
            filt.write(f'{s}\n')
        
        if len(simplices)>1:
            for sc in simplices[1].keys():
                if sc not in fff:
                    s=' '.join([str(v) for v in sc])
                    filt.write(f'{len(sc)} {s}\n')
        if len(simplices)>2:
            for sc in simplices[2].keys():
                s=' '.join([str(v) for v in sc])
                filt.write(f'{len(sc)} {s}\n')
        
    #run shortloop
    with cd(shortlooppath):
        result=subprocess.run(['./ShortLoop', filtfile, '-v'], capture_output=True, check=True)
    if verbose:
        print(result.stdout.decode("utf-8") )
   
    #get loops and simplex indices
    loop_count=0
    cyclesP=[]
    with open(loopsfile) as cf:
        for line_num, line in enumerate(cf):
            if line_num==0:
                l=line.split()
                if l[1]=='loops':
                    loop_count=int(l[0])
            else:
                if loop_count==0:
                    break
                if loop_count>0:
                    l=line.split()
                    if l[0]=='Loop':
                        cycle=l[5:]
                        cycle=list(map(int,cycle))
                        cyclesP.append(cycle)
    
    cycleEdges=[]
    for cycle in cyclesP:
        cycleE=[]
        for i in range(0,len(cycle)): #goes up to len(cycle)
            cycleE.append( frozenset([cycle[i], cycle[(i+1)%len(cycle)]]) )
        cycleEdges.append(cycleE)
    
    cyclesEindx=[]
    for cycle in cycleEdges:
        cycleE=[simplices[1][el] for el in cycle]
        cyclesEindx.append(cycleE)
        
    if delete_files:
        if os.path.isfile(filtfile):
            os.remove(filtfile)
        if os.path.isfile(loopsfile):
            os.remove(loopsfile)
        if os.path.isfile(off_loopsfile):
            os.remove(off_loopsfile)
            
    
    return cycleEdges, cyclesEindx





def dist_from_cycles(st, simplices, cyclesP, weighted=False):
    '''
    if no cycles, all distances are set to 1
    minimum distance to H_1 (to closest generator, if multiple)
    
    args:
        st: gudhi simplex tree
        simplices: list of dictionaries, one dict per dimension, {frozenset({v1, v2, ..., vk}):<index_in_boundary_mat_col>, ...}
        cyclesP: (output of persloop_optimal_cycles() : list of lists of edges [ [ frozenset({v1,v2}), frozenset({v2, v3}), ..., frozenset({vk, v1} ], ... ] one list per cycle
        weighted: if True, weight distances by cycle length
    returns:
        s_dist: list of lists of normalized distances [0,1] 0 closes, 1 furthest , one list per dimension 
        s_dist_all: list of lists of lists of normalized distances[0,1], one list per dimension, each simplex its own list of distances from all cycles (i.e. len(cyclesP))
    '''
    
    if len(cyclesP)==0: #no cycles, distance is 1.0 for all simplices (maximum possible)
        s_dist=[]
        for d in range(len(simplices)):
            s_dist.append([1.0]*len(simplices[d]))
        return s_dist, s_dist

    #sort cyclesP by descending length
    cyclesP.sort(key=len, reverse=True)
    
    #weights of cycles
    total_w=sum(np.unique([len(c) for c in cyclesP]))
    cycle_w=[len(c)/total_w for c in cyclesP]
    
    
    sD={}
    sD_all={}
    #initialize distances data structure
    for d in range(len(simplices)):
        for s in simplices[d].keys():
            sD[s]=np.inf
            sD_all[s]=np.ones(len(cyclesP))*np.inf
        
            
    for ci,cycle in enumerate(cyclesP):
        
        dist=0.0
        subc=cycle
        
        full_subc=get_full_subcomplex(subc)
        
        updated=[True]

        while any(updated):

            
            updated=[]
            
            for s in full_subc:
                if weighted:
                    if not cycle_w[ci]*dist<sD[s]:
                        updated.append(False)
                    else:
                        updated.append(True)
                        sD[s]=cycle_w[ci]*dist
                    
                    if cycle_w[ci]*dist<sD_all[s][ci]:
                        sD_all[s][ci]=cycle_w[ci]*dist  
                
                else:
                    if not dist<sD[s]:
                        updated.append(False)
                    else:
                        updated.append(True)
                        sD[s]=dist

                    if dist<sD_all[s][ci]:
                        sD_all[s][ci]=dist  

            dist+=1.0
            scA=list(set(get_subcomplex_star(st,full_subc))-set(full_subc))
            
    

            for s in scA:
                if weighted:
                    if not cycle_w[ci]*dist<sD[s]:
                        updated.append(False)
                    else:
                        updated.append(True)
                        sD[s]=cycle_w[ci]*dist
                    
                    if cycle_w[ci]*dist<sD_all[s][ci]:
                        sD_all[s][ci]=cycle_w[ci]*dist  
                
                else:
                    if not dist<sD[s]:
                        updated.append(False)
                    else:
                        updated.append(True)
                        sD[s]=dist

                    if dist<sD_all[s][ci]:
                        sD_all[s][ci]=dist
                        
            dist+=1
            scB=get_subcomplex_link(st, full_subc)
            full_subc=scB
 


    #normalizing minimum distance from H_1
    s_dist=[]
    for d in range(len(simplices)):
        s_dist.append([0]*len(simplices[d]))

    d_vals=sorted(np.unique(list(sD.values())), reverse=True)
    d_v=np.array(d_vals)
    ninf_vals=d_v[~np.isinf(d_vals)]
    if len(ninf_vals)>0:
        max_noninf=np.max(ninf_vals)
    else:
        max_noninf=0.0
    for s,dist in sD.items():
        if max_noninf>0:
            if dist is np.inf:
                s_dist[len(s)-1][simplices[len(s)-1][s]]=max_noninf/max_noninf
            else:
                s_dist[len(s)-1][simplices[len(s)-1][s]]=dist/max_noninf
        else:
            if dist==0.0:
                s_dist[len(s)-1][simplices[len(s)-1][s]]=0.0
            else:
                s_dist[len(s)-1][simplices[len(s)-1][s]]=1.0



    #normalizing all distances from H_1
    s_dist_all=[]
    for d in range(len(simplices)):
        s_dist_all.append([[0]*len(cyclesP)]*len(simplices[d]))

    d_vals=np.array(list(sD_all.values()))
    d_vals[np.isinf(d_vals)]=np.nan
    max_noninf=np.nanmax(d_vals, axis=0)
    
    for s,dist in sD_all.items():
        dist_all_c=np.array(dist)
        for di,e in enumerate(max_noninf):
            if e>0:
                if np.isinf(dist_all_c[di]):
                    dist_all_c[di]=1.0
                else:
                    dist_all_c[di]=dist_all_c[di]/e
            else:
                if np.isinf(dist_all_c[di]):
                    dist_all_c[di]=1.0
                else:
                    dist_all_c[di]=0.0
                
        s_dist_all[len(s)-1][simplices[len(s)-1][s]]=dist_all_c  
    
    
    return s_dist, s_dist_all



def link_homology(simplices, k, max_dim=3):
    ''' compute the homology of the link for all k-dim simplices in st
    
    k: k-simplices
    max_dim: max homology dimension
    '''
    link_hom=[]
    
    st=gd.SimplexTree()
    for s_d in simplices:
        for c,i in sorted(s_d.items(), key=lambda x: x[1]):
            st.insert(c)
    
    for simpx,indx in simplices[k].items():
        lk=get_subcomplex_link(st, [simpx])
        lk_st=gd.SimplexTree()
        for i,c in enumerate(sorted(lk,key=len)):
            lk_st.insert(c, filtration=0.0)
        lk_st.persistence(min_persistence=0.0, homology_coeff_field=2, persistence_dim_max=max_dim)
        simpx_lk_hom=np.array(lk_st.persistent_betti_numbers(from_value=0.0, to_value=0.1))
        if len(simpx_lk_hom)<max_dim:
            simpx_lk_hom=np.hstack([simpx_lk_hom, np.zeros(max_dim-len(simpx_lk_hom))])
        elif len(simpx_lk_hom)>max_dim:
            simpx_lk_hom=simpx_lk_hom[:max_dim]
        link_hom.append(simpx_lk_hom)
    return link_hom
   
  