import sys
sys.path.append('..')
import os

import numpy as np
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla
import tadasets
import gudhi as gd

from src.utils.converters import extract_simplices, build_boundaries, build_laplacians
from src.utils.cycles import shortloop_optimal_cycles, dist_from_cycles, find_optimal_cycles
from src.dataset.datagen_utils import *
from src.dataset.shapes import *

def homology_datagen(dim, 
                    npts_all, 
                    ncomplexes, 
                    f_val_slices, 
                    k,
                    prefix,
                    shortlooppath,
                    Hdim=2, 
                    Cfield=2,
                    complex_type='Alpha',
                    trivial_coin=1.0/80.0,
                    shape_coin=(0.0,0.0,1/4,1/4,1/4,1/4),
                    ntoricopies=(2,10)):
    
    '''
    Generates homology dataset

    args:
        dim: ambient dimension
        npts_all: range/iterable of #pts
        ncomplexes: number of complexes to generate for a specific number of pts
        f_val_slices: number of complex snapshots to consider of the whole persistence range
        k: number of eigenvectors to consider as features
        prefix: folder/file prefixes to save dataset (can be <path/prefix>)
        shortlooppath: path to shortloop executable, for dimensions <=3
        Hdim: homology dimension
        Cfield : homology coefficient field
        complex_type : simplicial complex, choices are ['Alpha', 'Rips']
        trivial_coin: ratio of trivial and nontrivial homology examples
        shape_coin: probabilities ([0,1], sum to 1) of sampling 
                    (uniform, sphere, torus, multi-holed tori, pinched_torus, multi_holed pinched tori)
        ntoricopies: controls number of tori copies for multi-holed tori (min, max)
    '''
    
    
    filtfolder=os.path.abspath('./filtrations')
    filtname='f'
    filtfile=os.path.abspath(filtfolder+'/'+filtname+'.txt')
    loopsfolder=filtfolder+'/'+filtname+'loops'    


    folder=f'{prefix}_{complex_type}_D{dim}_HDim{Hdim}_k{k}'
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        
        
    labels=[0,0] #trivial, nontrivial
    
    skip_count=0
    
    for complex_indx in range(ncomplexes):
        for npts in npts_all:
                
            print('***** Complex ind %d/%d'%(complex_indx, ncomplexes))
            print('***** npts: %d' % npts)
            
            
            #draw random samples uniformly from plane, disk, or torus, multiple tori
            coin=random.random()
            
            if coin<=shape_coin[0]:                                              #uniform samples
                pts=np.random.uniform(size=(npts,dim))
            elif coin<=sum(shape_coin[0:2]):                                    #sphere samples
                pts=tadasets.dsphere(n=npts, d=dim-1, r=1, noise=0.14)
            elif coin<=sum(shape_coin[0:3]):                                    #single torus samples
                if dim>3:
                    pts=tadasets.torus(n=npts, c=2, a=1, ambient=dim, noise=0.2)
                else:
                    pts=tadasets.torus(n=npts, c=2, a=1, noise=0.2)
                    pts=pts[:,0:dim]
            elif coin<=sum(shape_coin[0:4]):                                    #nulti-holed tori samples                                    
                pts=multi_holed_tori(random.randint(*ntoricopies), npts, noise=0.01)
                if dim<3:
                    pts=pts[:,0:dim]
                elif dim>3:
                    pts=np.hstack([pts, np.zeros((pts.shape[0], pts.shape[1]-dim))])
            elif coin<=sum(shape_coin[0:5]):                                    #single pinched torus samples
                    pts=pinched_torus(npts, R=2, r=1, slack=0.2)
                    pts=pts[:,0:dim]
            else:                                                                #nulti-holed pinched tori samples                                    
                pts=multi_holed_tori(random.randint(*ntoricopies), npts,pinched=True, noise=0.01)
                if dim<3:
                    pts=pts[:,0:dim]
                elif dim>3:
                    pts=np.hstack([pts, np.zeros((pts.shape[0], pts.shape[1]-dim))])
                
            #===========================================        
            #build rips complex
            #===========================================
            if complex_type=='Rips':
                cmplx=gd.RipsComplex(points=pts, max_edge_length=dim*np.sqrt(dim))
                st_original=cmplx.create_simplex_tree(max_dimension=Hdim)
            elif complex_type=='Alpha':
                cmplx=gd.AlphaComplex(points=pts)
                st_original=cmplx.create_simplex_tree()

            intervals=st_original.persistence()
            
            
            #===========================================
            #gather interesting complexes from filtration
            #===========================================

            interesting_f_vals=compute_interesting_f_vals(st_original, intervals, f_val_slices, trivial_coin)
            print(interesting_f_vals)
            
            #===========================================
            #for each value get complex and betti vector
            #===========================================
            
    
            for f in interesting_f_vals:
                
               
                print('***** filtration val: %f' % f)
    
                
                # in lieu of "prune_above_filtration()", which segmentation faults
                st=gd.SimplexTree()
                for sc in st_original.get_filtration():
                    if sc[1]<f:
                        st.insert(sc[0], sc[1])
                
                st_intervals=st.persistence()
                
                b_vec=st.betti_numbers()
                while len(b_vec)<Hdim:
                    b_vec.append(0)
                print(f"======HOMOLOGY {b_vec}")
                
                
                #check how balanced the dataset is  #1: trivial, 0:nontrivial
                
                L=int(b_vec[1]==0) #trivial if it has zero H_1
                print('Homology triviality in dataset: %d (1 for trivial) '%L)
                labels[L]+=1
                
                #===========================================
                #===========================================
                #converting data for GNN
                #===========================================
                #===========================================
                
                #===========================================
                # 1. make simplices into vertices (keep both index and simplex representations)
                #===========================================
                
                # sort by dimension of simplex
                simplices=extract_simplices(st)
                
                counts=[len(sc) for sc in simplices]
                no_simplices=sum(counts)
                no_feats=sum(counts[:-1])
                
                #===========================================
                #consecutive indexing of all simplices
                #===========================================
                start_time=time.time()
    
                
                cochains=simplices[0].copy()
                for d in range(1,len(simplices)):
                    for sc, i in simplices[d].items():
                        cochains[sc]=i+sum(counts[:d])
                       
                elapsed_time = time.time() - start_time
                print('Indexed cochains in %f.'%elapsed_time)
                    
                #===========================================
                #assemble laplacians into feature matrix
                #===========================================
                boundaries=build_boundaries(simplices)
                laplacians=build_laplacians(boundaries)
                
                print('Laplacians:')
                print([l.shape for l in laplacians])
                
                start_time=time.time()
                
                l_feats_LM, l_s_LM, l_feats_SM, l_s_SM,  skip_cmplx = laplacians_to_features(laplacians, k)
                
                if skip_cmplx:
                    #SVD failed, skip complex
                    labels[L]-=1
                    skip_count+=1
                    continue
                
                elapsed_time = time.time() - start_time
                print('Laplacian feats in %f'%elapsed_time)
                
                
                #===========================================
                # degrees
                #===========================================
            
                start_time=time.time()
                
                coface_deg=coface_degrees(simplices)
                coface_deg=np.hstack(coface_deg)
                

                
                elapsed_time = time.time() - start_time
                
                print('Degrees in %f'%elapsed_time)
                
                #===========================================
                #finding cycles
                #===========================================
                start_time = time.time()

                
                cyclesSimplices, cyclesIndices, skip_cmplx=find_optimal_cycles(pts, simplices, 
                                                                            shortlooppath, filtfolder)
                        
                if skip_cmplx:
                    #screwed up ShortLoop, skip complex
                    labels[L]-=1
                    skip_count+=1
                    continue
                
                print(cyclesSimplices)
                print(cyclesIndices)
                
                
                elapsed_time = time.time() - start_time
                print('Cycle Deys in %f'%elapsed_time)
            
                
                #===========================================
                #mark cycles
                #===========================================
                
                on_cycle_labels=np.array([0]*no_simplices)
                
                #mark simplex on cycle or not
                for cy in cyclesSimplices:
                    for sc in cy:
                        on_cycle_labels[cochains[sc]]=1
                    
                #===========================================
                #distance from cycles
                #===========================================
                                    
                start_time = time.time()


                                
                simplex_distances, simplex_distances_all=dist_from_cycles(st, simplices, cyclesSimplices)
            
                #flatten simplex_distances
                simplex_distances=np.array([item for sublist in simplex_distances for item in sublist])
                #fix simplex_distances_all
                simplex_distances_all=[np.vstack(s_d_a) for s_d_a in simplex_distances_all]
                simplex_distances_all=np.vstack(simplex_distances_all)

    
                elapsed_time = time.time() - start_time
                print('Cycle labels in %f'%elapsed_time)
                                
                #===========================================
                #===========================================
                
                
                complex_name=f'pts_{npts}_filtV_{f}_cid_{complex_indx}'
                    
                
                print('--------- sanity checks ---------------')
                print(f'cycles:{len(cyclesSimplices)}')
                print(f'{complex_name}')
                print('cochains:%d'%len(cochains))

                
                    
                print(f'Laplacians LM shape: {[lf.shape for lf in l_feats_LM]}')
                print(f'Laplacians SM shape: {[lf.shape for lf in l_feats_SM]}')
                    
                        
                if len(cochains)!=sum([lf.shape[0] for lf in l_feats_LM]) or len(cochains)!=sum([lf.shape[0] for lf in l_feats_SM]):
                    warnings.warn('Dimension missmatch between features')
                    #screwed up dimensions, skip complex
                    labels[L]-=1
                    skip_count+=1
                    continue

                if len(cyclesSimplices)!=b_vec[1]:
                    warnings.warn('Persistence mismatch. Betti vector of Gudhi and persistent cycles fo Shortloop/Persloop do not agree')
                    #screwed up homology, skip complex
                    labels[L]-=1
                    skip_count+=1
                    continue
                        
                if any(np.isnan(simplex_distances)):
                    raise Exception('Simplex distance is nan.')
                    #distances are false, skip complex
                    labels[L]-=1
                    skip_count+=1
                    continue
                    
                print('---------')
                

                np.save(f'{folder}/{complex_name}_pts.npy', pts)
                np.save(f'{folder}/{complex_name}_cochains.npy', cochains)
                np.save(f'{folder}/{complex_name}_simplices.npy', simplices)
                np.save(f'{folder}/{complex_name}_bettiVec.npy', b_vec)
                        
                np.save(f'{folder}/{complex_name}_laplaciansLM.npy', l_feats_LM)
                np.save(f'{folder}/{complex_name}_laplaciansSM.npy', l_feats_SM)
                np.save(f'{folder}/{complex_name}_laplaciansLMevals.npy', l_s_LM)
                np.save(f'{folder}/{complex_name}_laplaciansSMevals.npy', l_s_SM)
                
                
                np.save(f'{folder}/{complex_name}_cycleLabel.npy', on_cycle_labels)
                np.save(f'{folder}/{complex_name}_cyclesIndices.npy', cyclesIndices)
                np.save(f'{folder}/{complex_name}_cyclesSimplices.npy', cyclesSimplices)
        
                np.save(f'{folder}/{complex_name}_minCycleDistance.npy', simplex_distances)
                np.save(f'{folder}/{complex_name}_allCycleDistances.npy', simplex_distances_all)
                
                
                
                
                print(f'nontrivial/trivial:{labels}')
                if labels[0]>0:
                    print(f'ratio:{labels[1]/labels[0]} (trivial/nontrivial)')

    print('!!!! Skipped %d complexes !!!!'%skip_count)