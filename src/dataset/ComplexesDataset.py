import os
import time
import random
import itertools
import numpy as np
import numpy.linalg as npla

import copy

import warnings

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info

from src.utils.converters import extract_simplices, build_boundaries, build_laplacians, compute_Ltilde_pinv
from src.utils.cycles import link_homology



class PARAMS():
    labels=['min_dist_from_cycles']
    adjacency=['boundary', 'laplacian', 'complete']
    Ltildetype=['func', 'approx','norm', 'original', 'shiftedOriginal']



class ComplexesDataset(DGLDataset):
    '''
    dataset of simplicial complexes for "Topology-aware learning on simplicial complexes via Spectrum Filtering Graph Neural Networks"
    processes/loads the whole dataset
    '''
    def __init__(self,
                 raw_dir,
                 save_dir,
                 saved_dataname=None,
                 label_type='min_dist_from_cycles',
                 dataset_name='ComplexDataset',
                 feats=None,
                 mode='train',
                 adjacency='laplacian',
                 Ldim=1,
                 max_k_get=None,
                 max_k_store=None,
                 Ltype='func',
                 LapproxPower=5,
                 traincut=0.0,
                 valcut=0.0,
                 testcut=0.0,
                 url=None,
                 force_reload=False,
                 verbose=False):
        '''
        saved_dataname: looks directly for a dataset to load, so it must be the full dataset folder, not just the prefix, as in dataset_name
        '''
        
        self.graphs=[]
        self.labels=[]
        self.graphnames=[]
        
        self._blacklist=[] #graphnames of graphs that failed to load properly, to cleanup before saving processed dataset
        
        self.force_reload=force_reload
        
        self.Ldim=Ldim #laplacian dimension to build graph from
        if not Ltype in PARAMS.Ltildetype:
            warnings.warn('No Ltype selected, defaulting to func')
        self.Ltype=Ltype # aproximate Ltilde
        self.LapproxPower=LapproxPower # power for laplacian approximation
        
        self.max_k_get=max_k_get #limit the size of data on each node and edge
        self.max_k_store=max_k_store # limit the size of saved k
        
        if (self.max_k_store is not None) and (self.max_k_get is not None) and (self.max_k_get>self.max_k_store):
            warnings.warn(f'Requesting more k ({self.max_k_get}) than what is stored  ({self.max_k_store})!')
        
        
        self.label_type=label_type 
        
        if not adjacency in PARAMS.adjacency:
            warnings.warn('No adjacency selected, defaulting to boundary (more expressive)')
        self.adjacency=adjacency 
        
        self.feats=feats
        
        self.mode=mode
        
        #splitting dataset proportions
        if traincut>=1:
            self.traincut=int(traincut)
        else:
            self.traincut=traincut
        if valcut>=1:
            self.valcut=int(valcut)
        else:
            self.valcut=valcut
        if testcut>=1:
            self.testcut=int(testcut)
        else:
            self.testcut=testcut
        
        dataset_name=dataset_name+'_'+adjacency+'_'+str(int(self.Ldim))+'HD'+'_L'+self.Ltype
        
        if self.max_k_store is not None:
            dataset_name=dataset_name+'_k'+str(self.max_k_store)
        
        dataset_name=dataset_name+'_'+str(self.traincut)+'_'+str(self.valcut)+'_'+str(self.testcut)
        
        if saved_dataname is not None:
            #should bypass everything and straight up load the dataset directly, or save the dataset with a unique, rather than a composed, name
            dataset_name=saved_dataname
            print('@@@ IGNORE PREVIOUS WARNINGS OF MISSING ARGS @@@')


        super(ComplexesDataset, self).__init__(name=dataset_name,
                                             url=url,
                                             raw_dir=raw_dir,
                                             save_dir=save_dir,
                                             force_reload=force_reload,
                                             verbose=verbose)
        
    def download(self):
        # download raw data to local disk
        print("No link, loading from storage.")
        pass
    
    def process(self):
        # process raw data to graphs, labels, splitting masks
        print('Processing dataset...')
        start_time=time.time()
        
        #===========================================
        #===========================================
        #get graphs from files and assemble in list of graphs with labels
        #===========================================
        #===========================================
        
        
        #===========================================
        ### 1. read data names from folder
        #===========================================
        
        #data names
        datafiles=os.listdir(self.raw_dir)
        datanames=[]
        for t in datafiles:
            #skip folders
            if os.path.isdir(self.raw_dir+'/'+t):
                continue
            s=t.split('.')
            if s[-1]=='npy':
                s=t.split('_')
                datanames.append('_'.join(s[:-1]))
        datanames=sorted(np.unique(datanames)) #sort them alphabetically for consistency
        print('Number of files/complexes: %d:'%len(datanames))
        
        #===========================================
        ### 2. split to train,val, test sets
        #===========================================
        
        #split datasets with representative proportions of trivial and non-trivial homology examples
        label_zeros=[]
        label_ones=[]
        for name in datanames:
            L=np.load(self.raw_dir+'/'+name+"_bettiVec.npy", allow_pickle=True)
            Hdim=len(L)
            
            #classifying trivial or nontrivial according to dimension
            if self.Ldim==0:
                L=(L[0]==1).astype(int)
            else:
                L=(L[self.Ldim]==0).astype(int)
            
            if L==0:
                label_zeros.append(name)
            else:
                label_ones.append(name)
                
        random.seed(666) #fix seed so selections and shuffles are consistent for datasets of different features but on the same complexes
        random.shuffle(label_zeros)
        random.shuffle(label_ones)
        
        print(f'Nontrivial: {len(label_zeros)}, Trivial: {len(label_ones)}')
        
        if self.traincut<1:
            #compute number of examples in each set
            t_zeros_split=int(np.floor(self.traincut*len(label_zeros)))
            t_ones_split=int(np.floor(self.traincut*len(label_ones)))

        else:
            t_zeros_split=min(np.round(self.traincut*(len(label_zeros)/len(datanames))),len(datanames))
            t_ones_split=min(np.round(self.traincut*(len(label_ones)/len(datanames))),len(datanames))
            
        if self.valcut<1:
            #compute number of examples in each set
            v_zeros_split=int(np.floor(self.valcut*len(label_zeros)))
            v_ones_split=int(np.floor(self.valcut*len(label_ones)))

        else:
            v_zeros_split=min(np.round(self.valcut*(len(label_zeros)/len(datanames))),len(datanames))
            v_ones_split=min(np.round(self.valcut*(len(label_ones)/len(datanames))),len(datanames))
            
        if self.testcut<1:
            #compute number of examples in each set
            te_zeros_split=int(np.floor(self.testcut*len(label_zeros)))
            te_ones_split=int(np.floor(self.testcut*len(label_ones)))

        else:
            te_zeros_split=min(np.round(self.testcut*(len(label_zeros)/len(datanames))),len(datanames))
            te_ones_split=min(np.round(self.testcut*(len(label_ones)/len(datanames))),len(datanames))

        
        
        #make them into indices
        t_zeros_split=int(t_zeros_split)
        v_zeros_split=int(v_zeros_split+t_zeros_split)
        te_zeros_split=int(te_zeros_split+v_zeros_split)
        
        t_ones_split=int(t_ones_split)
        v_ones_split=int(v_ones_split+t_ones_split)
        te_ones_split=int(te_ones_split+v_ones_split)


        #assign filenames in each set
        train_names=label_zeros[:t_zeros_split]+label_ones[:t_ones_split]
        val_names=label_zeros[t_zeros_split:v_zeros_split]+label_ones[t_ones_split:v_ones_split]
        test_names=label_zeros[v_zeros_split:te_zeros_split]+label_ones[v_ones_split:te_ones_split]
        
        random.shuffle(train_names)
        random.shuffle(val_names)
        random.shuffle(test_names)
        
        print(f'train: {len(train_names)}')
        print(f'val: {len(val_names)}')
        print(f'test: {len(test_names)}')
        
        #===========================================
        ### 3. load graphs or save cuts for later
        #===========================================
        
        #because _process_and_save loads to self the 'mode' dataset, run last
        if self.mode=='train':
            if not self._has_cache('val') or self.force_reload:
                self._process_and_save(val_names, 'val')
                self._cleanup()
            if not self._has_cache('test') or self.force_reload:
                self._process_and_save(test_names, 'test')
                self._cleanup()
            self._process_and_save(train_names, 'train')
        elif self.mode=='val':
            if not self._has_cache('train') or self.force_reload:
                self._process_and_save(val_names, 'train')
                self._cleanup()
            if not self._has_cache('test') or self.force_reload:
                self._process_and_save(test_names, 'test')
                self._cleanup()
            self._process_and_save(val_names, 'val')
        elif self.mode=='test':
            if not self._has_cache('val') or self.force_reload:
                self._process_and_save(val_names, 'val')
                self._cleanup()
            if not self._has_cache('train') or self.force_reload:
                self._process_and_save(test_names, 'train')
                self._cleanup()
            self._process_and_save(test_names, 'test')
    
    def _cleanup(self):
        '''
        clean up data structures
        '''
        del self.graphs[:]
        del self.labels[:]
        del self.graphnames[:]
        del self._blacklist[:]
    
    def _process_and_save(self, datasetnames, dstype):    
        print('PROCESS AND SAVE: ===== %s ====='% dstype)
        self.graphnames=datasetnames
        self.graphs=self._load_graphs(self.graphnames)
        if len(self._blacklist)>0:
            for gn in self._blacklist:
                self.graphnames.remove(gn)
        self.labels=self._load_labels(self.graphnames, return_list=True)
        if self.feats is not None:
            self._load_features(self.graphnames, self.graphs)
        self._save_mode(dstype)
        
        return self.graphnames, self.graphs, self.labels
    
    
    
    def _load_graphs(self,datasetnames):
        if not type(datasetnames) is list:
            datasetnames=[datasetnames]            
        graphs=[]
        for data_indx, name in enumerate(datasetnames):
            
            if data_indx%10==0:
                print(f'file {data_indx}/{len(datasetnames)} ({round(100*data_indx/len(datasetnames),1)}%)')
                
            #first add all nodes with self-weights (indexing in graph will be that of simplices dict)
            simplices=np.load(self.raw_dir+'/'+name+'_simplices.npy', allow_pickle=True)
            
            boundaries=build_boundaries(simplices)
            laplacians=build_laplacians(boundaries)
            lk_hom=link_homology(simplices, k=1, max_dim=3)
            
            
            B=boundaries[self.Ldim-1].todense()
            L=laplacians[self.Ldim].todense()
            
            
            if self.adjacency=='boundary':
                Adj=B.T@B
            elif self.adjacency=='laplacian':
                Adj=L
            elif self.adjacency=='complete':
                Adj=np.ones(L.shape)
            
            if self.Ltype=='original':
                L_tilde_pinv=L
                V,S,_=npla.svd(L)
            elif self.Ltype=='shiftedOriginal':
                L_tilde_pinv=L+np.eye(L.shape[0])
                V,S,_=npla.svd(L+np.eye(L.shape[0]))
            else:
                L_tilde_pinv, V, S=compute_Ltilde_pinv(L, ltype=self.Ltype, power=self.LapproxPower)
            
            if (L_tilde_pinv is None) or (V is None) or (S is None):
                warnings.warn('Computing pinv and svd failed, skipping this graph')
                self._blacklist.append(name)
                continue
            
            if self.max_k_store is not None:
                    
                if self.max_k_store>len(S):
                    #pad with zeros
                    V=np.hstack([V, np.zeros((V.shape[0], self.max_k_store-V.shape[1]))])
                    S=np.hstack([S, np.zeros(self.max_k_store-len(S))])
                
                V=V[:,:self.max_k_store]
                S=S[:self.max_k_store]
            
            
            g = dgl.DGLGraph()
            g.add_nodes(L.shape[0])
            #eigenvector embeddings as features of nodes
            g.ndata['V'] = torch.tensor(V).float()
            g.ndata['lk_hom']=torch.tensor(lk_hom).float()
            
            edge_source=[]
            edge_dest=[]
            edge_w=[]
            edge_S=[]
            for i in range(Adj.shape[0]):
                for j in range(i,Adj.shape[1]):
                    if i==j:
                        #self loop edge
                        #pinv laplacian entries as weights
                        #eigenvalues of pinv as additional features of edge
                        edge_source.append(i)
                        edge_dest.append(i)
                        edge_w.append([L_tilde_pinv[i,i]])
                        edge_S.append(S)
                    elif Adj[i,j]!=0.0:
                        #add both directions for undirected graph!
                        #pinv laplacian entries as weights
                        #eigenvalues of pinv as additional features of edge
                        edge_source.append(i)
                        edge_dest.append(j)
                        edge_w.append([L_tilde_pinv[i,j]])
                        edge_S.append(S)
                        
                        edge_source.append(j)
                        edge_dest.append(i)
                        edge_w.append([L_tilde_pinv[j,i]])
                        edge_S.append(S)
                
            g=dgl.add_edges(g, torch.tensor(edge_source) ,torch.tensor(edge_dest), 
                            {'w':torch.tensor(edge_w).float(), 'S':torch.tensor(edge_S).float()})
            
            print(f'Nodes: {g.num_nodes()}, Edges : {g.num_edges()}')

            graphs.append(g)

        if len(graphs)==1:
            return graphs[0]
        else:
            return graphs
        
    def _load_labels(self,datasetnames, return_list=False):
        #load labels for given set of files
        if not type(datasetnames) is list:
            datasetnames=[datasetnames]
        
        labels=[]
        for name in datasetnames:
            
            cochains=np.load(self.raw_dir+'/'+name+"_cochains.npy", allow_pickle=True).item()
            #get indices corresponding to dimension we are interested in
            co_sorted=sorted(cochains.items(), key=lambda t:t[1])
            co_sel=[co[1] for co in co_sorted if len(co[0])==(self.Ldim+1)]
            
            L=np.load(self.raw_dir+'/'+name+"_minCycleDistance.npy", allow_pickle=True)
            L=torch.tensor(L[co_sel])
            
            
            labels.append(L)
                        
        if not return_list and len(labels)==1:
            return labels[0]
        else:
            return labels
        
    def __getitem__(self, idx):
        # get one example by index
        name=self.graphnames[idx]
        graph=copy.deepcopy(self.graphs[idx])
        
        
        if self.max_k_get is not None: #to make it fit into GPU
            
            curr_k=graph.ndata['V'].size(1)
            if self.max_k_get>curr_k:
                #pad with zeros
                graph.ndata['V']=torch.hstack([graph.ndata['V'], torch.zeros(graph.ndata['V'].size()[0], self.max_k_get-curr_k)])
                graph.edata['S']=torch.hstack([graph.edata['S'], torch.zeros(graph.edata['S'].size()[0], self.max_k_get-curr_k)])
            graph.ndata['V']=graph.ndata['V'][:,:self.max_k_get]
            graph.edata['S']=graph.edata['S'][:,:self.max_k_get]
        label=self.labels[idx]    
        
        return graph,label,name
    
    def __len__(self):
        # number of data examples
        return len(self.graphnames)
    
    def _save_mode(self, mode):
        # save processed data to directory `self.save_path
        if not type(self.graphs) is list:
            self.graphs=[self.graphs]
        if len(self.graphs)>0:
            graph_path = os.path.join(self.save_path, mode + '_dgl_graph.bin')
            info_path = os.path.join(self.save_path, mode + '_dgl_graph_info.bin')
            save_graphs(str(graph_path), self.graphs)
            save_info(str(info_path), { 'graphnames': self.graphnames, 'labels': self.labels, 'blacklist': self._blacklist})

    def load(self):
        # load processed data from directory `self.save_path`
        graphs, _ = load_graphs(os.path.join(self.save_path, self.mode + '_dgl_graph.bin'))
        self.graphs = graphs

        info_dict = load_info(os.path.join(self.save_path, self.mode + '_dgl_graph_info.bin'))
        self.graphnames = info_dict['graphnames']
        self.labels = info_dict['labels']
            
    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        print('IN HAS CACHE: %s, %d'%(graph_path, os.path.exists(graph_path)))
        return os.path.exists(graph_path)
    
    def _has_cache(self, mode):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, mode + '_dgl_graph.bin')
        print('IN HAS CACHE: %s, %d'%(graph_path, os.path.exists(graph_path)))
        return os.path.exists(graph_path)
    
def Complex_collate(data):
    # The input `data` is a list of triples
    #  (graph, label, name).
    graphs, labels, names = map(list, zip(*data))
    min_k=np.inf
    for g in graphs:
        if g.ndata['V'].shape[1]<min_k:
            min_k=g.ndata['V'].shape[1]
    
    for g in graphs:
        g.ndata['V']=g.ndata['V'][:,:min_k]
        g.edata['S']=g.edata['S'][:,:min_k]
        
    bg = dgl.batch(graphs)
    
    #if we have different label for each node, assemble the labels into a big array tensor
    if type(labels[0]) is torch.Tensor and len(labels[0])>1:
        labels = torch.hstack(labels)
        
    return bg, labels