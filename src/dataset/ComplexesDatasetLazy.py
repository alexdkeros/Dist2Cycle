import sys
sys.path.append('..')

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

from src.dataset.ComplexesDataset import ComplexesDataset
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info

from src.utils.converters import extract_simplices, build_boundaries, build_laplacians, compute_Ltilde_pinv



class ComplexesDatasetLazy(ComplexesDataset):
    '''
    dataset of simplicial complexes for "Topology-aware learning on simplicial complexes via Spectrum Filtering Graph Neural Networks"
    saves/loads a dataset one graph at a time (more memory efficient)
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
        
        dataset_name='LAZY_'+dataset_name

        super(ComplexesDatasetLazy, self).__init__(raw_dir=raw_dir,
                                                   save_dir=save_dir,
                                                   saved_dataname=saved_dataname,
                                                   label_type=label_type,
                                                   dataset_name=dataset_name,
                                                   feats=feats,
                                                   mode=mode,
                                                   adjacency=adjacency,
                                                   Ldim=Ldim,
                                                   max_k_get=max_k_get,
                                                   max_k_store=max_k_store,
                                                   Ltype=Ltype,
                                                   LapproxPower=LapproxPower,
                                                   traincut=traincut,
                                                   valcut=valcut,
                                                   testcut=testcut,
                                                   url=url,
                                                   force_reload=force_reload,
                                                   verbose=verbose)
        


    

    
    
    def _process_and_save(self, datasetnames, dstype):    
        print('PROCESS AND SAVE: ===== %s ====='% dstype)
        self.graphnames=datasetnames
        
        #process and save LAZILY
        for data_indx, name in enumerate(self.graphnames):
            
            if data_indx%10==0:
                print(f'file {data_indx}/{len(self.graphnames)} ({round(100*data_indx/len(self.graphnames),1)}%)')
            
            graph=self._load_graphs(name)
    
            #do not include graph in dataset if something failed
            if len(self._blacklist)>0:
                if name in self._blacklist:
                    self.graphnames.remove(name)
                    continue
                    
            label=self._load_labels(name, return_list=False)
            
            if self.feats is not None:
                #not implemented but OK
                self._load_features(self.graphnames, self.graphs)
                
            graph_path=os.path.join(self.save_path, name + '_dgl_graph_'+dstype+'.bin')
            graph_labels={'labels':label}
            save_graphs(graph_path, graph,graph_labels)
                        
        self._save_mode(dstype)
        
        return self.graphnames
    
    
    def __getitem__(self, idx):
        # get one example by index
        name=self.graphnames[idx]
        
        graph_path=os.path.join(self.save_path, name + '_dgl_graph_'+self.mode+'.bin')
        graph, graph_dict=load_graphs(graph_path)
        graph=graph[0]
        label=graph_dict['labels']
                
        if self.max_k_get is not None: #to make it fit into GPU
            
            curr_k=graph.ndata['V'].size(1)
            if self.max_k_get>curr_k:
                #pad with zeros
                graph.ndata['V']=torch.hstack([graph.ndata['V'], torch.zeros(graph.ndata['V'].size()[0], self.max_k_get-curr_k)])
                graph.edata['S']=torch.hstack([graph.edata['S'], torch.zeros(graph.edata['S'].size()[0], self.max_k_get-curr_k)])
            graph.ndata['V']=graph.ndata['V'][:,:self.max_k_get]
            graph.edata['S']=graph.edata['S'][:,:self.max_k_get]
        
        return graph,label,name
    

    
    def _save_mode(self, mode):
        # save processed datanames to directory `self.save_path (lazyload lists)
        info_path = os.path.join(self.save_path, mode + '_dgl_graph_info.bin')
        save_info(str(info_path), { 'graphnames': self.graphnames,  'blacklist': self._blacklist})

    def load(self):
        # load processed data from directory `self.save_path`
        info_dict = load_info(os.path.join(self.save_path, self.mode + '_dgl_graph_info.bin'))
        self.graphnames = info_dict['graphnames']
        #validate that all graphs exist, if not remove them from the list
        missing=[]
        for name in self.graphnames:
            graph_path=os.path.join(self.save_path, name + '_dgl_graph_'+self.mode+'.bin')
            if not os.path.exists(graph_path):
                print(f'!{graph_path} is missing!')
                missing.append(name)
        for missname in missing:
            self.graphnames.remove(missname)
            
    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph_info.bin')
        print('IN HAS CACHE: %s, %d'%(graph_path, os.path.exists(graph_path)))
        return os.path.exists(graph_path)
    
    def _has_cache(self, mode):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, mode + '_dgl_graph_info.bin')
        print('IN HAS CACHE: %s, %d'%(graph_path, os.path.exists(graph_path)))
        return os.path.exists(graph_path)
    
