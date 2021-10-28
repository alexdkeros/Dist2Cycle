import sys
import os
import time
import random
import itertools
import numpy as np
import numpy.linalg as npla
import argparse

import json

import dgl
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.dataset.ComplexesDatasetLazy import ComplexesDatasetLazy
from src.dataset.ComplexesDataset import ComplexesDataset, Complex_collate
from src.model.Dist2CycleRegressor import Dist2CycleRegressor

device = "cuda"
print(f'CUDA: {torch.cuda.is_available()}')




if __name__=="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['Dist2Cycle'], required=True,
                        help='model selection')
    parser.add_argument('--rawdataset', type=str, required=True,
                        help='path to raw dataset files')
    parser.add_argument('--datasetpath', type=str, required=True,
                        help='dataset full path (must match the one in the dataset folder)')
    parser.add_argument('--modelparams', type=str, required=True,
                        help='json file with model parameters [if hidden_feats is list, n_layers=len(hidden_feats)+1 (1 for incorporating the output layer)')
    parser.add_argument('--savedir', type=str, required=True,
                        help='path to save trained model')
    parser.add_argument('--infeaturescols', default=5,
                        help='spectral embedding dimension')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--batchsize', type=int, default=3,
                        help='training batch size')
    parser.add_argument('--verbose', action='store_true',
                        help='verbose output')
    
    
    args=parser.parse_args()
    
    modelarch=args.model
    raw_dir=args.rawdataset
    dataset_path=args.datasetpath
    epochs=args.epochs
    batch_size=args.batchsize
    
    infeaturescols=args.infeaturescols
    
    model_save_dir=args.savedir

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    verbose=args.verbose  
    
    dataset_path_split=dataset_path.split('/')
    dataset_name=dataset_path_split[-1]
    
    lazyload=False
    if dataset_name.split('_')[0]=='LAZY':
        lazyload=True

    
    modelparamfile=args.modelparams
    
    with open(modelparamfile) as f:
        model_params=json.load(f)
        
    for param, pval in model_params.items():
        if pval=='None':
            model_params[param]=None
        elif pval=='True':
            model_params[param]=True
        elif pval=='False':
            model_params[param]=False
        elif pval=='Sigmoid':
            model_params[param]=nn.Sigmoid()
        elif pval=='ReLU':
            model_params[param]=nn.ReLU()
        elif pval=='Softsign':
            model_params[param]=nn.Softsign()
        elif pval=='Tanh':
            model_params[param]=nn.Tanh()
        elif pval=='LeakyReLU':
            model_params[param]=nn.LeakyReLU()
    
    
    if isinstance(model_params['hidden_feats'], list):
        model_params['n_layers']=len(model_params['hidden_feats'])
            
    
    #################### ADDITIONAL PARAMETERS ####################
    
    label_type='min_dist_from_cycles'

    ###############################################################
    
    train_dataset=ComplexesDatasetLazy(raw_dir=raw_dir,
                                       save_dir='/'.join(dataset_path_split[:-1]),
                                       saved_dataname=dataset_name,
                                       mode='train')
    
    val_dataset=ComplexesDatasetLazy(raw_dir=raw_dir,
                                   save_dir='/'.join(dataset_path_split[:-1]),
                                   saved_dataname=dataset_name,
                                   mode='val')
    
    loss_criterion = torch.nn.MSELoss()
    
    
    model_params['in_feats']=infeaturescols+3
    
    model_params['featsize']=infeaturescols
    model_params['label_type']=label_type
    model_params['batch_size']=batch_size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,  collate_fn=Complex_collate) #, sampler=train_subsampler

    model=Dist2CycleRegressor(in_feats=model_params['in_feats'],
                         n_layers=model_params['n_layers'],
                         out_feats=model_params['out_feats'],
                         hidden_feats=model_params['hidden_feats'],
                         aggregator_type=model_params['aggregator_type'],
                         weighted_edges=model_params['weighted_edges'],
                         fc_bias=model_params['fc_bias'],
                         norm=model_params['norm'],
                         fc_activation=model_params['fc_activation'],
                         out_activation=model_params['out_activation'],
                         initialization=model_params['initialization'],
                         verbose=verbose)
    
    model.to(device)

    model.reset_weights()
        
    if model_params['lr'] is None:
        model_params['lr']=1e-3
    if model_params['weight_decay'] is None:
        model_params['weight_decay']=0.0
        
    optimizer = Adam(model.parameters(), lr=model_params['lr'], weight_decay=model_params['weight_decay'])
    
    print('---------- TRAINING ----------')
    
    for epoch in range(epochs):
    
        running_loss=0.0
        
        t_start=time.time()
    
        for batch_id, batch_data in enumerate(train_loader):
    
            graph, labels=batch_data
            
            eigenvecs=graph.ndata['V'][:,0:infeaturescols]
            feats=torch.hstack([graph.ndata['lk_hom'],eigenvecs])
            
            
            #remove unnecessary data from the graph before loading it to GPU
            del graph.ndata['V']
            del graph.edata['S']            
            if 'lk_hom' in graph.ndata:
                del graph.ndata['lk_hom']
                
            
            graph=graph.to(device)
            feats=feats.float().to(device)
            labels=labels.float().to(device)

            logits=model(graph,feats)
            
            loss=loss_criterion(logits.squeeze(-1),labels)
            
            running_loss+=loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            
            grad_norm=nn.utils.clip_grad_norm_(model.parameters(), 0.5)
          
            optimizer.step()
            
            t_stop=time.time()
                
            with torch.no_grad():
                print(f'** epoch: {epoch}/{epochs}, loss:{running_loss/len(train_loader)}, ({round(t_stop-t_start,2)} s) **')
                
    
    #save trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss/len(train_loader),
        }, f'{model_save_dir}/model_snapshot.pkl')
            
    print('---------- TESTING ----------')
    
    test_loss=0.0
    
    model.eval()
    with torch.no_grad():
        for g_id in range(len(val_dataset)):
            
            t_start=time.time()
            
            graph, labels,gname=val_dataset[g_id]
            
            eigenvecs=graph.ndata['V'][:,0:infeaturescols]
            feats=torch.hstack([graph.ndata['lk_hom'],eigenvecs])

            #remove unnecessary data from the graph before loading it to GPU
            del graph.ndata['V']
            del graph.edata['S']            
            if 'lk_hom' in graph.ndata:
                del graph.ndata['lk_hom']
            
            graph=graph.to(device)
            feats=feats.float().to(device)
            labels=labels.float().to(device)

            logits=model(graph,feats)
            loss=loss_criterion(logits.squeeze(-1),labels)
            
            print(f'Test loss {gname}: {loss.item()}')
                
            test_loss+=loss.item()
            
        print(f'*** Mean test loss: {test_loss/len(val_dataset)} ***')
    