import sys
sys.path.append('..')

import argparse

from src.dataset.ComplexesDatasetLazy import ComplexesDatasetLazy

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--rawdataset', type=str, required=True,
                        help='path to raw dataset files')
    parser.add_argument('--datasetname', type=str, required=True,
                        help='datasetname (must match the one in the dataset folder, without the train and structure details)')
    parser.add_argument('--datasetsavedir', type=str, required=True,
                        help='directory to look into, or save dataset if not exists')
    parser.add_argument('--adjacency', type=str, choices=['laplacian', 'boundary', 'complete'], default='boundary',
                        help="adjacency type from which to build gnn")
    parser.add_argument('--Ltildetype', type=str, choices=['func', 'approx', 'norm', 'original', 'shiftedOriginal'], default='func',
                        help="way to compute the L tilde pseudoinverse")
    parser.add_argument('--maxkget', type=int, default=600,
                        help='maximum number of eigenvectors to return with getitem, as features')
    parser.add_argument('--maxkstore', type=int, default=600,
                        help='maximum number of eigenvectors to store in dataset graphs')
    parser.add_argument('--traincut', type=float, default=0.8,
                        help='float: training data percentage, int: num of complexes')
    parser.add_argument('--valcut', type=float, default=0.2,
                        help='float:validation data percentage, int: num of complexes')
    parser.add_argument('--testcut', type=float, default=0.0,
                        help='float: test data percentage, int: num of complexes')
    
    args=parser.parse_args()


    raw_dir=args.rawdataset
    dataset_name=args.datasetname
    save_dir=args.datasetsavedir
    
        
    feats=None
    mode='train'
    Ldim=1
    
    adjacency=args.adjacency
    max_k_get=args.maxkget
    max_k_store=args.maxkstore
    
    Ltype=args.Ltildetype
    LapproxPower=5
    
    traincut=args.traincut
    valcut=args.valcut
    testcut=args.testcut
    
    train_dataset=ComplexesDatasetLazy(raw_dir=raw_dir,
                             dataset_name=dataset_name,
                             save_dir=save_dir,
                             feats=None,
                             mode=mode,
                             Ldim=Ldim,
                             adjacency=adjacency,
                             max_k_get=max_k_get,
                             max_k_store=max_k_store,
                             Ltype=Ltype,
                             LapproxPower=LapproxPower,
                             traincut=traincut,
                             valcut=valcut,
                             testcut=testcut)