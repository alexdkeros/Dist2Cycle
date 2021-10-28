import sys

import argparse
from src.dataset.raw_data import homology_datagen


if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    
    
    parser.add_argument('--dim', type=int, required=True,
                        help='ambient dimension/dimensionality of sampled points')
    parser.add_argument('--npts', type=int, nargs=3, required=True,
                        help='number of points to sample, range [start, stop, step]')
    parser.add_argument('--ncomplexes', type=int, required=True,
                        help='number of complexes to generate per number of samples')
    parser.add_argument('--fvals', type=int,required=True,
                        help="number of snapshots to consider per complex's filtration")
    parser.add_argument('--k', type=int,required=True,
                        help='eigenvectors to generate as features')
    parser.add_argument('--prefix', type=str,required=True,
                        help='file saving prefix (can be path/to/folder/prefix_of_files) ')
    parser.add_argument('--shortlooppath', type=str,required=True,
                        help='path to shortloop executable (parent folder "<...>/Shortloop/")')
    parser.add_argument('--Hdim', type=int, default=2,
                        help='homological dimension')
    parser.add_argument('--Cfield', type=int, default=2,
                        help='homology coefficient field')
    parser.add_argument('--complex', type=str, choices=['Alpha', 'Rips'], default='Alpha',
                        help='type of complex')
    parser.add_argument('--trivial', type=float, default=1/8,
                        help='ratio of complexes with trivial homology to allow in the dataset')
    parser.add_argument('--shapes', type=float, nargs=6, default=[0.0,0.0,1/4,1/4,1/4,1/4],
                        help='probabilities ([0,1], sum to 1) of sampling [uniform, sphere, torus, multi-holed tori, pinched_torus, multi_holed pinched tori]')
    parser.add_argument('--ntori', type=int, nargs=2, default=[2,10],
                        help='number of tori copies (random selection within range [min, max]')
    
    args=parser.parse_args()

        
    homology_datagen(args.dim, 
                    args.npts, 
                    args.ncomplexes, 
                    args.fvals, 
                    args.k,
                    args.prefix,
                    args.shortlooppath,
                    Hdim=args.Hdim, 
                    Cfield=args.Cfield,
                    complex_type=args.complex,
                    trivial_coin=args.trivial,
                    shape_coin=args.shapes,
                    ntoricopies=args.ntori)