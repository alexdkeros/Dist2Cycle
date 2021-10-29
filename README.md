Dist2Cycle : code, scripts & showcase
=====================================
Paper: https://arxiv.org/abs/2110.15182

(preceded with $ are linux terminal commands)

Folders:

- `raw_data`: contains raw data examples of 2D & 3D TORI datasets
- `datasets`: 2D & 3D TORI processed datasets from raw_data in Hodge Laplacian graph form, ready for training and validating a Dist2Cycle model
- `models`  : trained models in 2D and 3D used to produce the results of the main paper
- `src` : all necessary dataset and model definitions, alongside utility functions

Files:

- `dependencies.txt` : code dependencies of python scripts & jupyter notebook

Python scripts:
- `gen_raw_data.py`  
Generate raw data instances from the TORI dataset  
*requires Shortloop*  
Help: 
`$ python gen_raw_data.py -h`  
Example command:  
`$ python gen_raw_data.py --dim 2 --npts 100 121 10 --ncomplexes 1 --fvals 2 --k 10 --prefix raw_data/2D/EXAMPLES --shortlooppath <path_to_shortloop_folder>`
- `gen_dataset.py`  
Generate Hodge Laplacian graphs from raw data, suitable for training and evaluation  
Help:  
`$ python gen_dataset.py -h`  
Example command:  
`$ python gen_dataset.py --rawdataset raw_data/2D/EX_Alpha_D2_HDim2_k10 --datasetname EX_Alpha_D2_HDim2_k10 --datasetsavedir datasets/2D`
- `training_suite.py`:  
Train a Dist2Cycle model, based on model_params.json configuration file provided  
Help:  
`$ python training_suite.py`  
Example command:  
`$ python training_suite.py --model Dist2Cycle --rawdataset raw_data/2D/EX_Alpha_D2_HDim2_k10 --datasetpath datasets/2D/LAZY_EX_Alpha_D2_HDim2_k10_boundary_1HD_Lfunc_k10_7_6_0.0 --modelparams model_params.json --savedir models/CUSTOM_MODEL`

Jupyter notebooks:

- `Dist2Cycle_showcase.ipynb`:  
The notebook loads a trained model and evaluates it against a validation dataset, providing complex and regression vizualizations.
		
