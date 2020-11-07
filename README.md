# PCRP

## Introduction
This is the official implementation of "". 
## Requirements
- Python 3.6
- Pytorch 1.0.1
## Datasets
- N-UCLA:  
  Download  transformed data from https://github.com/shlizee/Predict-Cluster/tree/master/ucla_github_pytorch/UCLAdata
- NTU RGB+D 60:  
  Download  transformed data from https://github.com/shlizee/Predict-Cluster.
- NTU RGB+D 120:  
  Download raw data from https://github.com/shahroudy/NTURGB-D.  
  Use `ntu_gendata_for_predictCluster_right.py` to reprocess raw data for view invariant transformation.

## Usage
- pretrain and then linear evaluation:  
  `python pc_test.py`


## License
PCRP is released under the MIT License.
