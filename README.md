# PCRP

## Introduction
This is the official implementation of "Prototypical Contrast and Reverse Prediction: Unsupervised   Skeleton based Action Recognition". 
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

Put the data into the folder that matches the codes in `pc_test.py`

## Usage
- pretrain and then linear evaluation:  
  `python pc_test.py`

## Learned Models
https://drive.google.com/drive/folders/1cck_0od9LqMIt2fvCOcca956dvCJTxcR?usp=sharing

## License
PCRP is released under the MIT License.

## Citation
@misc{xu2020prototypical,  
      title={Prototypical Contrast and Reverse Prediction: Unsupervised Skeleton Based Action Recognition},   
      author={Shihao Xu and Haocong Rao and Xiping Hu and Bin Hu},  
      year={2020},  
      eprint={2011.07236},  
      archivePrefix={arXiv},  
      primaryClass={cs.CV}  
      }
