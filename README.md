# Pytorch Implementation of PointNet and PointNet++ 

## Data Preparation
* Download **ModelNet** [here](http://modelnet.cs.princeton.edu/ModelNet40.zip) for classification and **ShapeNet** [here](https://www.shapenet.org/) for part segmentation. Putting them respectively in `./data/ModelNet` and `./data/ShapeNet`.
* Run `download_data.sh`  and download prepared **S3DIS** dataset for sematic segmantation and save it in `./data/indoor3d_sem_seg_hdf5_data/`

## Training for Classification
### PointNet
* python train_clf.py --model_name pointnet 
### PointNet++
* python train_clf.py --model_name pointnet2 
### Performance
| Model | Accuracy |
|--|--|
| PointNet (Original) |  89.2|
| PointNet (Pytorch) |  **89.4**|
| PointNet++ (Original) | **91.9** |
| PointNet++ (Pytorch) | 91.8 |

## Training for Part Segmentation
### PointNet
* python train_partseg.py --model_name pointnet
### PointNet++
* python train_partseg.py --model_name pointnet2

## Training for Sematic Segmentation
### PointNet
* python train_semseg.py --model_name pointnet
### PointNet++
* python train_semseg.py --model_name pointnet2

## Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)
