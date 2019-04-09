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
| PointNet (Official) |  89.2|
| PointNet (Pytorch) |  **89.4**|
| PointNet++ (Official) | **91.9** |
| PointNet++ (Pytorch) | 91.8 |

* Train Pointnet with 0.001 learning rate in SGD, 24 batchsize and 141 epochs can gain the results above.
* Train Pointnet++ with 0.001 learning rate in SGD, 12 batchsize and 45 epochs can gain the results above.

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
### Performance (test on Area_5)
|Model  | Mean IOU | ceiling | floor | wall | beam | column | window | door |  chair| tabel| bookcase| sofa | board | clutter | 
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| PointNet (Official) | 41.09|88.8|**97.33**|69.8|0.05|3.92|**46.26**|10.76|**52.61**|**58.93**|**40.28**|5.85|26.38|33.22|
| PointNet (Pytorch) | **44.43**|**91.1**|96.8|**72.1**|**5.82**|**14.7**|36.03|**37.1**|49.36|50.17|35.99|**14.26**|**33.9**|**40.23**|

* Train Pointnet with 0.001 learning rate in Adam, 24 batchsize and 84 epochs can gain the results above.
## Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)
