# Pytorch Reproduction of PointNet and PointNet++ 

### Data Preparation
* Download **ModelNet** [here](http://modelnet.cs.princeton.edu/ModelNet40.zip) for classification and **ShapeNet** [here](https://www.shapenet.org/) for part segmentation. Put them respectively in `./data/ModelNet` and `./data/ShapeNet`.
* Run `download_data.sh` for **S3DIS** sematic segmantation and save it in `./indoor3d_sem_seg_hdf5_data/`

### Training for Classification
#### PointNet
* python train_clf.py --model_name pointnet 
#### PointNet++
* python train_clf.py --model_name pointnet2 

### Training for Part Segmentation
#### PointNet
* python train_seg.py --model_name pointnet
#### PointNet++
* python train_seg.py --model_name pointnet2

### Training for Sematic Segmentation
#### PointNet
* python train_semseg.py --model_name pointnet
#### PointNet++
* python train_semseg.py --model_name pointnet2

### Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)
