# Pytorch Reproduction of PointNet and PointNet++ 

### Data Preparation
* Download ModelNet [here](http://modelnet.cs.princeton.edu/ModelNet40.zip) and ShapeNet [here](https://www.shapenet.org/)
* Put them respectively in `./data/ModelNet` and `./data/ShapeNet`

### Training for Classification
#### PointNet
* python train_clf.py --data ModelNet --model_name pointnet 
* python train_clf.py --data ShapeNet --model_name pointnet
#### PointNet++
* python train_clf.py --data ModelNet --model_name pointnet2 
* python train_clf.py --data ShapeNet --model_name pointnet2
### Training for Part Segmentation
#### PointNet
* python train_seg.py --data ModelNet --model_name pointnet  
* python train_clf.py --data ShapeNet --model_name pointnet
#### PointNet++
* python train_seg.py --data ModelNet --model_name pointnet2 
* python train_clf.py --data ShapeNet --model_name pointnet2

### Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)
