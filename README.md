# Pytorch implementation of PointNet and PointNet++ 

## Data Preparation
* Download ModelNet [here](http://modelnet.cs.princeton.edu/ModelNet40.zip) and ShapeNet [here](https://www.shapenet.org/)
* Put them respectively in `./data/ModelNet` and `./data/ShapeNet`

## Training for classification
### PointNet
python train_clf.py --data ModelNet --model_name pointnet or python train_clf.py --data ShapeNet --model_name pointnet
### PointNet++
python train_clf.py --data ModelNet --model_name pointnet2 or python train_clf.py --data ShapeNet --model_name pointnet2
## Training for segmentation
### PointNet
python train_seg.py --data ModelNet --model_name pointnet or python train_clf.py --data ShapeNet --model_name pointnet
### PointNet++
python train_seg.py --data ModelNet --model_name pointnet2 or python train_clf.py --data ShapeNet --model_name pointnet2

## Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)
