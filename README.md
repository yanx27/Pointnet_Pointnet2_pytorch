# Pytorch Implementation of PointNet and PointNet++ 

This repo is implementation for [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) in pytorch.

## Update
**2019/11/26:**

(1). Fixed some errors in previous codes and added data augmentation tricks. Now classification can achieve 92.5\%! 

(2). Added testing codes, including classification and segmentation, and semantic segmentation with visualization. 

(3). Organized all models into `./models` files for easy use.


## Classification
### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`

### Run
```
## Check model in ./models folder
## E.g. pointnet2_msg
python train_cls.py --model pointnet2_cls_msg --normal --log_dir pointnet2_cls_msg
python test_cls.py --normal --log_dir pointnet2_cls_msg
```

### Performance
| Model | Accuracy |
|--|--|
| PointNet (Official) |  89.2|
| PointNet++ (Official) | 91.9 |
| PointNet (Pytorch without normal) |  90.6|
| PointNet (Pytorch with normal) |  91.4|
| PointNet2_ssg (Pytorch without normal) |  92.2|
| PointNet2_ssg (Pytorch with normal) |  **92.4**|

## Part Segmentation
### Data Preparation
Download alignment **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)  and save in `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`
### Run
```
## Check model in ./models folder
## E.g. pointnet2_msg
python train_partseg.py --model pointnet2_part_seg_msg --normal
python test_partseg.py --normal --log_dir pointnet2_part_seg_msg
```
### Performance
| Model | Inctance avg | Class avg	 
|--|--|--|
|PointNet (Official)	|**83.7**|**80.4**	
|PointNet (Pytorch)|	-	|-|	
|PointNet++ (Official)|**85.1**	|**81.9**	
|PointNet++ (Pytorch)|	-|	-	


## Semantic Segmentation
### Run
```
## Check model in ./models folder
## E.g. pointnet2_ssg
python train_semseg.py --model pointnet2_sem_seg --normal
python test_semseg.py --normal --log_dir pointnet2_sem_seg
```
### Performance (test on Area_5)
|Model  | Mean IOU | 
|--|--|
| PointNet (Official) | 41.09|
| PointNet (Pytorch) | -|
| PointNet++ (Official) |N/A | 
| PointNet++ (Pytorch) | -|


## Visualization
### Using show3d_balls.py
```
cd visualizer
bash build.sh #build C++ code for visualization
## run one example 
python show3d_balls.py
```
![](/visualizer/pic.png)
### Using MeshLab
![](/visualizer/pic2.png)


## Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)

## Links
[Official PointNet](https://github.com/charlesq34/pointnet) and [Official PointNet++](https://github.com/charlesq34/pointnet2)
