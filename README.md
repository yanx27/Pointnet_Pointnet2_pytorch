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
### Performance
| Model | Inctance avg | Class avg	 |aero |	bag |	cap	 |car	 |chair	 |ear phone	 |guitar |	knife |	lamp	 |laptop |	motor	 |mug |	pistol	 |rocket |	skate board |	 table |
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
|PointNet (Official)	|**83.7**|**80.4**	|83.4|	78.7|	82.5|	74.9|	89.6	|73|	91.5|	85.9	|80.8|	95.3|	65.2	|93|	81.2|	57.9|	72.8|	80.6|
|PointNet (Pytorch)|	82.1	|78.1|	81.1	|77.8	|83.7	|74.3	|83.3|	65.7|	90.5	|85.1|	78.1	|94.5	|63.7	|91.7	|80.5|56.2	|73.7	|67.5|
|PointNet++ (Official)|**85.1**	|**81.9**	|82.4|79	|87.7	|77.3|	90.8|	71.8|	91|	85.9|	83.7|	95.3	|71.6|	94.1	|81.3|	58.7|	76.4|	82.6|
|PointNet++ (Pytorch)|	84.4|	80.5	|82.6|	78.7|	82.3	|78.1|86.8|	63.8	|91.6|	88.9|	83.6	|96.8	|63.3	|95.7	|82.8|	55.7	|76.3	|71.1|

* Train Pointnet and Pointnet++ with 0.001 learning rate in Adam, 16 batchsize and decay 0.5 learning rate every 20 epochs can gain the results above.
* **Class avg** is the mean IoU averaged across all object categories, and **inctance avg** is the mean IoU across all objects.
* We did not use data augmentation (norm and randomly jitter) so the results is relatively lower than the official version's.
* In official version PointNet, author use 2048 point cloud in training and 3000 point cloud with norm in testing. In official version PointNet++, author use 2048 point cloud with its norm (Bx2048x6) in both training and testing.
  


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

## TODO

- [x] PointNet and PointNet++ 
- [x] Experiment 
- [ ] Visualization tool



## Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)

## Links
[Official PointNet](https://github.com/charlesq34/pointnet) and [Official PointNet++](https://github.com/charlesq34/pointnet2)
