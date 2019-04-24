import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
from utils import to_categorical
from collections import defaultdict
from torch.autograd import Variable
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
from utils import test_partseg
from tqdm import tqdm
from model.pointnet2 import PointNet2PartSeg_msg_one_hot
from model.pointnet import PointNetDenseCls,PointNetLoss

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def parse_args():
    parser = argparse.ArgumentParser('PointNet2')
    parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=201, help='number of epochs for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--model_name', type=str, default='pointnet2', help='Name of model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--multi_gpu', type=str, default=None, help='whether use multi gpu training')
    parser.add_argument('--jitter', default=False, help="randomly jitter point cloud")
    parser.add_argument('--step_size', type=int, default=20, help="randomly rotate point cloud")

    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.multi_gpu is None else '0,1,2,3'
    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) +'/%sPartSeg-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s_partseg.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    norm = True if args.model_name == 'pointnet' else False
    TRAIN_DATASET = PartNormalDataset(npoints=2048, split='trainval',normalize=norm, jitter=args.jitter)
    dataloader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batchsize,shuffle=True, num_workers=int(args.workers))
    TEST_DATASET = PartNormalDataset(npoints=2048, split='test',normalize=norm,jitter=False)
    testdataloader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=10,shuffle=True, num_workers=int(args.workers))
    print("The number of training data is:",len(TRAIN_DATASET))
    logger.info("The number of training data is:%d",len(TRAIN_DATASET))
    print("The number of test data is:", len(TEST_DATASET))
    logger.info("The number of test data is:%d", len(TEST_DATASET))
    num_classes = 16
    num_part = 50
    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = PointNet2PartSeg_msg_one_hot(num_part) if args.model_name == 'pointnet2'else PointNetDenseCls(cat_num=num_classes,part_num=num_part)

    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')
    pretrain = args.pretrain
    init_epoch = int(pretrain[-14:-11]) if args.pretrain is not None else 0


    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    '''GPU selection and multi-GPU'''
    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model.cuda()
    criterion = PointNetLoss()
    LEARNING_RATE_CLIP = 1e-5

    history = defaultdict(lambda: list())
    best_acc = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(init_epoch,args.epoch):
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'],LEARNING_RATE_CLIP)
        print('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for i, data in tqdm(enumerate(dataloader, 0),total=len(dataloader),smoothing=0.9):
            points, label, target, norm_plt = data
            points, label, target = Variable(points.float()),Variable(label.long()),  Variable(target.long())
            points = points.transpose(2, 1)
            norm_plt = norm_plt.transpose(2, 1)
            points, label, target,norm_plt = points.cuda(),label.squeeze().cuda(), target.cuda(), norm_plt.cuda()
            optimizer.zero_grad()
            model = model.train()
            if args.model_name == 'pointnet':
                labels_pred, seg_pred, trans_feat = model(points, to_categorical(label, 16))
                seg_pred = seg_pred.contiguous().view(-1, num_part)
                target = target.view(-1, 1)[:, 0]
                loss, seg_loss, label_loss = criterion(labels_pred, label, seg_pred, target, trans_feat)
            else:
                seg_pred = model(points, norm_plt, to_categorical(label, 16))
                seg_pred = seg_pred.contiguous().view(-1, num_part)
                target = target.view(-1, 1)[:, 0]
                loss = F.nll_loss(seg_pred, target)

            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()

        forpointnet2 = args.model_name == 'pointnet2'
        test_metrics, test_hist_acc, cat_mean_iou = test_partseg(model.eval(), testdataloader, seg_label_to_cat,50,forpointnet2)

        print('Epoch %d %s accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
                 epoch, blue('test'), test_metrics['accuracy'],test_metrics['class_avg_iou'],test_metrics['inctance_avg_iou']))

        logger.info('Epoch %d %s Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
                 epoch, blue('test'), test_metrics['accuracy'],test_metrics['class_avg_iou'],test_metrics['inctance_avg_iou']))
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth' % (checkpoints_dir,args.model_name, epoch, best_acc))
            logger.info(cat_mean_iou)
            logger.info('Save model..')
            print('Save model..')
            print(cat_mean_iou)
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        print('Best accuracy is: %.5f'%best_acc)
        logger.info('Best accuracy is: %.5f'%best_acc)
        print('Best class avg mIOU is: %.5f'%best_class_avg_iou)
        logger.info('Best class avg mIOU is: %.5f'%best_class_avg_iou)
        print('Best inctance avg mIOU is: %.5f'%best_inctance_avg_iou)
        logger.info('Best inctance avg mIOU is: %.5f'%best_inctance_avg_iou)


if __name__ == '__main__':
    args = parse_args()
    main(args)

