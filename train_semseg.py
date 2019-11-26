"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader import S3DISDataset, S3DISDatasetWholeScene
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training [default: 12]')
    parser.add_argument('--epoch',  default=1024, type=int, help='Epoch to run [default: 251]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=8192, help='Point Number [default: 2048]')
    parser.add_argument('--with_rgb', action='store_true', default=False, help='Whether to use RGB information [default: False]')
    parser.add_argument('--step_size', type=int,  default=200, help='Decay step for lr decay [default: every 200 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/stanford_indoor3d/'
    NUM_CLASSES = 13
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    FEATURE_CHANNEL = 3 if args.with_rgb else 0

    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(root, split='train', with_rgb=args.with_rgb, test_area=args.test_area, block_points=NUM_POINT)
    print("start loading whole scene validation data ...")
    TEST_DATASET_WHOLE_SCENE = S3DISDatasetWholeScene(root, split='test', with_rgb=args.with_rgb, test_area=args.test_area, block_points=NUM_POINT)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,shuffle=True, num_workers=4)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" %  len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES, with_rgb=args.with_rgb).cuda()
    criterion = MODEL.get_loss().cuda()


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mean_correct = []
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))

        '''learning one epoch'''
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, target, weight = data
            points = points.data.numpy()
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.rotate_point_cloud_z(points[:,:, 0:3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(),target.long().cuda()
            points = points.transpose(2, 1)
            optimizer.zero_grad()
            classifier = classifier.train()
            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            optimizer.step()
        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)

        # evaluate on whole scenes, for each block, only sample NUM_POINT points
        if epoch % 10 ==0:
            with torch.no_grad():
                num_batches = len(TEST_DATASET_WHOLE_SCENE)

                total_correct = 0
                total_seen = 0
                loss_sum = 0
                total_seen_class = [0 for _ in range(NUM_CLASSES)]
                total_correct_class = [0 for _ in range(NUM_CLASSES)]
                total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

                labelweights = np.zeros(NUM_CLASSES)
                is_continue_batch = False

                extra_batch_data = np.zeros((0, NUM_POINT, 3 + FEATURE_CHANNEL))
                extra_batch_label = np.zeros((0, NUM_POINT))
                extra_batch_smpw = np.zeros((0, NUM_POINT))
                for batch_idx in tqdm(range(num_batches),total=num_batches):
                    if not is_continue_batch:
                        batch_data, batch_label, batch_smpw = TEST_DATASET_WHOLE_SCENE[batch_idx]
                        batch_data = np.concatenate((batch_data, extra_batch_data), axis=0)
                        batch_label = np.concatenate((batch_label, extra_batch_label), axis=0)
                        batch_smpw = np.concatenate((batch_smpw, extra_batch_smpw), axis=0)
                    else:
                        batch_data_tmp, batch_label_tmp, batch_smpw_tmp = TEST_DATASET_WHOLE_SCENE[batch_idx]
                        batch_data = np.concatenate((batch_data, batch_data_tmp), axis=0)
                        batch_label = np.concatenate((batch_label, batch_label_tmp), axis=0)
                        batch_smpw = np.concatenate((batch_smpw, batch_smpw_tmp), axis=0)
                    if batch_data.shape[0] < BATCH_SIZE:
                        is_continue_batch = True
                        continue
                    elif batch_data.shape[0] == BATCH_SIZE:
                        is_continue_batch = False
                        extra_batch_data = np.zeros((0, NUM_POINT, 3 + FEATURE_CHANNEL))
                        extra_batch_label = np.zeros((0, NUM_POINT))
                        extra_batch_smpw = np.zeros((0, NUM_POINT))
                    else:
                        is_continue_batch = False
                        extra_batch_data = batch_data[BATCH_SIZE:, :, :]
                        extra_batch_label = batch_label[BATCH_SIZE:, :]
                        extra_batch_smpw = batch_smpw[BATCH_SIZE:, :]
                        batch_data = batch_data[:BATCH_SIZE, :, :]
                        batch_label = batch_label[:BATCH_SIZE, :]
                        batch_smpw = batch_smpw[:BATCH_SIZE, :]

                    batch_label = torch.Tensor(batch_label)
                    batch_data = torch.Tensor(batch_data)
                    batch_data, batch_label = batch_data.float().cuda(), batch_label.long().cuda()
                    batch_data = batch_data.transpose(2, 1)
                    seg_pred, _ = classifier(batch_data)
                    seg_pred = seg_pred.contiguous()
                    batch_label = batch_label.cpu().data.numpy()
                    pred_val = seg_pred.cpu().data.max(2)[1].numpy()
                    correct = np.sum((pred_val == batch_label) & (batch_smpw > 0))
                    total_correct += correct
                    total_seen += np.sum(batch_smpw > 0)
                    tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                    labelweights += tmp
                    for l in range(NUM_CLASSES):
                        total_seen_class[l] += np.sum((batch_label == l) & (batch_smpw > 0))
                        total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) & (batch_smpw > 0))
                        total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)) & (batch_smpw > 0))

                mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
                log_string('eval whole scene mean loss: %f' % (loss_sum / float(num_batches)))
                log_string('eval point avg class IoU: %f' % mIoU)
                log_string('eval whole scene point accuracy: %f' % (total_correct / float(total_seen)))
                log_string('eval whole scene point avg class acc: %f' % (
                    np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
                labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))

                iou_per_class_str = '------- IoU --------\n'
                for l in range(NUM_CLASSES):
                    iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                        seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l],
                        total_correct_class[l] / float(total_iou_deno_class[l]))
                log_string(iou_per_class_str)

                if (mIoU >= best_iou):
                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': epoch,
                        'class_avg_iou': mIoU,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                    log_string('Saving model....')

        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)

