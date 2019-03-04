import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
import torch.nn.functional as F
from torch.autograd import Variable
from data_utils.ShapeNetDataLoader import ShapeNetDataLoader, load_data
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils import test, save_checkpoint, to_categorical
from model.pointnet2 import PointNet2ClsMsg
from model.pointnet import PointNetCls


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet2')

    parser.add_argument('--batchsize', type=int, default=8,
                        help='batch size in training')
    parser.add_argument('--epoch',  default=5, type=int,
                        help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0001, type=float,
                        help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--train_metric', type=str, default=False,
                        help='whether evaluate on training dataset')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='whether use pretrain model')
    parser.add_argument('--result_dir', type=str, default='./experiment/results/',
                        help='dir to save pictures')
    parser.add_argument('--data', type=str, default='ShapeNet',
                        help='data path')
    parser.add_argument('--log_dir', type=str, default='./experiment/logs/',
                        help='decay rate of learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.9,
                        help='decay rate of learning rate')
    parser.add_argument('--rotation',  default=None,
                        help='range of training rotation')
    parser.add_argument('--model_name',  default='pointnet2',
                        help='range of training rotation')
    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    BATCHSIZE = args.batchsize
    LEARNING_RATE = args.learning_rate
    EPOCH = args.epoch
    COMPUTE_TRAIN_METRICS = args.train_metric
    DATA_PATH = './data/%s/' % args.data

    if args.rotation is not None:
        ROTATION = (int(args.rotation[0:2]),int(args.rotation[3:5]))
    else:
        ROTATION = None

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(exist_ok=True)
    checkpoints_dir = Path('./experiment/checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("PointNet2")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.log_dir + 'train-'+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))+'.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    train_data, train_label, test_data, test_label = load_data(DATA_PATH, classification=True)
    logger.info("The number of training data is: %d",train_data.shape[0])
    logger.info("The number of test data is: %d", test_data.shape[0])
    trainDataset = ShapeNetDataLoader(train_data, train_label, rotation=ROTATION)
    if ROTATION is not None:
        print('The range of training rotation is',ROTATION)
    testDataset = ShapeNetDataLoader(test_data, test_label, rotation=ROTATION)
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCHSIZE, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=BATCHSIZE, shuffle=False)

    '''MODEL LOADING'''
    num_class = 16 if args.data == 'ShapeNet' else 40
    model = PointNetCls(num_class).cuda() if args.model_name == 'pointnet' else PointNet2ClsMsg().cuda()
    if args.pretrain is not None:
        print('Use pretrain model...')
        logger.info('Use pretrain model')
        checkpoint = torch.load(args.pretrain)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-08
    )
    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0
    history = defaultdict(lambda: list())

    '''TRANING'''
    logger.info('Start training...')
    total_train_acc = []
    total_test_acc = []
    for epoch in range(start_epoch, EPOCH):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, EPOCH))
        logger.info('Epoch %d (%d/%s):' ,global_epoch + 1, epoch + 1, EPOCH)

        for batch_id, (data, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            data = data.permute(0,2,1).float().cuda()
            target = target.squeeze()
            if args.model_name =='pointnet':
                target = to_categorical(target, num_class)
            target = target.long().cuda()
            y_pred = model(data)

            loss = F.nll_loss(y_pred,target)
            history['loss'].append(loss.cpu().data.numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        train_metrics, train_hist_acc = test(model, trainDataLoader) if COMPUTE_TRAIN_METRICS else (None, [])
        test_metrics, test_hist_acc = test(model, testDataLoader)
        total_train_acc += train_hist_acc
        total_test_acc += test_hist_acc

        print('\r Loss: %f' % history['loss'][-1])
        logger.info('Loss: %.2f' , history['loss'][-1])
        if COMPUTE_TRAIN_METRICS:
            print('Train Accuracy: %f' % (train_metrics['accuracy']))
            logger.info('Train Accuracy: %f' , (train_metrics['accuracy']))
        print('\r Test Accuracy: %.2f%%' % test_metrics['accuracy'])
        logger.info('Test Accuracy: %f' , test_metrics['accuracy'])

        if (test_metrics['accuracy'] >= best_tst_accuracy) and epoch > 5:
            best_tst_accuracy = test_metrics['accuracy']
            logger.info('Save model...')
            save_checkpoint(
                global_epoch + 1,
                train_metrics['accuracy'] if COMPUTE_TRAIN_METRICS else 0.0,
                test_metrics['accuracy'],
                model,
                optimizer,
                str(checkpoints_dir)
            )
        global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
