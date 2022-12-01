r""" Hypercorrelation Squeeze training (validation) code """
import argparse

import torch.optim as optim
import torch.nn as nn
import torch

# from model.hsnet import HypercorrSqueezeNetwork
from model.PFENet import PFENet
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset


seed = 123

def train(epoch, model, dataloader, optimizer, training,criterion ):
    r""" Train HSNet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(seed) if training else utils.fix_randseed(0)
    model.train() if training else model.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = utils.to_cuda(batch)

        pred_mask,main_loss, aux_loss = model( batch['query_img'],batch['support_imgs'], batch['support_masks'],batch['query_mask'].long())

        # 2. Compute loss & update model parameters
        loss = main_loss + aux_loss
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/home/user4/datasets/VOCdevkit')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--skipnovel', type=int, default=0)
    parser.add_argument('--stop_interval', type=int, default=10)
    parser.add_argument('--optim', type=str, default='adam',choices=['adam', 'sgd','sgd-weight-decay','adam-weight-decay'])
    args = parser.parse_args()
    Logger.initialize(args, training=True)
    print('Logger initialized')

    # Model initialization
    model = PFENet()

    print('model initialized')
    Logger.log_params(model)

    #随机种子注入
    seed = args.seed
    utils.fix_randseed(seed)
    # print('seed')

    # Device setup
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model.cuda(), device_ids=[0])
    model.train()

    # Helper classes (for training) initialization

    if args.optim == 'adam':
        optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    elif args.optim == 'sgd':
        optimizer = optim.SGD([{"params": model.parameters()}],lr=args.lr,momentum=0.9)
    elif args.optim == 'sgd-weight-decay':
        optimizer = optim.SGD([{"params": model.parameters()}],lr=args.lr,momentum=0.9,weight_decay=0.0005)
    elif args.optim == 'adam-weight-decay':
        optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}],weight_decay=0.0005)
    else:
        print('optim error')

    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=473, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn',skip_novel=(args.skipnovel != 0))
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    # Train HSNet
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    prev_best_epoch=-1

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    for epoch in range(args.niter):

        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True,criterion=criterion)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False,criterion=criterion)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
                # Save the best model
        if val_miou > best_val_miou:
            Logger.info(f'### New best !!!! prev best idx:{prev_best_epoch} @ miou:{best_val_miou}')
            Logger.save_model_miou(model, epoch, val_miou)
            prev_best_epoch=epoch
            best_val_miou = val_miou
        else:
            if(epoch-prev_best_epoch>args.stop_interval):
                Logger.info(f'{args.stop_interval} epochs no best, stop train')
                break
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
