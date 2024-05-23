# coding: utf-8

import torch
import torch.nn as nn
import argparse
import importlib
import copy
from tensorboardX import SummaryWriter
import pdb
from model.resnet_voc import resnet34
import voc_dataset

from utils import base, models

parser = argparse.ArgumentParser(description='KSE Experiments')
parser.add_argument('--dataset', dest='dataset', help='training dataset', default='cifar10', type=str)
parser.add_argument('--net', dest='net', help='training network', default='resnet56', type=str)
parser.add_argument('--pretrained', dest='pretrained', help='whether use pretrained model', default=False, type=bool)
parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint dir', default=None, type=str)
parser.add_argument('--train_dir', dest='train_dir', help='training data dir', default="tmp", type=str)
parser.add_argument('--save_best', dest='save_best', help='whether only save best model', default=True, type=bool)
# parser.add_argument('--pretrain_dir', dest='pretrain_dir', help='training data dir', default="tmp", type=str)

parser.add_argument('--train_batch_size', dest='train_batch_size', help='training batch size', default=64, type=int)
parser.add_argument('--test_batch_size', dest='test_batch_size', help='test batch size', default=50, type=int)
parser.add_argument('--gpus', dest='gpus', help='gpu id',default=[0], type=int,nargs='+')

parser.add_argument('--learning_rate', dest='learning_rate', help='learning rate', default=0.01, type=float)
parser.add_argument('--momentum', dest='momentum', help='momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', dest='weight_decay', help='weight decay', default=1e-5, type=float)
parser.add_argument('--epochs', dest='epochs', help='epochs', default=200, type=int)
parser.add_argument('--schedule', dest='schedule', help='Decrease learning rate',default=[100, 150],type=int,nargs='+')
parser.add_argument('--gamma', dest='gamma', help='gamma', default=0.1, type=float)

parser.add_argument('--G', dest='G', help='G', default=None, type=int)
parser.add_argument('--T', dest='T', help='T', default=None, type=int)

args = parser.parse_args()
device = torch.device(f"cuda:0")

class VocModel(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        # Use a pretrained model
        self.network = resnet34(weights=weights, kse=True)
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)

if __name__ == "__main__":
    # model = importlib.import_module("model.model_deploy").__dict__[args.net](args.pretrained, args.checkpoint)
    model = VocModel(num_classes=20).to(device)
    # model.load_state_dict(torch.load(args.pretrain_dir), strict=False)

    loader = voc_dataset.Data()
    train_loader, test_loader = loader.loader_train, loader.loader_test

    writer = SummaryWriter(args.train_dir)

    # KSE
    models.KSE(model, args.G, args.T)

    # network forward init
    models.forward_init(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(filter(lambda i: i.requires_grad, model.parameters()), args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    if torch.cuda.is_available():
        criterion = criterion.cuda(args.gpus[0])
        model = model.cuda(args.gpus[0])

    if len(args.gpus) != 1:
        model = torch.nn.DataParallel(model, args.gpus).cuda()

    if len(args.gpus) != 1:
        model_tmp = copy.deepcopy(model.module)
    else:
        model_tmp = copy.deepcopy(model)
    models.save(model_tmp)
    torch.save(model_tmp.state_dict(), args.train_dir + "/model.pth")

    # if torch.__version__ == "0.3.1":
    #     train = base.train_3
    #     validate = base.validate_3
    # else:
    #     train = base.train_4
    #     validate = base.validate_4
    train = base.train_voc
    validate = base.validate_voc

    best_mAP = 0
    for i in range(args.epochs):
        train(train_loader, model, criterion, optimizer, i, writer)
        mAP, f1_score = validate(test_loader, model, criterion)
        lr_scheduler.step()

        if len(args.gpus) != 1:
            model_tmp = copy.deepcopy(model.module)
        else:
            model_tmp = copy.deepcopy(model)
        models.save(model_tmp)

        if args.save_best:
            if best_mAP < mAP:
                torch.save(model_tmp.state_dict(), args.train_dir+"/model_best.pth")
                best_mAP = mAP
        else:
            torch.save(model_tmp.state_dict(), args.train_dir + "/model_"+str(i)+".pth")

        writer.add_scalar('val-mAP', mAP, i)
        writer.add_scalar('val-f1_score', f1_score, i)

    print("best acc: {:.2f}".format(best_mAP))


