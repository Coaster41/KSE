# coding: utf-8

import torch
import time
import numpy as np
from sklearn.metrics import average_precision_score, f1_score

device = 'cuda:0'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def compute_mAP(labels,outputs):
    AP = []
    for i in range(labels.shape[0]):
        AP.append(average_precision_score(labels[i],outputs[i]))
    return np.mean(AP)

def compute_f1(labels, outputs):
    outputs = outputs > 0.5
    return f1_score(labels, outputs, average="samples")

# def train(model, optimizer, train_loader, args, epoch):
def train_voc(train_loader, model, criterion, optimizer, epoch, writer):

    model.train()
    losses = AverageMeter()
    mAP_meter = AverageMeter()
    f1_meter = AverageMeter()
    print_freq = len(train_loader.dataset) // 256 // 10
    #print_freq = 1
    #import pdb;pdb.set_trace()
    start_time = time.time()
    i = 0 
    for batch, (inputs, targets) in enumerate(train_loader):
        i+=1
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad_()
        optimizer.zero_grad()
        output = model(inputs)
        #adjust_learning_rate(optimizer, epoch, batch, print_freq, args)
        loss = criterion(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()
        #print(pop_config)
        labels_cpu = targets.cpu().detach().numpy()
        outputs_cpu = output.cpu().detach().numpy()
        mAP = compute_mAP(labels_cpu, outputs_cpu)
        # prec1 = utils.accuracy(output, targets)
        mAP_meter.update(mAP, inputs.size(0))
        f1_meter.update(compute_f1(labels_cpu, outputs_cpu), inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            writer.add_scalar('loss', losses.val, i + epoch * len(train_loader))
            writer.add_scalar('mAP', float(mAP_meter.avg), i + epoch * len(train_loader))
            writer.add_scalar('f1', f1_meter.val, i + epoch * len(train_loader))
            print(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'mAP {:.2f}%\t'
                'f1 score {:.2f}\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * 256, len(train_loader.dataset),
                    float(losses.avg), float(mAP_meter.avg)*100, float(f1_meter.avg), cost_time
                )
            )
            start_time = current_time

def validate_voc(testLoader, model, loss_func):
    model.eval()

    losses = AverageMeter()
    mAP = AverageMeter()
    f1 = AverageMeter()

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            labels_cpu = targets.cpu().detach().numpy()
            outputs_cpu = outputs.cpu().detach().numpy()
            mAP.update(compute_mAP(labels_cpu, outputs_cpu), inputs.size(0))
            f1.update(compute_f1(labels_cpu, outputs_cpu), inputs.size(0))

        current_time = time.time()
        print(
            'Test Loss {:.4f}\tmAP {:.2f}%\tf1 score {:.2f}\tTime {:.2f}s\n'
            .format(float(losses.avg), float(mAP.avg*100), float(f1.avg), (current_time - start_time))
        )
    return float(mAP.avg), float(f1.avg)

def train_3(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.to(device)
        target = target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            writer.add_scalar('loss', losses.val, i + epoch * len(train_loader))
            writer.add_scalar('acc', top1.val, i + epoch * len(train_loader))
            writer.add_scalar('top5-acc', top5.val, i + epoch * len(train_loader))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate_3(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.to(device)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def train_4(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.to(device)
        target = target.to(device)
        input.requires_grad_()

        # compute output
        output = model(input)
        loss = criterion(output, target)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            writer.add_scalar('loss', losses.val, i + epoch * len(train_loader))
            writer.add_scalar('acc', top1.val, i + epoch * len(train_loader))
            writer.add_scalar('top5-acc', top5.val, i + epoch * len(train_loader))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate_4(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.to(device)

            input.requires_grad_()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg