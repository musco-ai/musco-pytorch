import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os


def save_checkpoint(state, is_best, save_dir='.', suffix=''):
    """
    Save the training self.model
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = "{}/checkpoint{}.pth.tar".format(save_dir, suffix)
    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, '{}/model_best{}.pth.tar'.format(save_dir, suffix))


def rm_checkpoints(save_dir):
    for f_name in os.listdir(save_dir):
        if "checkpoint" in f_name and ".pth.tar" in f_name:
            os.remove(os.path.join(save_dir, f_name))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.values = []

    def update(self, val, n=1):
        self.values += [val]
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_var(self):
        return np.var(self.values)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
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


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def l1_penalty(scaler_params):
    '''
    Args:
        scales - diagonal matrix, which contains weight coefficients for branches in ResNeXt block
    '''
    l1 = torch.mean(torch.tensor([torch.abs(v).sum() for v in scaler_params.values()]))
    return l1


def adjust_learning_rate(optimizer, epoch, lr0=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = lr0 * (0.99 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def validate(val_loader, model, device='cuda',
             iters=10 ** 7, suffix='', prefix="Test",
             loggs_dir=None, print_every=10):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # switch to evaluate mode
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            iter_no = i + 1
            input = input.to(device)
            target = target.to(device)

            # compute output
            begin = time.time()
            output = model(input)
            batch_time.update(time.time() - begin)

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            print_info = ''.join(['{0}: [{1}/{2}]\t',
                                  'ElapsedTime {batch_time.val:.6f} ({batch_time.avg:.6f})\t',
                                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t',
                                  'Prec@1 {top1.val:.6f} ({top1.avg:.6f})\t',
                                  'Prec@5 {top5.val:.6f} ({top5.avg:.6f})\n']).format(prefix,
                                                                                      iter_no,
                                                                                      min(len(val_loader), iters),
                                                                                      batch_time=batch_time,
                                                                                      loss=losses,
                                                                                      top1=top1,
                                                                                      top5=top5)

            if loggs_dir is not None:
                loggs_file = '{}/val_loggs{}.txt'.format(loggs_dir, suffix)
                with open(loggs_file, 'a') as f:
                    f.write(print_info)

            if iter_no % print_every == 0 or iter_no == 1:
                print(print_info)
            if iter_no > 0 and iter_no % iters == 0:
                break

        print(' * Prec@1 {top1.avg:.6f} Prec@5 {top5.avg:.6f}'
              .format(top1=top1, top5=top5))

        return top1.avg, top5.avg, losses.avg


def train(train_loader, model, teacher_model,
          optimizer, epoch, kd_params=None,
          device='cuda', iters=10 ** 7, add_l1_penalty=False,
          suffix='', loggs_dir=None, print_every=20):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.to(device)
    model.train()

    if kd_params is not None:
        teacher_model.to(device)
        teacher_model.eval()

    for i, (input, target) in enumerate(train_loader):
        iter_no = i + 1

        input = input.to(device)
        target = target.to(device)

        # compute output
        begin = time.time()
        output = model(input)

        if kd_params is not None:
            teacher_output = teacher_model(input)
            loss = loss_fn_kd(output, target, teacher_output, kd_params) + loss_fn(output, target)
        else:
            loss = loss_fn(output, target)

        scaler_params = {k: v for k, v in dict(model.named_parameters()).items() if k.endswith('scaler.weight')}
        l1_loss = l1_penalty(scaler_params)

        if add_l1_penalty:
            loss = loss + 0.0001 * l1_loss

        batch_time.update(time.time() - begin)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        print_info = ''.join(['Epoch: [{0}][{1}/{2}]\t',
                              'ElapsedTime {batch_time.val:.6f} ({batch_time.avg:.6f})\t',
                              'LoadingTime {data_time.val:.6f} ({data_time.avg:.6f})\t',
                              'Loss {loss.val:.6f} ({loss.avg:.6f})\t',
                              'Prec@1 {top1.val:.6f} ({top1.avg:.6f})\t',
                              'Prec@5 {top5.val:.6f} ({top5.avg:.6f})\n']).format(epoch + 1,
                                                                                  iter_no,
                                                                                  min(len(train_loader), iters),
                                                                                  batch_time=batch_time,
                                                                                  data_time=data_time,
                                                                                  loss=losses,
                                                                                  top1=top1,
                                                                                  top5=top5)
        if loggs_dir is not None:
            loggs_file = '{}/train_loggs{}.txt'.format(loggs_dir, suffix)
            with open(loggs_file, 'a') as f:
                f.write(print_info)

        if iter_no % print_every == 0 or iter_no == 1:
            print(print_info)
        if iter_no > 0 and iter_no % iters == 0:
            break
