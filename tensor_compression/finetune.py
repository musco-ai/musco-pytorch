import torch
from .train_validate import train, validate, save_checkpoint

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def adjust_momentum(optimizer, epoch, momentum0):
    """Sets the momentum to the initial momentum decayed by 10 every 30 epochs"""
    momentum = min(0.9, 1 - momentum0 * (0.9 ** (epoch)))
    for param_group in optimizer.param_groups:
        param_group['momentum'] = momentum


def adjust_learning_rate_local(optimizer, epoch, lr0):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #     if lr0<=1e-4:
    #         if epoch < 90:
    #             lr = lr0 * (0.99**(epoch//30))
    #         elif epoch < 180:
    #             lr = lr0 * (0.9**(epoch//30))
    #         elif epoch < 270:
    #             lr = lr0 * (0.7**(epoch//30))

    #     if lr0==1e-3*0.5:
    #         if epoch < 90:
    #             lr = lr0 * (0.9**(epoch//7))
    #         elif epoch < 180:
    #             lr = lr0 * (0.9**(epoch//7))
    #         elif epoch < 270:
    #             lr = lr0 * (0.88**(epoch//7))

    #     lr = lr0 * (0.99**(epoch//30))
    lr = lr0 * (0.1 ** epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fine_tune(model, loaders, device='cuda', save_dir=None,
              batches_per_train=100, batches_per_val=100, suffix='',
              ft_epochs=20, add_l1_penalty=False, loggs_dir=None, patience=3):
    lr0 = 1e-5
    momentum0 = 0.99
    weight_decay = 5e-4
    kd_params = None

    optimizer = torch.optim.SGD(model.parameters(), lr0,
                                momentum=momentum0,
                                weight_decay=weight_decay,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                               T_max=1,
                                               eta_min=1e-6,
                                               last_epoch=-1)
    # criterion = torch.nn.CrossEntropyLoss().to(device)

    best_prec1 = 0
    start_epoch = 0
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(start_epoch, ft_epochs):
        i = 0

        if kd_params is not None:
            train(loaders['train'], model, teacher_model, optimizer, epoch, kd_params, device, batches_per_train,
                  add_l1_penalty=add_l1_penalty, suffix=suffix, loggs_dir=loggs_dir)
        else:
            train(loaders['train'], model, None, optimizer, epoch, kd_params, device, batches_per_train,
                  add_l1_penalty=add_l1_penalty, suffix=suffix, loggs_dir=loggs_dir)

        for param_group in optimizer.param_groups:
            print("Optimizer params", param_group["lr"], epoch, i)

        # print("???", epoch)
        # evaluate on validation set
        prec1, _, val_loss = validate(loaders['val'], model, device, batches_per_val, suffix=suffix, loggs_dir=loggs_dir)
        scheduler.step(val_loss)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(model, is_best, save_dir, suffix)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def fine_tune_adv(model, loaders, device='cuda', save_dir=None,
                  train_iters=100, val_iters=100, suffix='',
                  ft_epochs=20, add_l1_penalty=False, loggs_dir=None):
    lr0 = 1e-3
    momentum0 = 0.99
    weight_decay = 5e-4
    kd_params = None

    from copy import deepcopy
    old_model = deepcopy(model)



    optimizer = torch.optim.SGD(model.parameters(), lr0,
                                momentum=momentum0,
                                weight_decay=weight_decay,
                                nesterov=True)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    best_prec1 = 0
    start_epoch = 0

    for epoch in range(start_epoch, ft_epochs):

        adjust_learning_rate_local(optimizer, epoch, lr0)
        adjust_momentum(optimizer, epoch, momentum0)

        prec1_old, prec5_old, _ = validate(loaders['val'], old_model, device, val_iters, suffix=suffix,
                                           loggs_dir=loggs_dir,
                                           prefix="Before_fine", )

        # train for one epoch
        if kd_params is not None:
            train(loaders['train'], model, teacher_model, optimizer, epoch, kd_params, device, train_iters,
                  add_l1_penalty=add_l1_penalty, suffix=suffix, loggs_dir=loggs_dir)
        else:
            train(loaders['train'], model, None, optimizer, epoch, kd_params, device, train_iters,
                  add_l1_penalty=add_l1_penalty, suffix=suffix, loggs_dir=loggs_dir)

        # evaluate on validation set

        prec1, prec5, _ = validate(loaders['val'], model, device, val_iters, suffix=suffix, loggs_dir=loggs_dir)

        if prec5_old > prec5:
            print("Achtung bitte! Fine tuned model is worther that the original-one. Rolling back.")
            model, prec1, prec5 = old_model, prec1_old, prec5_old
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(model, is_best, save_dir, suffix)

        old_model = deepcopy(model)
