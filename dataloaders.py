import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def get_batch(dataloader):
    batch = dataloader.__iter__().__next__()
    if isinstance(batch, list):
        batch = batch[0]
    return batch


def sample_dataloader(train_loader, n_samples):
    for _x, _y in train_loader:
        s = _x.shape
        break
    X = torch.zeros(n_samples, s[1], s[2], s[3])
    y = torch.zeros(n_samples, dtype=_y.dtype)
    idx = 0
    for batch in train_loader:
        if isinstance(batch, list):
            batch_x, batch_y = batch
        else:
            batch_x, batch_y = batch, None
        bs = batch_x.shape[0]
        X[idx: min(idx + bs, n_samples)] = batch_x[:min(bs, n_samples - idx)]
        if batch_y is not None:
            y[idx: min(idx + bs, n_samples)] = batch_y[:min(bs, n_samples - idx)]
        idx += bs
        if idx >= n_samples:
            break
    if batch_y is not None:
        dataset = torch.utils.data.TensorDataset(X, y)
    else:
        dataset = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=train_loader.batch_size)
    return dataloader


def get_loader(batch_size, data_name, data_root=None,
               train=True, val=True,
               simple_normalize=False,
               **kwargs):
    def target_transform_svhn(target):
        # return int(target[0]) - 1
        return target - 1

    kwargs.pop('input_size', None)

    # num_workers = kwargs.setdefault('num_workers', 1)
    num_workers = kwargs['num_workers']
    print("Building {} data loader with {} workers".format(data_name, num_workers))

    if data_name in ['imagenet', 'tiny-imagenet-200']:
        traindir = os.path.join(data_root, data_name, 'train')
        valdir = os.path.join(data_root, data_name, 'val')
    else:
        data_root = os.path.expanduser(os.path.join(data_root,
                                                    '{}'.format(data_name)))

    dataloaders = {}

    if data_name == 'mnist':
        normalize = transforms.Normalize((0.1307,), (0.3081,))
    elif data_name in ['imagenet', 'tiny-imagenet-200', 'cifar10']:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if simple_normalize and data_name == 'cifar10':
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if val:
        if data_name == ('cifar10'):
            val_dataset = datasets.CIFAR10(root=data_root, train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               normalize,
                                           ]))
        elif data_name == ('cifar100'):
            val_dataset = datasets.CIFAR100(root=data_root, train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                normalize,
                                            ]))
        elif data_name == 'mnist':
            val_dataset = datasets.MNIST(root=data_root, train=False, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             normalize
                                         ]))
        elif data_name == 'svhn':
            val_dataset = datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]),
                target_transform=target_transform_svhn)

        elif data_name == 'stl10':
            val_dataset = datasets.STL10(
                root=data_root, split='test', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))

        elif data_name == 'tiny-imagenet-200':
            val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(64),
                # transforms.CenterCrop(54),
                transforms.ToTensor(),
                normalize,
            ]))
        elif data_name == 'imagenet':
            val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size, shuffle=False, **kwargs)
        dataloaders['val'] = val_loader

    if train:
        if data_name == ('cifar10'):
            train_dataset = datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

        elif data_name == ('cifar100'):
            train_dataset = datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
        elif data_name == 'mnist':
            train_dataset = datasets.MNIST(root=data_root, train=True, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               normalize
                                           ]))
        elif data_name == 'svhn':
            train_dataset = datasets.SVHN(
                root=data_root, split='train', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]),
                target_transform=target_transform_svhn)

        elif data_name == 'stl10':
            train_dataset = datasets.STL10(
                root=data_root, split='train', download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(96),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

        elif data_name == 'tiny-imagenet-200':
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    # transforms.RandomResizedCrop(54),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
        elif data_name == 'imagenet':
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True, **kwargs)
        dataloaders['train'] = train_loader

    return dataloaders
