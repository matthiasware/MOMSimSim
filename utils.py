import torchvision
from torchvision.models import resnet50, resnet18, vgg16
import torch

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.log = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.log.append(self.avg)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    
def get_dataset(dataset, data_dir, transform, train=True, download=False):
    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train,
                                             transform=transform, download=download)
    elif dataset == 'stl10':
        dataset = torchvision.datasets.STL10(data_dir, 
                                             split='train+unlabeled' if train else 'test', 
                                             transform=transform, download=download)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, 
                                               transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, 
                                                transform=transform, download=download)
    elif dataset == 'imagenet':
        raise Exception("To much memory")
        dataset = torchvision.datasets.ImageNet(data_dir, 
                                                split='train' if train == True else 'val',
                                                transform=transform, download=download)
    else:
        raise Exception("Dataset {} does not exist!".format(dataset))
    return dataset

def get_backbone(backbone:str):
    if backbone == "resnet50":
        return resnet50()
    elif backbone == "resnet18":
        return resnet18()
    elif backbone == "vgg16":
        return vgg16()
    else:
        raise Exception(f"Backbone '{backbone}' does not exist!")


def get_optimizer(optimizer:str, model, optimizer_args):
    if optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), **optimizer_args)
    elif optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **optimizer_args)
    else:
        raise Exception(f"Optimizer '{optimizer}' does not exist!")

        
def get_scheduler(scheduler:str, optimizer, scheduler_args):
    if scheduler == "cosine_decay":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
    else:
        raise Exception(f"Scheduler '{scheduler}' does not exist!")