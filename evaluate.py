#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import torch
from augmentations import get_aug
from utils import get_dataset
from torchvision.models import resnet50
from model import SimSiam
import torch.nn as nn
import torch.nn.functional as F
from utils import AverageMeter
from tqdm import tqdm
import time


# In[ ]:


from pathlib import Path
#
import torch
import torch.nn as nn
import torch.nn.functional as F
#
from dotted_dict import DottedDict
import pprint
from tqdm import tqdm
#
from utils import AverageMeter, get_dataset, get_backbone, get_optimizer, get_scheduler
from augmentations import get_aug
from model import SimSiam


# In[ ]:


pp = pprint.PrettyPrinter(indent=4)


# In[ ]:


p_ckpt = Path(
    "/usr/experiments/simsiam/run_stl10_resnet18_20201203-102151/ckpts/model_stl10_epoch_0000010.ckpt")
assert p_ckpt.exists()


# In[ ]:


ckpt = torch.load(p_ckpt)


# In[ ]:


train_config = ckpt["config"]
pp.pprint(train_config)


# In[ ]:


config = DottedDict()
config.device = 'cuda:1'
config.optimizer = 'sgd'
config.optimizer_args = {
    'lr': 0.1,
    #'weight_decay': 0,
    #'momentum': 0.9
}
config.batch_size = 256
config.img_size = train_config.img_size
config.debug = False
config.num_workers = 8
config.num_epochs = 800


# prepare data
train_set = get_dataset(
    train_config.dataset,
    train_config.p_data,
    transform=get_aug(config.img_size, train=True, train_classifier=True),
    train=True,
    download=False
)
if train_config.dataset == "stl10":
    train_set = torch.utils.data.Subset(train_set, range(0, 5000))

test_set = get_dataset(
    train_config.dataset,
    train_config.p_data,
    transform=get_aug(config.img_size, train=True, train_classifier=True),
    train=False,
    download=True  # default is False
)
if config.debug:
    train_set = torch.utils.data.Subset(train_set, range(
        0, config.batch_size))  # take only one batch
    test_set = torch.utils.data.Subset(test_set, range(0, config.batch_size))

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True,
    drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=True,
    drop_last=True
)


# load backbone
backbone = get_backbone(train_config["backbone"])
in_features = backbone.fc.in_features

backbone.fc = nn.Identity()
model = backbone

state_dict = {k[9:]: v for k, v in ckpt['state_dict'].items() if k.startswith('backbone.')}
model.load_state_dict(state_dict, strict=True)
model = model.to(config.device)


# import pdb
# pdb.set_trace()

# classifier
classifier = nn.Linear(in_features=in_features,
                       out_features=len(test_set.classes), bias=True)
classifier = classifier.to(config.device)


optimizer = get_optimizer(config.optimizer, classifier, config.optimizer_args)

loss_meter = AverageMeter(name='Loss')
acc_meter = AverageMeter(name='Accuracy')

for epoch in range(1, config.num_epochs + 1):
    #
    # TRAIN LOOP
    #
    loss_meter.reset()
    model.eval()
    classifier.train()
    p_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.num_epochs}')
    for idx, (images, labels) in enumerate(p_bar):
        classifier.zero_grad()
        with torch.no_grad():
            feature = model(images.to(config.device))
        preds = classifier(feature)
        #
        loss = F.cross_entropy(preds, labels.to(config.device))
        optimizer.step()
        loss_meter.update(loss.item())
        p_bar.set_postfix({"loss": loss_meter.val, 'loss_avg': loss_meter.avg})
    #
    # EVAL LOOP
    #
    classifier.eval()
    correct, total = 0, 0
    acc_meter.reset()
    p_bar = tqdm(test_loader, desc=f'Test {epoch}/{config.num_epochs}')
    for idx, (images, labels) in enumerate(p_bar):
        with torch.no_grad():
            feature = model(images.to(config.device))
            preds = classifier(feature).argmax(dim=1)
            correct = (preds == labels.to(config.device)).sum().item()
            acc_meter.update(correct / preds.shape[0])
            p_bar.set_postfix({'accuracy': acc_meter.avg})