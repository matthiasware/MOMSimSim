# python
from pathlib import Path
import time

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# non torch
from dotted_dict import DottedDict
import pprint
from tqdm import tqdm

# local
from utils import AverageMeter, get_dataset, get_backbone, get_optimizer, get_scheduler
from augmentations import get_aug
from model import SimSiam
import utils
import configs

pp = pprint.PrettyPrinter(indent=4)

p_ckpt = Path(
    "/mnt/experiments/simsiam/run_cifar10_resnet18_20201203-185153/ckpts/model_cifar10_epoch_000600.ckpt")
assert p_ckpt.exists()

ckpt = torch.load(p_ckpt)

train_config = ckpt["config"]
pp.pprint(train_config)

config = configs.get_config(train_config.dataset,train=False)
pp.pprint(config)


# prepare data
train_set = get_dataset(
    train_config.dataset,
    train_config.p_data,
    transform=get_aug(train_config.img_size, train=True, train_classifier=True),
    train=True,
    download=False
)
if train_config.dataset == "stl10":
    # stl10 has only 5000 labeled samples in its train set
    train_set = torch.utils.data.Subset(train_set, range(0, 5000))

test_set = get_dataset(
    train_config.dataset,
    train_config.p_data,
    transform=get_aug(train_config.img_size, train=True, train_classifier=True),
    train=False,
    download=False
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True,
    drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=True,
    drop_last=True
)

print("Batches train set: {}".format(len(train_loader)))
print("Barches test set:  {}".format(len(test_loader)))


# model
backbone = get_backbone(train_config.backbone)
model = SimSiam(backbone, train_config.projector_args, train_config.predictor_args)
msg = model.load_state_dict(ckpt["state_dict"], strict=True)
print("Loading weights: {}".format(msg))
model = model.encoder
model = model.to(config.device)

# classifier
classifier = nn.Linear(in_features=train_config.projector_args["out_dim"],
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