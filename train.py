
#
import pprint
#
import configs

import numpy as np
import torch
from torchvision.models import resnet50
from tqdm import tqdm
from pathlib import Path
from model import SimSiam
from utils import AverageMeter, get_dataset, get_backbone, get_optimizer, get_scheduler
from augmentations import get_aug
from dotted_dict import DottedDict
import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier

DEBUG = False
DEVICE = "cuda:1"
DATASET = "cifar10"

config = configs.get_config(dataset=DATASET,
                            train=True,
                            debug=DEBUG,
                            device=DEVICE)


pp = pprint.PrettyPrinter(indent=2)

pp.pprint(config)

# prepare data
train_transform = get_aug(img_size=config.img_size,
                          train=True,
                          train_classifier=False,
                          means_std=config.mean_std)

classifier_transform = get_aug(img_size=config.img_size,
                               train=False,
                               train_classifier=True,
                               means_std=config.mean_std)
#
train_set = get_dataset(config.dataset, config.p_data,
                        train=True, transform=train_transform)
classifier_set = get_dataset(config.dataset, config.p_data,
                             train=False, transform=classifier_transform)
#
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True,
    drop_last=True
)

classifier_loader = torch.utils.data.DataLoader(
    dataset=classifier_set,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True,
    drop_last=True
)

# model
backbone = get_backbone(config.backbone)
model = SimSiam(backbone, config.projector_args, config.predictor_args)
model = model.to(config.device)

print(model)

optimizer = get_optimizer(config.optimizer, model, config.optimizer_args)

# define lr scheduler
lr_scheduler = get_scheduler(
    config.scheduler, optimizer, config.scheduler_args)

loss_meter = AverageMeter("loss")

writer = SummaryWriter(config.p_logs)

# create train dir
config.p_logs.mkdir(exist_ok=True, parents=True)
config.p_ckpts.mkdir(exist_ok=True, parents=True)
#
# tensorboard writer
writer = SummaryWriter(config.p_logs)
print("tensorboard --logdir={} --host=0.0.0.0".format(str(config.p_logs)))
#
for epoch in range(1, config.num_epochs + 1):
    #
    # TRAIN LOOP
    #
    loss_meter.reset()
    model.train()
    p_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.num_epochs}')
    for idx, ((images1, images2), labels) in enumerate(p_bar):
        model.zero_grad()
        loss = model.forward(images1.to(config.device),
                             images2.to(config.device))
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        #
        writer.add_scalar('loss', loss_meter.val,
                          epoch * len(train_loader) + idx)
        writer.add_scalar('avg_loss', loss_meter.avg,
                          epoch * len(train_loader) + idx)
        p_bar.set_postfix({"loss": loss_meter.val, 'loss_avg': loss_meter.avg, 'lr': lr_scheduler.get_last_lr()[0]})

    writer.add_scalar('epoch_avg', loss_meter.avg, epoch)
    lr_scheduler.step()

    # Save checkpoint
    p_ckpt = config.p_ckpts / config.fs_ckpt.format(config.dataset, epoch)
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        # 'optimizer':optimizer.state_dict(), # will double the checkpoint file size
        'lr_scheduler': lr_scheduler.state_dict(),
        'config': config,
        'loss_meter': loss_meter
    }, p_ckpt)
    print(f"Model saved to {p_ckpt}")
    #
    # CLASSIFIER LOOP
    #
    if epoch % config.freq_knn == 0:
        model.eval()
        all_projections = []
        all_labels = []
        for images, labels in classifier_loader:
            with torch.no_grad():
                z = model.encoder(images.to(config.device))
                all_projections.append(z.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_labels = np.concatenate(all_labels)
        all_projections = np.concatenate(all_projections)
        
        # knn 
        neigh = KNeighborsClassifier(n_neighbors=5, algorithm='brute', n_jobs=8)
        neigh.fit(all_projections, all_labels)
        score = neigh.score(all_projections, all_labels)
        writer.add_scalar('knn_acc', score, epoch)

        # std
        # ideally around 1/np.sqrt(d)
        norms = np.linalg.norm(all_projections, axis=1)
        z_bars = all_projections / norms[:, None]
        std = z_bars.std(axis=0).mean()

        writer.add_scalar('std', std, epoch)
