from pathlib import Path
import datetime
#
from dotted_dict import DottedDict
import torch


def add_paths(config, debug=False):
    # unique identifier for run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fs_run = "run_{}_{}_{}".format(config.dataset, config.backbone, timestamp)
    config.p_data = Path("/mnt/data/pytorch")
    if debug:
        config.p_train = Path("/mnt/experiments/tmp") / fs_run
    else:
        config.p_train = Path("/mnt/experiments/simsiam") / fs_run
    config.p_ckpts = config.p_train / "ckpts"
    config.p_logs = config.p_train / "logs"
    return config


def simsiam_default(debug=False):
    config = DottedDict()
    config.fs_ckpt = "model_{}_epoch_{:0>6}.ckpt"
    config.mean_std = [[0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]]
    config.dataset = "imagenet"
    config.backbone = "resnet50"
    config.batch_size = 512
    config.num_epochs = 100
    config.img_size = 224
    config.projector_args = {
        "hidden_dim": 2048,
        "out_dim": 2048,
        "n_hidden_layers": 1
    }
    config.predictor_args = {
        "hidden_dim": config.projector_args["out_dim"] // 4,  # see Appendix B.
        "in_dim": config.projector_args["out_dim"],
        "out_dim": config.projector_args["out_dim"]
    }
    config.optimizer = "sgd"
    config.base_lr = 0.05
    config.optimizer_args = {
        "lr": config.base_lr * (config.batch_size / 256),
        "weight_decay": 0.0001,  # used always
        "momentum": 0.9
    }
    config.scheduler = "cosine_decay"
    config.scheduler_args = {
        "T_max": config.num_epochs,
        "eta_min": 0,
    }
    config.debug = False
    config.num_workers = 8
    config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config.resume = False
    #
    # Frequencies (epochs)
    #
    config.freq_knn = 1
    #
    # debug settings
    if config.debug:
        config.batch_size = 2
        config.num_epochs = 5  # train only one epoch
        config.num_workers = 1

    return config


def linear_default(debug=False):
    config = DottedDict()
    config.optimizer = "sgd"
    config.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    config.batch_size = 256
    config.base_lr = 0.5
    config.num_workers = 8
    config.num_epochs = 100
    config.optimizer_args = {
        "lr": config.base_lr,
        "weight_decay": 5e-4,  # adapted, paper says 0
        "momentum": 0.9
    }
    config.scheduler = "cosine_decay"
    config.scheduler_args = {
        "T_max": config.num_epochs,
        "eta_min": 0
    }
    return config


def simsiam_imagenet(debug=False):
    return simsiam_default(debug)


def linear_cifar10(debug=False):
    return linear_default()


def simsiam_cifar10(debug=False):
    config = simsiam_default(debug)
    config.backbone = "resnet18"
    config.dataset = "cifar10"
    config.batch_size = 512
    config.num_epochs = 800
    config.base_lr = 0.03
    config.optimizer = "sgd"
    config.optimizer_args = {
        "lr": config.base_lr * (config.batch_size / 256),
        "weight_decay": 0.0005,
        "momentum": 0.9
    }
    config.scheduler = "cosine_decay"
    config.scheduler_args = {
        "T_max": config.num_epochs,
        "eta_min": 0,
    }
    config.projector_args["n_hidden_layers"] = 0
    config.img_size = 32
    return config


def get_config(dataset, train=True, debug=False, device=None):
    if dataset == "cifar10":
        if train:
            config = simsiam_cifar10()
        else:
            config = linear_cifar10()
    elif dataset == "imagenet":
        config == simsiam_imagenet()
    if train:
        config = add_paths(config, debug)
    if device:
        config.device = device
    return config
