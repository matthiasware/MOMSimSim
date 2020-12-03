import torchvision.transforms as T
from PIL import Image


class SimSiamAugmentations():
    def __init__(self, img_size, means_std=None):
        if means_std is None:
            normalize = lambda x: x
        else:
            normalize = T.Normalize(*means_std)
        # Not in SimSiam paper
        # see SimCLR: https://arxiv.org/pdf/2002.05709.pdf Appendix A
        p_blur = 0.5 if img_size > 32 else 0 
        #
        self.transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=img_size // 20 * 2 + 1,
                                          sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


class LinearProbAugmentations():
    def __init__(self, img_size:int, train:bool, means_std=None):
        if means_std is None:
            normalize = lambda x: x
        else:
            normalize = T.Normalize(*means_std)
        #
        if train:
            self.transform = T.Compose([
                # T.Resize(int(img_size*(8/7)), interpolation=Image.BICUBIC),
                T.RandomResizedCrop(img_size, scale=(0.08, 1.0),
                                    ratio=(3.0 / 4.0,
                                           4.0 / 3.0),
                                    interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transform = T.Compose([
                T.Resize(int(img_size * (8 / 7)),
                         interpolation=Image.BICUBIC),   # 224 -> 256
                T.CenterCrop(img_size),
                T.ToTensor(),
                normalize
            ])

    def __call__(self, x):
        return self.transform(x)


def get_aug(img_size:int, train:bool, train_classifier:bool=True,
            means_std=None):
    if train_classifier:
        augmentation = LinearProbAugmentations(img_size, train=train,
                                               means_std=means_std)
    else:
        augmentation = SimSiamAugmentations(img_size, means_std)
    return augmentation
