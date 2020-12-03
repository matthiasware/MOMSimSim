import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    # same thing, much faster. Scroll down, speed test in __main__
    elif version == 'simplified':
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048, n_hidden_layers=1):
        super().__init__()
        ''' page 3 baseline setting
            Projection MLP. The projection MLP (in f) has BN ap-
            plied to each fully-connected (fc) layer, including its out- 
            put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
            This MLP has 3 layers.
        '''

        # INPUT LAYER
        self.layer_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # HIDDEN LAYERS
        layers_hidden = []
        for _ in range(n_hidden_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True))
            layers_hidden.append(layer)
        self.layers_hidden = nn.Sequential(*layers_hidden)

        # OUTPUT LAYERS, NO RELU!
        self.layer_out = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        # using x.squeeze() will break the program when batch size is 1
        # x.shape [4, 2048, 1, 1] -> [4, 2048]
        # x = self.layer_in(x.squeeze(dim=-1).squeeze(dim=-1))
        x = self.layer_in(x)
        x = self.layers_hidden(x)
        x = self.layer_out(x)
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):
    def __init__(self, backbone, projector_args, predictor_args):
        super().__init__()
        self.projector = projection_MLP(backbone.fc.in_features,
                                        **projector_args)

        # cut output transform as it is ImageNet specific
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.encoder = nn.Sequential(  # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP(**predictor_args)

    def forward(self, x1, x2):
        # x1, x2 are augmented
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)   # projections
        p1, p2 = h(z1), h(z2)   # prediction
        L = 0.5 * D(p1, z2) + 0.5 * D(p2, z1)  # loss
        return L
