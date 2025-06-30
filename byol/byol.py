"""

# augmentation => simclr와 동일

# backbone encoder => VGGNet으로 동일

# projector => simclr에 있는 MLP 부분 가져오기

# predictor (online만 해당)


# 1. x1, x2 => augmented view
# 2. online network: x1 -> y1 -> z1 -> q1
# 3. target network" x2 -> y2 -> z2

# 4. loss 계산: L2 norm -> cos sim between q1, z2) => q1이 z2를 예측하도록 학습

# 5. paramter update: online network => backpropagation 하고, target network => EMA로 online parameter 따라가기
# 5.1 EMA: target = decay * target + (1-decay) * online

"""


import copy
import torch
from torch import nn
import torch.nn.functional as F


from byol.augmentation import SigAugmentation
from byol.utils import *


# MLP class for projector and predictor
def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    layers = nn.Sequential(
        nn.Linear(dim, hidden_size),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),  # BatchNorm
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

    return layers


def SimSiamMLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(projection_size, affine=False)
    )


class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        sync_batchnorm = None
    ):
        super(BYOL, self).__init__()
        self.net = net

        # default SimCLR augmentation
        self.aug1, self.aug2 = aug.convert_augmentation

        # Online Encoder
        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size,
            layer = hidden_layer,
            use_simsiam_mlp = not use_momentum,
            sync_batchnorm = sync_batchnorm
        )

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)


    # Target Encoder
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

   def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True
    ):

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)

        x1, x2 = self.augment1(x), self.augment2(x)

        x_ = torch.cat((x1, x2), dim = 0)

        online_projections, _ = self.online_encoder(x_)
        online_predictions = self.online_predictor(online_projections)

        online_y1, online_y2 = online_predictions.chunk(2, dim = 0)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder

            target_projections, _ = target_encoder(x_)
            target_projections = target_projections.detach()

            target_y1, target_y2 = target_projections.chunk(2, dim = 0)

        loss1 = loss_fn(online_y1, target_y1.detach())
        loss2 = loss_fn(online_y2, target_y2.detach())

        loss = loss1 + loss2
        return loss.mean()  # MSE


# EMA
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2*(x*y).sum(dim=-1)


if __name__ == '__main__':
    aug = SigAugmentation()

