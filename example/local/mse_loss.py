import torch
from dacapo.tasks.losses.loss import Loss


class MSELoss(torch.nn.MSELoss, Loss):

    def __init__(self):
        super(MSELoss, self).__init__()
