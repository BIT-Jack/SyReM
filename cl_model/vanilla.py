import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt

def get_parser(parser):
    return parser



class Vanilla(nn.Module):
    NAME = 'vanilla'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args):
        super(Vanilla, self).__init__()
        self.net = backbone
        self.loss = loss
        self.args = args
        self.device = args.device
        self.transform = None
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)

    def observe(self, inputs, labels, task_id=None, record_list=None):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        log_lanescore, heatmap, heatmap_reg = outputs

        outputs_prediction = [log_lanescore, heatmap, heatmap_reg]
        loss = self.loss(outputs_prediction, labels) 

        loss.backward()
        self.opt.step()

        return loss.item()

