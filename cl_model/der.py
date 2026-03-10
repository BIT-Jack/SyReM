import torch
from torch.nn import functional as F
from torch import nn
# from utils.der_buffer import Buffer
from utils.baseline_buffer import Buffer
from torch.optim import Adam
import matplotlib.pyplot as plt

def get_parser(parser):
    return parser



class Der(nn.Module):
    NAME = 'der'
    COMPATIBILITY = ['domain-il']
    def __init__(self, backbone, loss, args):
        super(Der, self).__init__()
        self.net = backbone
        self.loss = loss
        self.args = args
        self.device = args.device
        self.transform = None
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)
        self.buffer = Buffer(self.args.buffer_size, self.device, self.NAME)

    def observe(self, inputs, labels, task_id=None, record_list=None):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        log_lanescore, heatmap, heatmap_reg = outputs
        


        outputs_prediction = [log_lanescore, heatmap, heatmap_reg]
        loss = self.loss(outputs_prediction, labels) #OverallLoss in UQnet


        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs) 
            buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg = buf_outputs
            
            loss += self.args.alpha*F.mse_loss(buf_heatmap_logits, buf_logits) #heatmaps are logits

        loss.backward()
        self.opt.step()
        
        #clean buf temp
        if not self.buffer.is_empty():
            del buf_outputs, buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg
            torch.cuda.empty_cache()
            
        if task_id is not None and record_list is not None:
            self.buffer.add_data(examples=inputs, logits=heatmap.detach(), task_order=task_id, record_data_list=record_list)
        else:
            self.buffer.add_data(examples=inputs, logits=heatmap.detach())
        return loss.item()
