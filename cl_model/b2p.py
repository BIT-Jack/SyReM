import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from utils.der_gssp_buffer import Buffer
from utils.derpp_buffer import Buffer as Buffer_RSVR

import matplotlib.pyplot as plt

def get_parser(parser):
    return parser



class B2P(nn.Module):
    NAME = 'b2p'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args):
        super(B2P, self).__init__()
        self.net = backbone
        self.loss = loss
        self.args = args
        self.device = args.device
        self.transform = None
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)
        # the summation of rapid memory buffer size and distinctive memory buffer size is the buffer size of B2P.
        self.buffer = Buffer(int(self.args.buffer_size/2), self.device, minibatch_size=8, model_name=self.NAME,model=self) #distinctive memory buffer
        self.buffer_r = Buffer_RSVR(int(self.args.buffer_size/2), self.device, self.NAME) #rapid memory buffer

    #gss function get_grads
    def get_grads(self, inputs, labels):
        self.net.eval()
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        grads = self.net.get_grads().clone().detach()
        self.opt.zero_grad()
        self.net.train()
        if len(grads.shape) == 1:
            grads = grads.unsqueeze(0)
        return grads


    def observe(self, inputs, labels, task_id=None):

        self.buffer.drop_cache()
        self.buffer.reset_fathom()


        self.opt.zero_grad()
        outputs = self.net(inputs)
        log_lanescore, heatmap, heatmap_reg = outputs

        outputs_prediction = [log_lanescore, heatmap, heatmap_reg]
        loss = self.loss(outputs_prediction, labels) # OverallLoss

   
        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, give_index=False)
            buf_outputs = self.net(buf_inputs) 
            buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg = buf_outputs
            
            loss += self.args.alpha*F.mse_loss(buf_heatmap_logits, buf_logits) # heatmaps are logits 
  
            del buf_inputs, buf_logits, buf_outputs, buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg
            torch.cuda.empty_cache()

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, give_index=False)
            buf_outputs = self.net(buf_inputs)
            loss += self.loss(buf_outputs, buf_labels) 
        
        if not self.buffer_r.is_empty():
            buf_inputs, _, buf_logits = self.buffer_r.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg = buf_outputs
            loss += self.args.alpha * F.mse_loss(buf_heatmap_logits, buf_logits)
            
            del buf_inputs, buf_logits, buf_outputs, buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg
            torch.cuda.empty_cache()

            buf_inputs, buf_labels, _ = self.buffer_r.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            #set the beta as 1
            loss +=  self.loss(buf_outputs, buf_labels)



        loss.backward()
        self.opt.step()
        
        if not self.buffer.is_empty:
            del buf_inputs, buf_outputs, buf_labels
            torch.cuda.empty_cache()

        # recording the update of memory samples
        if task_id is not None:
            self.buffer.add_data(examples=inputs, labels=labels, logits=heatmap.detach(), task_order=task_id)
            self.buffer_r.add_data(examples=inputs, labels=labels, logits=heatmap.detach(), task_order=task_id)
        else:
            self.buffer.add_data(examples=inputs, labels=labels, logits=heatmap.detach())
            self.buffer_r.add_data(examples=inputs, labels=labels, logits=heatmap.detach())
        return loss.item()

