import torch
from torch.nn import functional as F
from torch import nn
# from utils.der_buffer import Buffer
from utils.derpp_buffer_new import Buffer
from torch.optim import Adam
import matplotlib.pyplot as plt

def get_parser(parser):
    return parser



class DerNew(nn.Module):
    NAME = 'der_new'
    COMPATIBILITY = ['domain-il']
    def __init__(self, backbone, loss, args):
        super(DerNew, self).__init__()
        self.net = backbone
        self.loss = loss
        self.args = args
        self.device = args.device
        self.transform = None
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)
        self.buffer = Buffer(self.args.buffer_size, self.device, self.args.gss_minibatch_size if
                             self.args.gss_minibatch_size is not None
                             else self.args.minibatch_size, self.NAME, self)

        self.total_task_id = []

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
        self.opt.zero_grad()
        outputs = self.net(inputs)
        log_lanescore, heatmap, heatmap_reg = outputs

        
        


        outputs_prediction = [log_lanescore, heatmap, heatmap_reg]
        loss = self.loss(outputs_prediction, labels) #OverallLoss in UQnet


        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits, sampled_task_id = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=False, random=True)
            
            self.total_task_id.append(sampled_task_id)
            
            
            buf_outputs = self.net(buf_inputs) 
            buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg = buf_outputs
            
            loss += self.args.alpha*F.mse_loss(buf_heatmap_logits, buf_logits) #heatmaps are logits

            # For Plasticity, not sampled randomly. random = True for ablation stu
            buf_inputs_p, buf_labels_p, buf_logits_p, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=False, random=False)
            
            buf_outputs_p = self.net(buf_inputs_p)
            buf_log_lanescore_p, buf_heatmap_logits_p, buf_heatmap_reg_p = buf_outputs_p
            # To mimic the original buffer outputs, using buf_logits_p
            # loss += self.args.alpha*F.mse_loss(buf_heatmap_logits_p, buf_logits_p)

            # Directly replay the loss, using buf_labels_p
            loss += self.loss(buf_outputs_p, buf_labels_p)


        loss.backward()
        self.opt.step()
        
        #clean buf temp
        if not self.buffer.is_empty():
            del buf_outputs, buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg
            torch.cuda.empty_cache()
            
        if task_id is not None:
            self.buffer.add_data(examples=inputs, labels=labels, logits=heatmap.detach(), task_order=task_id)
        else:
            self.buffer.add_data(examples=inputs, labels=labels , logits=heatmap.detach())


        if task_id is not None:
            return loss.item(), self.total_task_id
        else:
            return loss.item()
