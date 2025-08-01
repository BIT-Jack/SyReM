import torch
from torch.nn import functional as F
from torch import nn
# from utils.der_buffer import Buffer
from utils.derpp_buffer_new import Buffer
from torch.optim import Adam
import matplotlib.pyplot as plt

def get_parser(parser):
    return parser



class DerPe(nn.Module):
    NAME = 'der_pe'
    COMPATIBILITY = ['domain-il']
    def __init__(self, backbone, loss, args):
        super(DerPe, self).__init__()
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

    def observe(self, inputs, labels, id_bc=None, current_loader = None):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        log_lanescore, heatmap, heatmap_reg = outputs
    
        
        


        outputs_prediction = [log_lanescore, heatmap, heatmap_reg]
        loss = self.loss(outputs_prediction, labels) #OverallLoss in UQnet
        replayed_info = None
        replayed_score = None

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits, _, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=False, random=True)

            
            
            buf_outputs = self.net(buf_inputs) 
            buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg = buf_outputs
            
            loss += self.args.alpha*F.mse_loss(buf_heatmap_logits, buf_logits) #heatmaps are logits

            # plasticity enhancement, random = True when running the ablation experiment
            buf_inputs_p, buf_labels_p, _, replayed_info, replayed_score = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=False, random=False, record=id_bc, flag_of_new_task=current_loader)
            
            buf_outputs_p = self.net(buf_inputs_p)
            loss += self.loss(buf_outputs_p, buf_labels_p)


        loss.backward()
        self.opt.step()
        
        #clean buf temp
        if not self.buffer.is_empty():
            del buf_outputs, buf_log_lanescore, buf_heatmap_logits, buf_heatmap_reg
            torch.cuda.empty_cache()
            

        if id_bc is None:
            self.buffer.add_data(examples=inputs, labels=labels , logits=heatmap.detach())
        else:
            self.buffer.add_data(examples=inputs, labels=labels , logits=heatmap.detach(), samples_id=id_bc)


        if id_bc is None:
            return loss.item()
        else:
            return loss.item(), replayed_info, replayed_score
