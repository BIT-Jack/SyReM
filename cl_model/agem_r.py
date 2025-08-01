import numpy as np
import torch
from torch.optim import Adam
from cl_model.gem import overwrite_grad, store_grad
from cl_model.continual_model import ContinualModel
from utils.derpp_buffer_new import Buffer
from torch import nn

def get_parser(parser):
    return parser



def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class AGemR(nn.Module):
    NAME = 'agem_r'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args):
        super(AGemR, self).__init__()
        self.net = backbone
        self.loss = loss
        self.args = args
        self.device = args.device
        self.transform = None
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)

        self.buffer = Buffer(self.args.buffer_size, self.device, self.args.gss_minibatch_size if
                             self.args.gss_minibatch_size is not None
                             else self.args.minibatch_size, self.NAME, self)
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)


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

    # def end_task(self, dataset):
    #     # samples_per_task = self.args.buffer_size // self.args.train_task_num
    #     loader = dataset.train_loader
        
    #     for add_batch_num in range(self.args.buffer_size // self.args.batch_size):
    #         traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls = next(iter(loader))
    #         tensors_list_tmp = [traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls]
    #         tensors_list_tmp = [t.to(self.device) for t in tensors_list_tmp]
    #         cur_x =  (traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask)
    #         cur_y = [ls, y]
    #         self.buffer.add_data(examples= cur_x,labels=cur_y)


    def observe(self, inputs, labels, task_id=None, record_list=None):

        if self.buffer.is_empty():
            self.zero_grad()
            p = self.net.forward(inputs)
            loss = self.loss(p, labels)
            loss.backward()

        else:
            self.zero_grad()
            p = self.net.forward(inputs)
            loss = self.loss(p, labels)

            # plasticity enhancement is not used in agem_r, as a baseline for comparison
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, return_index=False, random=False)
            buf_outputs = self.net.forward(buf_inputs)
            loss += self.loss(buf_outputs, buf_labels)

            loss.backward()
            
            # A-GEM algorithm
            store_grad(self.parameters, self.grad_xy, self.grad_dims)

            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, return_index=False, random=True)
            self.net.zero_grad()
            buf_outputs = self.net.forward(buf_inputs)
            penalty = self.loss(buf_outputs, buf_labels)
            penalty.backward()
            store_grad(self.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self.parameters, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self.parameters, self.grad_xy, self.grad_dims)

        self.opt.step()

        self.buffer.add_data(examples=inputs, labels=labels)

        return loss.item()
