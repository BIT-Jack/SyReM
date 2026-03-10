import numpy as np
import torch
from torch.optim import Adam
from cl_model.vgp import overwrite_grad, store_grad
from cl_model.continual_model import ContinualModel
from utils.memory_buffer import Buffer
from torch import nn
import math

def get_parser(parser):
    return parser



def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class Syfull(nn.Module):
    NAME = 'syfull'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args):
        super(Syfull, self).__init__()
        self.net = backbone
        self.loss = loss
        self.args = args
        self.device = args.device
        self.transform = None
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)

        self.buffer = Buffer(self.args.buffer_size, self.device, self.args.gss_minibatch_size if
                             self.args.gss_minibatch_size is not None
                             else self.args.minibatch_size, self.NAME, self, candidate_num=self.args.num_candidate)
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


    def observe(self, inputs, labels, id_bc=None, current_loader=None):

        self.zero_grad()

        # ========== 1. Current batch ==========
        p = self.net(inputs)
        cur_loss = self.loss(p, labels)
        cur_loss.backward()   #

        # ========== 2. Entire buffer replay (upper bound) ==========
        if not self.buffer.is_empty():
            num_valid = self.buffer.current_size
            num_batches = math.ceil(num_valid / self.args.batch_size)

            for ii in range(num_batches):
                start = ii * self.args.batch_size
                end = min(start + self.args.batch_size, num_valid)
                indices = list(range(start, end))

                buf_inputs, buf_labels = self.buffer.get_data_by_index(
                    indices, transform=self.transform
                )

                buf_outputs = self.net(buf_inputs)
                replay_loss = self.loss(buf_outputs, buf_labels)

                replay_loss.backward()   # batch backward one by one

            # ========== 3. Store plasticity gradient ==========
            store_grad(self.parameters, self.grad_xy, self.grad_dims)

            # ========== 4. ER gradient for constraint ==========
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                return_index=False,
                random=True
            )

            self.net.zero_grad()
            buf_outputs = self.net(buf_inputs)
            penalty = self.loss(buf_outputs, buf_labels)
            penalty.backward()
            store_grad(self.parameters, self.grad_er, self.grad_dims)

            # ========== 5. Gradient projection ==========
            dot_prod = torch.dot(self.grad_xy, self.grad_er)

            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self.parameters, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self.parameters, self.grad_xy, self.grad_dims)

        # ========== 6. Optimizer step ==========
        self.opt.step()

        # ========== 7. Buffer update ==========
        if id_bc is None:
            self.buffer.add_data(examples=inputs, labels=labels)
        else:
            self.buffer.add_data(examples=inputs, labels=labels, samples_id=id_bc)

        return cur_loss.item()
