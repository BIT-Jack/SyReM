import torch
from torch import nn
from utils.der_gssp_buffer import Buffer as Buffer
from torch.optim import Adam

def get_parser(parser):
    return parser


class Gss(nn.Module):
    NAME = 'gss'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args):
        super(Gss, self).__init__()
        self.net = backbone
        self.loss = loss
        self.args = args
        self.device = args.device
        self.transform = None
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)
        self.buffer = Buffer(self.args.buffer_size, self.device,
                             self.args.gss_minibatch_size if
                             self.args.gss_minibatch_size is not None
                             else self.args.minibatch_size, self.NAME, self)
        self.alj_nepochs = 1

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

        real_batch_size = inputs[0].shape[0]
        self.buffer.drop_cache()
        self.buffer.reset_fathom()

        for _ in range(self.alj_nepochs):
            self.opt.zero_grad()
            if not self.buffer.is_empty():
                buf_inputs, buf_labels, sampled_task_id = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform)
                self.total_task_id.append(sampled_task_id)

                tinputs = ()
                for ii in range(len(buf_inputs)):
                    tinputs += (torch.cat((inputs[ii], buf_inputs[ii])),)
                tlabels = ()
                for jj in range(len(buf_labels)):
                    tlabels += (torch.cat((labels[jj], buf_labels[jj])),)
            else:
                tinputs = inputs
                tlabels = labels

            outputs = self.net(tinputs)
            loss = self.loss(outputs, tlabels)
            loss.backward()
            self.opt.step()
            
        if task_id is not None:
            self.buffer.add_data(examples=inputs,
                             labels=labels, task_order=task_id)
            return loss.item(), self.total_task_id
        else:
            self.buffer.add_data(examples=inputs,
                             labels=labels)
            return loss.item()
