import torch
import sys
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from copy import deepcopy

def get_parser(parser):
    return parser



class FLB(nn.Module):
    NAME = 'flb'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args):
        super(FLB, self).__init__()
        self.net = backbone
        self.loss = loss
        self.args = args
        self.device = args.device
        self.transform = None
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)

        self.checkpoint = None
        self.fish = None
        self.checkpoint_pl = None
        self.fish_pl = None
        self.task = 0
        self.old_net = None
    
    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            # print(penalty)
            return penalty
    

    def end_task(self, dataset):

        fish = torch.zeros_like(self.net.get_params())

        self.net.eval()

        for j, data in enumerate(dataset.train_loader):
            
            if self.args.debug_mode and j >= 10:
                print("\n ---------end task ing------------")
                break


            traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls = data
            tensors_list = [traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls]
            tensors_list = [t.to(self.device) for t in tensors_list]

            inputs = (traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask)
            labels = (ls, y)

            self.opt.zero_grad()

            # forward
            output = self.net(inputs)

            # The loss used in motion forecastingas the surrogate of -log p(y|x) 
            loss = self.loss(output, labels)
            if not self.args.debug_mode:
                current = j + 1
                percent = 100.0 * current / len(dataset.train_loader)

                sys.stdout.write(
                    f"\rProgress in end_task: "
                    f"{current}/{len(dataset.train_loader)} "
                    f"({percent:6.2f}%) "
                    f"Loss: {loss:.6f}"
                )
                sys.stdout.flush()


            # no need the exp_cond_prob
            loss.backward()

            fish += self.net.get_grads_full() ** 2

        fish /= len(dataset.train_loader)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.args.gamma
            self.fish += fish

        self.checkpoint = self.net.get_params().data.clone()
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
        self.task += 1


    def observe(self, inputs, labels, task_id=None, record_list=None):
        self.opt.zero_grad()
        penalty = self.penalty()
        outputs = self.net(inputs)
        log_lanescore, heatmap, heatmap_reg = outputs

        outputs_prediction = [log_lanescore, heatmap, heatmap_reg]
        loss = self.loss(outputs_prediction, labels) + self.args.e_lambda * penalty

        loss.backward()
        self.opt.step()

        return loss.item()
    

    def get_fish_pl(self, dataset):

        fish = torch.zeros_like(self.net.get_params())
        # self.net.eval()

        for j, data in enumerate(dataset.train_loader):
            if self.args.debug_mode and j >= 10:
                print("\n ~~~~~~~debuging~~~~~~~")
                break
            
            traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls = data
            tensors_list = [traj, splines, masker, lanefeature,
                            adj, A_f, A_r, c_mask, y, ls]
            tensors_list = [t.to(self.device) for t in tensors_list]

            inputs = (traj, splines, masker, lanefeature,
                    adj, A_f, A_r, c_mask)
            labels = (ls, y)

            self.opt.zero_grad()

            outputs = self.net(inputs)

            # trajectory loss ≈ -log p(y | x)
            loss = self.loss(outputs, labels)
            if not self.args.debug_mode:
                current = j + 1
                percent = 100.0 * current / len(dataset.train_loader)

                sys.stdout.write(
                    f"\rProgress in get_fish_pl: "
                    f"{current}/{len(dataset.train_loader)} "
                    f"({percent:6.2f}%) "
                    f"Loss: {loss:.6f}"
                )
                sys.stdout.flush()


            
            loss.backward()

            # PL Fisher of flashback
            fish += self.net.get_grads_full() ** 2

        fish /= len(dataset.train_loader)

        self.fish_pl = fish
        self.checkpoint_pl = self.net.get_params().data.clone()
        # self.net.train()


    def penalty_pl(self):
        if self.checkpoint_pl is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.fish_pl * ((self.net.get_params() - self.checkpoint_pl) ** 2)).sum()
            return penalty
        
    def flashback(self, initial_teacher, primary_new_model, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        log_lanescore, heatmap, heatmap_reg = outputs
        outputs_prediction = [log_lanescore, heatmap, heatmap_reg]

        penalty_preserve = self.penalty()
        penalty_plasticity = self.penalty_pl()

        loss = self.loss(outputs_prediction, labels)
        loss += self.args.e_lambda * penalty_preserve
        loss += self.args.alpha_p * penalty_plasticity

        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()

        return loss.item()
