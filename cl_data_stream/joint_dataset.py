from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from traj_predictor.interaction_model import UQnet
from traj_predictor.losses import *
from utils.args_loading import *
from cl_data_stream.traj_joint_dataset import JointInteractionDataset

args = args_loading()
device = args.device
abs_dir = data_dir+'/'



def store_interaction_loaders(self, joint_number):
    scenario_info = {0:'MA', 1:'FT', 2:'LN', 3:'ZS2', 4:'OF', 5:'EP0', 6:'GL', 7:'ZS0', 8:'EP', 9:'MT', 10:'SR', 11:'EP1'}
    
    tasks_to_joint = []
    for i in range(0,joint_number):
        tasks_to_joint.append(scenario_info[i])
    print(f"Number of joint tasks: {joint_number}, Scenario Name: {tasks_to_joint}")
    trainset = JointInteractionDataset(tasks_to_joint,'train', paralist, paralist['mode'], filters=False)
    train_loader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
    return train_loader



class Joint_INTERACTION():
    NAME = 'joint-interaction'
    SETTING = 'domain-il'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, args) -> None:
        super(Joint_INTERACTION, self).__init__()
        self.args = args
    
    
    def get_data_loaders(self, joint_num) -> Tuple[DataLoader, DataLoader]: 
        train = store_interaction_loaders(self, joint_num)
        return train

    @staticmethod
    def get_backbone():
        return UQnet(paralist, test=True, drivable=False).to(device)


    @staticmethod
    def get_loss():
        return OverAllLoss(paralist).to(device)
    