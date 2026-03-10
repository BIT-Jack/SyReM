from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import nn
from traj_predictor.interaction_model import UQnet
from traj_predictor.losses import *
from utils.args_loading import *
from cl_data_stream.traj_dataset import InteractionDataset

args = args_loading()
device = args.device
abs_dir = data_dir+'/'



def store_interaction_loaders(self, scenario_id, task_order_id=0):
    scenario_info_set = [{0:'MA', 1:'FT', 2:'LN', 3:'ZS2', 4:'OF', 5:'EP0', 6:'GL', 7:'ZS0', 8:'EP', 9:'MT', 10:'SR', 11:'EP1'}, 
    {
    0: 'EP',
    1: 'MA',
    2: 'ZS0',
    3: 'FT',
    4: 'GL',
    5: 'EP1',
    6: 'LN',
    7: 'SR',
    8: 'OF',
    9: 'ZS2',
    10: 'MT',
    11: 'EP0'
    },
    {
    0: 'GL',
    1: 'ZS2',
    2: 'MA',
    3: 'EP0',
    4: 'SR',
    5: 'FT',
    6: 'EP',
    7: 'LN',
    8: 'MT',
    9: 'ZS0',
    10: 'OF',
    11: 'EP1'
    }
    ]

    scenario_info = scenario_info_set[task_order_id]

    scenario_index = scenario_id ## choose scenario
    scenario_name = scenario_info[scenario_index]

    print(f"Scenario Index: {scenario_index+1}, Scenario Name: {scenario_name}")

    trainset = InteractionDataset(['train'], scenario_name,'train', paralist, paralist['mode'], filters=False)
    train_loader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
    self.train_loader = train_loader
    return train_loader


class SequentialINTERACTION():
    NAME = 'seq-interaction'
    SETTING = 'domain-il'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __init__(self, args) -> None:
        super(SequentialINTERACTION, self).__init__()
        self.args = args
    #data loader
    def get_data_loaders(self, task_id, order_id) -> Tuple[DataLoader, DataLoader]: 
        train = store_interaction_loaders(self, task_id, order_id)
        return train
