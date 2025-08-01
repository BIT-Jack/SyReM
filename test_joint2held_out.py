from traj_predictor.interaction_model import UQnet
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
import torchvision.datasets as dataset
from torch.optim.lr_scheduler import StepLR
import datetime
from scipy.special import expit
from traj_predictor.utils import *
from cl_data_stream.seq_dataset import *
from traj_predictor.losses import *
from traj_predictor.evaluation import *
from utils.args_loading import *
from absl import logging
logging._warn_preinit_stderr = 0
logging.warning('Worrying Stuff')
import argparse
from utils.metrics import *
from experiments.testing_1_task_joint2held_out import *
from utils.args_loading import scenario_info
    
def main():
    parser = argparse.ArgumentParser(description='Testing process of CL', allow_abbrev=False)
    parser.add_argument('--test_task_id', type=int, default=1, help='The index of the task to be tested.')
    parser.add_argument('--continual_learning',  type=bool, default=False, help='Whether the CL strategies are used in training.')
    parser.add_argument('--model', type=str, default='der')
    parser.add_argument('--buffer_size', type=str, default='2000')
    parser.add_argument('--batch_size', type=int, default= 8)
    #parameter to store observed trajectories and predictions for visualization (can be used when running test_CL.py)
    parser.add_argument('--store_traj', type=bool, default=False,
                        help='turn True to store observed trajectories and predicted results')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default= device)
    args = parser.parse_args()
    args.scenario_info = scenario_info

    print("args.continual_learning:", args.continual_learning)
    print("args.model:", args.model)
    


    task_to_test = args.test_task_id
    jt2ho_result = test_1_task_jt2ho(task_index=task_to_test, args = args)
    with open(result_dir+'/'+'joint_task_'+str(args.test_task_id)+'_held_out_test.pkl', 'wb') as pickle_file:
        pickle.dump(jt2ho_result, pickle_file)

if __name__ == '__main__':
    main()
    