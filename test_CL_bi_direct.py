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
from experiments.testing_1_task_bi_direct import *
from utils.args_loading import scenario_info
import pickle

def main():
    parser = argparse.ArgumentParser(description='Testing process of CL', allow_abbrev=False)
    parser.add_argument('--num_tasks', type=int, default=12, help='The number of continuous tasks to be tested.')
    parser.add_argument('--continual_learning',  type=bool, default=True, help='Whether the CL strategies are used in training.')
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

    print("args.num_tasks:", args.num_tasks)
    # print("args.continual_learning:", args.continual_learning)
    print("args.model:", args.model)
    
    results_fde_dict = {}
    results_mr_dict = {}
    # args.num_tasks is the number of total tasks
    # learned_num indexes the trained model
    for task_id in range(0, args.num_tasks):
        fde_dict, mr_dict = test_1_task(learned_num = task_id+1, args = args)


        results_fde_dict[task_id+1] = fde_dict
        results_mr_dict[task_id+1] = mr_dict
   # calculate_forward_transfer(T=8, method='vanilla', b=0)

    with open(result_dir+'/Continual_'+str(args.num_tasks)+'_tasks_'+str(args.model)+'_bf_'+str(args.buffer_size)+'_fde.pkl', 'wb') as pickle_file1:
        pickle.dump(results_fde_dict, pickle_file1)
    with open(result_dir+'/Continual_'+str(args.num_tasks)+'_tasks_'+str(args.model)+'_bf_'+str(args.buffer_size)+'_mr.pkl', 'wb') as pickle_file2:
        pickle.dump(results_mr_dict, pickle_file2)
    

    # FDE_BWT, MR_BWT =  calculate_backward_transfer(result_dir, args.num_tasks, args)
    # FDE_FWT, MR_FWT = calculate_forward_transfer(result_dir, args.num_tasks, args)

    # summary_metric = open(result_dir + '/metrics_total_'+str(args.num_tasks)+'_tasks_'+str(args.model)+'_bf_'+str(args.buffer_size)+'.txt', 'w')
    # summary_metric.writelines('FDE-BWT:'+ str(FDE_BWT)+'m')
    # summary_metric.writelines('\n')
    # summary_metric.writelines('MR-BWT:'+str(MR_BWT)+'%')
    # summary_metric.writelines('\n')
    # summary_metric.writelines('FDE-FWT:'+ str(FDE_FWT)+'m')
    # summary_metric.writelines('\n')
    # summary_metric.writelines('MR-FWT:'+str(MR_FWT)+'%')
    # summary_metric.writelines('\n')
    # summary_metric.close()

if __name__ == '__main__':
    main()
    
