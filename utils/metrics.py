import numpy as np
import os
from utils.args_loading import *
import pickle


def calculate_backward_transfer(result_direction, T, model_name, buffer_size):
    with open(result_direction+'/Continual_'+'12_tasks_'+model_name+'_bf_'+str(buffer_size)+'_fde.pkl', 'rb') as fde_file:
        result_all_fde = pickle.load(fde_file)
    with open(result_direction+'/Continual_'+'12_tasks_'+model_name+'_bf_'+str(buffer_size)+'_mr.pkl', 'rb') as mr_file:
        result_all_mr = pickle.load(mr_file)

    sum_fde = 0
    sum_mr = 0
    for i in range(1, T):
        tmp_delta_fde = result_all_fde[T][i] - result_all_fde[i][i]
        sum_fde += tmp_delta_fde

        tmp_delta_mr = result_all_mr[T][i] - result_all_mr[i][i]
        sum_mr += tmp_delta_mr
    fde_bwt = sum_fde/(T-1)
    mr_bwt = sum_mr/(T-1)

    return fde_bwt, mr_bwt




def calculate_forward_transfer_ref_random(result_direction, current_id, T, model_name, buffer_size):
    with open(result_direction+'/Continual_'+'12_tasks_'+model_name+'_bf_'+str(buffer_size)+'_fde.pkl', 'rb') as fde_file:
        result_all_fde = pickle.load(fde_file)
    with open(result_direction+'/Continual_'+'12_tasks_'+model_name+'_bf_'+str(buffer_size)+'_mr.pkl', 'rb') as mr_file:
        result_all_mr = pickle.load(mr_file)

    c = current_id

    sum_fde = 0
    sum_mr = 0
    for i in range(c+1, T+1):
        with open(result_direction+'/baseline_task_'+str(i)+'.pkl', 'rb') as baseline_file:
            baseline = pickle.load(baseline_file)
        
        baseline_fde = baseline['fde']
        baseline_mr = baseline['mr']

        tmp_delta_fde = result_all_fde[c][i] - baseline_fde
        sum_fde += tmp_delta_fde

        tmp_delta_mr = result_all_mr[c][i] - baseline_mr
        sum_mr += tmp_delta_mr

    fde_fwt = sum_fde/(T-c)
    mr_fwt = sum_mr/(T-c)

    return fde_fwt, mr_fwt

def calculate_forward_transfer_ref_joint(result_direction, current_id, T, model_name, buffer_size):
    with open(result_direction+'/Continual_'+'12_tasks_'+model_name+'_bf_'+str(buffer_size)+'_fde.pkl', 'rb') as fde_file:
        result_all_fde = pickle.load(fde_file)
    with open(result_direction+'/Continual_'+'12_tasks_'+model_name+'_bf_'+str(buffer_size)+'_mr.pkl', 'rb') as mr_file:
        result_all_mr = pickle.load(mr_file)

    c = current_id

    sum_fde = 0
    sum_mr = 0
    for i in range(c+1, T+1):
        with open(result_direction+'/joint_task_'+str(i)+'_held_out_test.pkl', 'rb') as baseline_file:
            baseline = pickle.load(baseline_file)
        
        baseline_fde = baseline['fde']
        baseline_mr = baseline['mr']

        tmp_delta_fde = result_all_fde[c][i] - baseline_fde
        sum_fde += tmp_delta_fde

        tmp_delta_mr = result_all_mr[c][i] - baseline_mr
        sum_mr += tmp_delta_mr

    fde_fwt = sum_fde/(T-c)
    mr_fwt = sum_mr/(T-c)

    return fde_fwt, mr_fwt



    
def calculate_intransigence_measure_for_task_k(result_direction, k,  model_name, buffer_size):
    with open(result_direction+'/Continual_'+'12_tasks_'+model_name+'_bf_'+str(buffer_size)+'_fde.pkl', 'rb') as fde_file:
        result_all_fde = pickle.load(fde_file)
    with open(result_direction+'/Continual_'+'12_tasks_'+model_name+'_bf_'+str(buffer_size)+'_mr.pkl', 'rb') as mr_file:
        result_all_mr = pickle.load(mr_file)
    with open(result_direction+'/joint_task_'+str(k)+'_held_out_test.pkl', 'rb') as reference_file:
        ref_result = pickle.load(reference_file)
    
    im_k_fde = result_all_fde[k][k] - ref_result['fde']
    im_k_mr = result_all_mr[k][k] - ref_result['mr']


    return im_k_fde, im_k_mr


    