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
from utils.metrics import *


def test_1_task_jt2ho(task_index, args):

    cl_method_name = args.model
    scenario_info = args.scenario_info
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_log = open(result_dir+'/'+'joint_task_'+str(task_index)+'_held_out_test.txt','w')

    fde_list = []
    mr_list = []

    scenario_index = task_index - 1  ##select
    scenario_name = scenario_info[scenario_index]
    print("\n Scenario testing:", scenario_name)


    paralist['inference'] = True #True can provide the calculation of Ue
    model = UQnet(paralist, test=True, drivable=False).to(device)  # set test=True here

    testset = InteractionDataset(['val'], scenario_name,'val', paralist, mode=paralist['mode'], filters=False) # for validation
        

    model.encoder.load_state_dict(torch.load(saved_dir+'/'+'joint'+str(task_index)+'_tasks'+'_encoder'+'.pt', map_location='cuda:0'))
    model.decoder.load_state_dict(torch.load(saved_dir+'/'+'joint'+str(task_index)+'_tasks'+'_decoder'+'.pt', map_location='cuda:0'))
    model.eval()

    Yp, Ua, Um, Y = prediction_test(model,
                                    scenario_name,
                                    testset, 
                                    paralist, 
                                    test=False, 
                                    return_heatmap = False, 
                                    mode = 'lanescore',
                                    batch_size = args.batch_size, 
                                    cl_method_name = cl_method_name, trained_to=task_index, args=args)

    jt2ho_result = {}
    FDE, MR = ComputeError(Yp,Y, r=2, sh=6)
    jt2ho_result['fde'] = np.mean(FDE)
    jt2ho_result['mr'] = np.mean(MR)*100



    if args.store_traj:
        np.savez_compressed('./logging/results_record/fde_mr_'+cl_method_name+"_buffer"+str(args.buffer_size)+'_{:.0f}tasks_learned'.format(task_index)+'_test_on_'+scenario_name, all_case_fde = FDE, all_case_mr = MR)
    fde_list.append(np.mean(FDE))
    mr_list.append(np.mean(MR))
    result_log.writelines('-----task:{:.0f}'.format(task_index)+'-----')
    result_log.writelines('\n')
    result_log.writelines("minFDE: "+str(np.mean(FDE))+'m')
    result_log.writelines('\n')
    result_log.writelines("MR: "+str(np.mean(MR)*100)+'%\n\n')
    
    
    # result_log.writelines('\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # #Averaged prediction erros
    # result_log.writelines('\nThe averaged FDE of all tasks: '+ str(np.mean(fde_list))+' m')
    # result_log.writelines('\nThe averaged Missing Rate of all tasks: '+str(np.mean(mr_list)*100)+' %')
    # #Backward transfer
    # result_log.writelines('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')

    result_log.close()
    return jt2ho_result
