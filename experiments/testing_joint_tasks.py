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
from cl_data_stream.joint_dataset import *
from traj_predictor.losses import *
from traj_predictor.evaluation import *
from utils.args_loading import *
from absl import logging
logging._warn_preinit_stderr = 0
logging.warning('Worrying Stuff')
from utils.metrics import *


def test_joint(joint_tasks, args):

    cl_method_name = args.model
    scenario_info = args.scenario_info
    learned_t = args.learned_tasks
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_log = open(result_dir+'/'+ cl_method_name+'_bf_'+str(args.buffer_size) +'_learn_'+ str(learned_t)+'_jt_'+str(joint_tasks)+ '.txt','w')

    fde_list = []
    mr_list = []

    tasks_to_joint = []
    for i in range(0,joint_tasks):
        tasks_to_joint.append(scenario_info[i])
    print("\n Joint num:", joint_tasks, "Joint Scenarios:", tasks_to_joint)


    paralist['inference'] = True #True can provide the calculation of Ue
    model = UQnet(paralist, test=True, drivable=False).to(device)  # set test=True here

    testset = JointInteractionDataset(tasks_to_joint,'val', paralist, mode=paralist['mode'], filters=False) # for validation
        
    if not cl_method_name=='joint':
        model.encoder.load_state_dict(torch.load(saved_dir+'/'+args.model+'_'+'tasks_'+str(learned_t)+'_'+'bf_'+str(args.buffer_size)+'_encoder'+'.pt'))
        model.decoder.load_state_dict(torch.load(saved_dir+'/'+args.model+'_'+'tasks_'+str(learned_t)+'_'+'bf_'+str(args.buffer_size)+'_decoder'+'.pt'))
    else:
        model.encoder.load_state_dict(torch.load(saved_dir+'/'+args.model+str(learned_t)+'_tasks'+'_encoder'+'.pt', map_location='cuda:0'))
        model.decoder.load_state_dict(torch.load(saved_dir+'/'+args.model+str(learned_t)+'_tasks'+'_decoder'+'.pt', map_location='cuda:0'))
    model.eval()

    Yp, Ua, Um, Y = prediction_joint_test(model,
                                    tasks_to_joint,
                                    testset, 
                                    paralist, 
                                    test=False, 
                                    return_heatmap = False, 
                                    mode = 'lanescore',
                                    batch_size = args.batch_size, 
                                    cl_method_name = cl_method_name, trained_to=joint_tasks, args=args)

    joint_result = {}
    FDE, MR = ComputeError(Yp,Y, r=2, sh=6)
    joint_result['fde'] = np.mean(FDE)
    joint_result['mr'] = np.mean(MR)*100



    # if args.store_traj:
    #     np.savez_compressed('./logging/results_record/fde_mr_'+cl_method_name+"_buffer"+str(args.buffer_size)+'_{:.0f}tasks_learned'.format(joint_tasks)+'_test_on_'+scenario_name, all_case_fde = FDE, all_case_mr = MR)
    fde_list.append(np.mean(FDE))
    mr_list.append(np.mean(MR))
    result_log.writelines('-----task:{:.0f}'.format(joint_tasks)+'-----')
    result_log.writelines('\n')
    result_log.writelines("minFDE: "+str(np.mean(FDE))+'m')
    result_log.writelines('\n')
    result_log.writelines("MR: "+str(np.mean(MR)*100)+'%\n\n')

    with open(result_dir+'/'+ cl_method_name+'_bf_'+str(args.buffer_size) +'_learn_'+ str(learned_t)+'_jt_'+str(joint_tasks)+'.pkl','wb') as fb:
        pickle.dump(joint_result, fb)
    

    result_log.close()
    return 0
