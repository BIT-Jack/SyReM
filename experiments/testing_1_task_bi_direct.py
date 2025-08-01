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


def test_1_task(learned_num, args):
    # learned_num = from 1
    cl_method_name = args.model
    scenario_info = args.scenario_info
    
    result_log = open(result_dir+'/After_'+str(learned_num)+'_CL_tasks_'+ cl_method_name+'_bf_'+str(args.buffer_size) + '.txt','w')
    fde_list = []
    mr_list = []
    fde_dict = {}
    mr_dict = {}


    # tested in both learned and unseen tasks
    for test_task_id in range(0, args.num_tasks):

        scenario_index = test_task_id ##select
        scenario_name = scenario_info[scenario_index]
        print("\n Current task number:", str(learned_num), " Now testing task ", str(test_task_id+1), "...")


        paralist['inference'] = True #True can provide the calculation of Ue
        model = UQnet(paralist, test=True, drivable=False).to(device)  # set test=True here

        testset = InteractionDataset(['val'], scenario_name,'val', paralist, mode=paralist['mode'], filters=False) # for validation
         
        if not cl_method_name=='joint':
            model.encoder.load_state_dict(torch.load(saved_dir+'/'+args.model+'_'+'tasks_'+str(learned_num)+'_'+'bf_'+str(args.buffer_size)+'_encoder'+'.pt'))
            model.decoder.load_state_dict(torch.load(saved_dir+'/'+args.model+'_'+'tasks_'+str(learned_num)+'_'+'bf_'+str(args.buffer_size)+'_decoder'+'.pt'))
        else:
            model.encoder.load_state_dict(torch.load(saved_dir+'/'+'joint'+'_'+'tasks_'+str(learned_num)+'_'+'bf_'+str(args.buffer_size)+'_encoder'+'.pt'))
            model.decoder.load_state_dict(torch.load(saved_dir+'/'+'joint'+'_'+'tasks_'+str(learned_num)+'_'+'bf_'+str(args.buffer_size)+'_decoder'+'.pt'))
        model.eval()

        Yp, Ua, Um, Y = prediction_test(model,
                                        scenario_name,
                                        testset, 
                                        paralist, 
                                        test=False, 
                                        return_heatmap = False, 
                                        mode = 'lanescore',
                                        batch_size = args.batch_size, 
                                        cl_method_name = cl_method_name, trained_to=learned_num, args=args)

       
        FDE, MR = ComputeError(Yp,Y, r=2, sh=6)
        

        #saving
        fde_dict[test_task_id+1] = np.mean(FDE) # m
        mr_dict[test_task_id+1] = np.mean(MR)*100 # %


        #logging
        if args.store_traj:
            np.savez_compressed('./logging/prediction_record/fde_mr_'+cl_method_name+"_buffer"+str(args.buffer_size)+'_{:.0f}tasks_learned'.format(learned_num)+'_test_on_'+scenario_name, all_case_fde = FDE, all_case_mr = MR)
        fde_list.append(np.mean(FDE))
        mr_list.append(np.mean(MR))
        result_log.writelines('-----Trained after task' + str(learned_num) + '--> Tested in task:{:.0f}'.format(test_task_id+1)+'-----')
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
    return fde_dict, mr_dict