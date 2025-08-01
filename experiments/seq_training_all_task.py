import sys
import torch
from cl_model.continual_model import ContinualModel
import time
#import UQnet loss
from traj_predictor.losses import *
from traj_predictor.utils import *
from utils.args_loading import *

import pickle


def train(model: ContinualModel,
          dataset,
          args):
    
    model.net.to(model.device)
    if args.replayed_rc:
        global replayed_data_recording
        replayed_data_recording = [1]*args.buffer_size

    print("The model for training:", args.model)
    
    if args.restart_training:
        task_num_pre = args.restart_pre_task_num
        print("Restart from Scenario ", task_num_pre)
        model.net.encoder.load_state_dict(torch.load(saved_dir+'/'+args.model+'_'+'tasks_'+str(task_num_pre)+'_'+'bf_'+str(args.buffer_size)+'_encoder'+'.pt',
                                              map_location='cuda:0'))
        model.net.decoder.load_state_dict(torch.load(saved_dir+'/'+args.model+'_'+'tasks_'+str(task_num_pre)+'_'+'bf_'+str(args.buffer_size)+'_decoder'+'.pt',
                                              map_location='cuda:0'))
        print("The trained weights loaded.")
    
    if args.restart_training:
        start_id = task_num_pre
    else:
        start_id = 0
    
    
    for t in range(start_id, args.train_task_num):
        model.net.train(True)
        train_loader = dataset.get_data_loaders(t)
        task_sample_num = len(train_loader)*args.batch_size


        loss_history = []
        for epoch in range(args.n_epochs):
            start_time = time.time()
            current = 0
            for i, data in enumerate(train_loader):
                current =current+args.batch_size
                if args.debug_mode and i >= 10:
                    print("\n >>>>>>>>>>>>debuging>>>>>>>>>>>")
                    break
                
                traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls = data
                tensors_list = [traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls]
                tensors_list = [t.to(model.device) for t in tensors_list]

                inputs = (traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask)
                labels = [ls, y]

                # to record replayed data for each task, for further analysis
                if args.replayed_rc and args.model=='b2p':
                    loss = model.observe(inputs, labels, t+1)
                elif args.replayed_rc:
                    loss, list_task_id = model.observe(inputs, labels, t+1)
                    # print(list_task_id)
                # normal trainning without the logging of replayed data
                else:
                    loss = model.observe(inputs, labels)

                sys.stdout.write(f"\rTraining Progress:"
                                 f"  Epoch: {epoch+1}"
                                 f"    [{current:>6d}/{task_sample_num:>6d}]"
                                 f"    Loss: {loss:>.6f}"
                                 f"   {(time.time()-start_time)/current:>.4f}s/sample")
                sys.stdout.flush()

                loss_history.append(loss)

        with open(f'./logging/{args.model}_bf_{args.buffer_size}_loss_in_task'+'_'+str(t+1)+'.pkl', 'wb') as file:
            pickle.dump(loss_history, file)
        



        if not hasattr(model, 'end_task'):
            if epoch==(args.n_epochs-1):
                save_path_encoder = saved_dir+'/'+args.model+'_'+'tasks_'+str(t+1)+'_'+'bf_'+str(args.buffer_size)+'_encoder'+'.pt'
                save_path_decoder = saved_dir+'/'+args.model+'_'+'tasks_'+str(t+1)+'_'+'bf_'+str(args.buffer_size)+'_decoder'+'.pt'
                save_dir = os.path.dirname(save_path_encoder)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(model.net.encoder.state_dict(), save_path_encoder)
                torch.save(model.net.decoder.state_dict(), save_path_decoder)



        #A-GEM, GEM methods
        elif hasattr(model, 'end_task'):
            model.end_task(dataset)          
            if epoch==(args.n_epochs-1):
                save_path_encoder = saved_dir+'/'+args.model+'_'+'tasks_'+str(t+1)+'_'+'bf_'+str(args.buffer_size)+'_encoder'+'.pt'
                save_path_decoder = saved_dir+'/'+args.model+'_'+'tasks_'+str(t+1)+'_'+'bf_'+str(args.buffer_size)+'_decoder'+'.pt'
                save_dir = os.path.dirname(save_path_encoder) 
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(model.net.encoder.state_dict(), save_path_encoder)
                torch.save(model.net.decoder.state_dict(), save_path_decoder)

    #recording the sampled memory
    if args.replayed_rc and args.model !='b2p':
        with open('./logging/'+str(args.model)+'_bf_'+str(args.buffer_size)+'_sampled_memory_task_labels.pkl', 'wb') as rf:
            pickle.dump(list_task_id, rf)
