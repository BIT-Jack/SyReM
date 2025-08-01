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
    

    
    t = args.train_task_id

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
            if args.replayed_rc:
                loss = model.observe(inputs, labels, t+1)
            # normal trainning without the logging of replayed data
            else:
                loss = model.observe(inputs, labels)
            loss_history.append(loss)
            sys.stdout.write(f"\rTraining Progress:"
                                f"  Epoch: {epoch+1}"
                                f"    [{current:>6d}/{task_sample_num:>6d}]"
                                f"    Loss: {loss:>.6f}"
                                f"   {(time.time()-start_time)/current:>.4f}s/sample")
            sys.stdout.flush()
        with open('./logging/'+'baseline_loss_single_task_'+str(t+1)+'.pkl', 'wb') as file:
            pickle.dump(loss_history, file)




    if epoch==(args.n_epochs-1):
        save_path_encoder = baseline_dir+'/'+str(args.n_epochs)+'_epoch_'+args.model+'_'+'tasks_'+str(t+1)+'_encoder'+'.pt'
        save_path_decoder = baseline_dir+'/'+str(args.n_epochs)+'_epoch_'+args.model+'_'+'tasks_'+str(t+1)+'_decoder'+'.pt'
        save_dir = os.path.dirname(save_path_encoder)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.net.encoder.state_dict(), save_path_encoder)
        torch.save(model.net.decoder.state_dict(), save_path_decoder)

