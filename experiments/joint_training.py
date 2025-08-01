import math
import sys
from argparse import Namespace
import torch
from cl_model.continual_model import ContinualModel
import time
from traj_predictor.losses import *
from traj_predictor.utils import *
from utils.args_loading import *




def train(model: ContinualModel,
           dataset,
          args: Namespace) -> None:

    model.net.to(model.device) 
    print("The model for training:", args.model)
    T = args.train_task_num 
    model.net.train(True)
    train_loader = dataset.get_data_loaders(T)
    task_sample_num = len(train_loader)*args.batch_size

    if hasattr(model, 'begin_task'):
        model.begin_task(dataset)

    for epoch in range(model.args.n_epochs):
        start_time = time.time()
        current = 0
        for i, data in enumerate(train_loader):
            current = current + args.batch_size
            if args.debug_mode and i >= 100:
                print("\n >>>>>>>>>>>>debuging>>>>>>>>>>>")
                break
            traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls = data
            tensors_list = [traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls]
            tensors_list = [t.to(model.device) for t in tensors_list]

            inputs = (traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask)
            labels = [ls, y]

            loss = model.observe(inputs, labels) 
            sys.stdout.write(f"\rTraining Progress:"
                    f"  Epoch: {epoch+1}"
                    f"    [{current:>6d}/{task_sample_num:>6d}]"
                    f"    Loss: {loss:>.6f}"
                    f"   {(time.time()-start_time)/current:>.4f}s/sample")
            sys.stdout.flush()
        
    
        
        if epoch==(args.n_epochs-1):
            save_path_encoder = saved_dir+'/'+'joint'+str(T)+'_'+'tasks'+'_encoder'+'.pt'
            save_path_decoder = saved_dir+'/'+'joint'+str(T)+'_'+'tasks'+'_decoder'+'.pt'
            save_dir = os.path.dirname(save_path_encoder)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.net.encoder.state_dict(), save_path_encoder)
            torch.save(model.net.decoder.state_dict(), save_path_decoder)
