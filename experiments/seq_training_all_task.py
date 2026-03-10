import sys
import torch
from cl_model.continual_model import ContinualModel
import time
#import UQnet loss
from traj_predictor.losses import *
from traj_predictor.utils import *
from utils.args_loading import *
import os
import pickle
from copy import deepcopy

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
        print('task order id:', args.task_order_id)
        train_loader = dataset.get_data_loaders(t, args.task_order_id)
        task_sample_num = len(train_loader)*args.batch_size


        loss_history = []

        #//////////////Revision: Computational Cost Record
        device = model.device  # e.g., torch.device("cuda:1")

        torch.cuda.reset_peak_memory_stats(device)

        batch_time_list = []
        if "flashback" in args.model:
            fb_batch_time_list = []
        #//////////////

        if "flashback" in args.model:
            initial_teacher = deepcopy(model.old_net)
        for epoch in range(args.n_epochs):
            start_time = time.time()
            current = 0
            for i, data in enumerate(train_loader):
                current =current+args.batch_size
                if args.debug_mode and i >= 10:
                    print("\n >>>>>>>>>>>>debuging main train>>>>>>>>>>>")
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
                    #//////////////Revision: Computational Cost Record
                    torch.cuda.synchronize(device)
                    start_time = time.time()
                    #//////////////
                    
                    loss = model.observe(inputs, labels)

                    

                    #//////////////Revision: Computational Cost Record
                    torch.cuda.synchronize(device)
                    batch_time = time.time() - start_time
                    batch_time_list.append(batch_time)
                    #//////////////

                if not args.debug_mode:
                    current = i + 1
                    percent = 100.0 * current / len(dataset.train_loader)

                    sys.stdout.write(
                        f"\rTraining Progress 1: "
                        f"{current}/{len(dataset.train_loader)} "
                        f"({percent:6.2f}%) "
                        f"Loss: {loss:.6f}"
                    )
                    sys.stdout.flush()

                loss_history.append(loss)

        if "flashback" in args.model:
            model.get_fish_pl(dataset)

            primary_new_model = deepcopy(model.net)

            for param in primary_new_model.parameters():
                param.requires_grad = False

            if initial_teacher is not None:
                for param in  initial_teacher.parameters():
                    param.requires_grad = False
            if model.old_net is not None:
                model.net.load_state_dict(deepcopy(model.old_net.state_dict()))

        
            for i, data in enumerate(train_loader):
                if args.debug_mode and i >= 10:
                    print("\n ---------flashback debuging------------")
                    break

                # unpack trajectory prediction data
                traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls = data

                tensors_list = [
                    traj, splines, masker, lanefeature,
                    adj, A_f, A_r, c_mask, y, ls
                ]
                tensors_list = [t.to(model.device) for t in tensors_list]

                # reconstruct after .to()
                traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls = tensors_list

                # structured inputs / labels
                inputs = (traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask)
                labels = (ls, y)

                # Flashback: no augmentation in trajectory prediction
                not_aug_inputs = None
                
                torch.cuda.synchronize(device)
                start_time_of_fb = time.time()
                
                loss = model.flashback(
                    initial_teacher,
                    primary_new_model,
                    inputs,
                    labels,
                    not_aug_inputs
                )
                if not args.debug_mode:
                    current = i + 1
                    percent = 100.0 * current / len(dataset.train_loader)

                    sys.stdout.write(
                        f"\rFlashback Progress: "
                        f"{current}/{len(dataset.train_loader)} "
                        f"({percent:6.2f}%) "
                        f"Loss: {loss:.6f}"
                    )
                    sys.stdout.flush()

                
                torch.cuda.synchronize(device)
                fb_batch_time = time.time() - start_time_of_fb
                fb_batch_time_list.append(fb_batch_time)
                # assert not math.isnan(loss)
                # progress_bar.prog(i, len(train_loader), epoch, t, loss)


                

        
        #//////////////Revision: Computational Cost Record
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        avg_batch_time = sum(batch_time_list) / len(batch_time_list) * 1000
        if "flashback" in args.model:
            avg_fb_batch_time = sum(fb_batch_time_list) / len(fb_batch_time_list) * 1000

        print("Peak Memory GPU:", peak_mem_mb, 'MB')
        print("Training time/batch:", avg_batch_time, 'ms')

        # ========= Save results =========
        save_dir = "./logging/efficiency"
        os.makedirs(save_dir, exist_ok=True)

        task_id = t + 1
        base_name = f"{args.model}_bf_{args.buffer_size}_task_{task_id}"

        # ---- txt (readable) ----
        txt_path = os.path.join(save_dir, base_name + ".txt")
        with open(txt_path, "w") as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Buffer size: {args.buffer_size}\n")
            f.write(f"Task ID: {task_id}\n")
            f.write(f"Peak GPU memory (MB): {peak_mem_mb:.2f}\n")
            f.write(f"Training time per batch (ms): {avg_batch_time:.2f}\n")
            if "flashback" in args.model:
                f.write(f"Flashback time per batch (ms): {avg_fb_batch_time:.2f}\n")

        # ---- pkl (for post-processing) ----
        pkl_path = os.path.join(save_dir, base_name + ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "model": args.model,
                    "buffer_size": args.buffer_size,
                    "task_id": task_id,
                    "peak_mem_mb": peak_mem_mb,
                    "avg_batch_time": avg_batch_time,
                },
                f
            )
        #//////////////

        # with open(f'./logging/{args.model}_bf_{args.buffer_size}_loss_in_task'+'_'+str(t+1)+'.pkl', 'wb') as file:
        #     pickle.dump(loss_history, file)


        save_dir_tmp = "./logging"
        os.makedirs(save_dir_tmp, exist_ok=True)

        file_path = os.path.join(
            save_dir_tmp,
            f"{args.model}_bf_{args.buffer_size}_loss_in_task_{t+1}.pkl"
        )

        with open(file_path, "wb") as file:
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



        #A-GEM, GEM, flashback methods
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
