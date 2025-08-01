import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from torchvision import transforms
from torch import nn
import argparse
from scipy.sparse import csr_matrix
from skimage.transform import rescale
from skimage.measure import block_reduce
from numpy.lib.stride_tricks import as_strided
from skimage.feature import peak_local_max
from scipy import ndimage, misc
from scipy.signal import convolve2d
from skimage.filters import gaussian
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

def prediction_test(model, scenarioname,  dataset, para, 
                    test, return_heatmap, 
                    mode, batch_size, cl_method_name, trained_to, args):
    
    print(str(cl_method_name),": into CL TESTING now...")
    H = []
    Ua = []
    Ue = []
    Yp = []
    L = []
    data = np.load(data_dir+'/'+'val'+'/'+ 'val' +'_'+scenarioname+'.npz', allow_pickle=True)
    if return_heatmap:
        Na = data['nbagents']
        Nm = data['nbsplines']
        T = data['trajectory']
        M = data['maps']
        print("------------here--------------")
    if not test:
        Y = data['intention']
        print("==================Y==================", Y.shape)
    if scenarioname == 'EP' or scenarioname == 'EP1':
        Y=Y[:-1]
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("len of ds:", len(loader))
    Hp = []
    Lp = []
    Hi = []
    Li = []
    #Tr is to store the traj for plotting
    Tr = []
    for k, data in enumerate(loader):
        if mode=='lanescore':
            if not test:
                traj, splines, masker, lanefeature, adj, af, ar, c_mask, y, ls = data
            else:
                traj, splines, masker, lanefeature, adj, af, ar, c_mask = data
            inputs = (traj, splines, masker, lanefeature, adj, af, ar, c_mask)
            lsp, heatmap = model(inputs)
            Hi.append(heatmap.detach().to('cpu').numpy())
            Li.append(lsp.detach().to('cpu').numpy())
            # if args.store_traj:
            # # store each batch of traj (batchsize, 26vehicles, 9timestamps, 8features)
            #     Tr.append(traj.detach().to('cpu').numpy())
    Hi = np.concatenate(Hi,0)
    Li = np.concatenate(Li,0)
    # if args.store_traj:
    # # concatenate all batches of traj
    #     Tr = np.concatenate(Tr, 0) # (number of cases, 26, 9, 8)
        
    
    Hp.append(Hi)
    Lp.append(Li)
    Hp = np.stack(Hp, 0)
    Lp = np.stack(Lp, 0)
    hm, ua, ue = ComputeUQ(Hp, para['resolution'], epsilon=5e-4)
    print("hm:",hm.shape)
    yp = ModalSampling(hm, para, r=3, k=6) #multi-madal prediction, k=num of modal
    Ua.append(ua)
    Ue.append(ue)
    Yp.append(yp)
    if return_heatmap:
        H.append(hm)
        L.append(Lp.squeeze())

    Ua = np.concatenate(Ua, 0)
    Ue = np.concatenate(Ue, 0)
    Yp = np.concatenate(Yp, 0)
    if args.store_traj:
        H.append(hm)
        H = np.concatenate(H, 0) 
    ##
    if return_heatmap:
        H = np.concatenate(H, 0) 

    if test:
        if return_heatmap:
            return M, T, Nm, Na, Yp, Ua, Ue, H#, L
        else:
            return Yp, Ua, Ue
    else:
        if return_heatmap:
            return M, T, Nm, Na, Yp, Ua, Ue, H, Y
        else:
            if args.store_traj:
                np.savez_compressed('./logging/prediction_record/'+cl_method_name+"_buffer"+str(args.buffer_size)+'_{:.0f}tasks_learned'.format(trained_to)+'_test_on_'+scenarioname,  pred = Yp, gt = Y, heatmap = H)

            return Yp, Ua, Ue, Y
        


def prediction_joint_test(model, scenarioname,  dataset, para, 
                    test, return_heatmap, 
                    mode, batch_size, cl_method_name, trained_to, args):
    
    print(str(cl_method_name),": into CL TESTING now...")
    H = []
    Ua = []
    Ue = []
    Yp = []
    L = []

    Y = []
    Na = []
    Nm = []
    M = []
    T = []

    for sub in scenarioname:
        data = np.load(data_dir+'/'+'val'+'/'+ 'val' +'_'+sub+'.npz', allow_pickle=True)
        Y.append(data['intention'])
    

        if return_heatmap:
            Na.append(data['nbagents'])
            Nm.append(data['nbsplines'])
            T.append( data['trajectory'])
            M.append(data['maps'])

    Y = np.concatenate(Y, axis=0)
    if return_heatmap:
        Na = np.concatenate(Na, axis=0)
        Nm = np.concatenate(Nm, axis=0)
        T = np.concatenate(T, axis=0)
        M = np.concatenate(M, axis=0)


    #     print("------------here--------------")
    # if not test:
    #     Y = data['intention']
    #     print("==================Y==================", Y.shape)
    # if scenarioname == 'EP' or scenarioname == 'EP1':
    #     Y=Y[:-1]
    #     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    # else:
    #     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # print("len of ds:", len(loader))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    Hp = []
    Lp = []
    Hi = []
    Li = []
    #Tr is to store the traj for plotting
    Tr = []
    for k, data in enumerate(loader):
        if mode=='lanescore':
            if not test:
                traj, splines, masker, lanefeature, adj, af, ar, c_mask, y, ls = data
            else:
                traj, splines, masker, lanefeature, adj, af, ar, c_mask = data
            inputs = (traj, splines, masker, lanefeature, adj, af, ar, c_mask)
            lsp, heatmap = model(inputs)
            Hi.append(heatmap.detach().to('cpu').numpy())
            Li.append(lsp.detach().to('cpu').numpy())
            # if args.store_traj:
            # # store each batch of traj (batchsize, 26vehicles, 9timestamps, 8features)
            #     Tr.append(traj.detach().to('cpu').numpy())
    Hi = np.concatenate(Hi,0)
    Li = np.concatenate(Li,0)
    # if args.store_traj:
    # # concatenate all batches of traj
    #     Tr = np.concatenate(Tr, 0) # (number of cases, 26, 9, 8)
        
    
    Hp.append(Hi)
    Lp.append(Li)
    Hp = np.stack(Hp, 0)
    Lp = np.stack(Lp, 0)
    hm, ua, ue = ComputeUQ(Hp, para['resolution'], epsilon=5e-4)
    print("hm:",hm.shape)
    yp = ModalSampling(hm, para, r=3, k=6) #multi-madal prediction, k=num of modal
    Ua.append(ua)
    Ue.append(ue)
    Yp.append(yp)
    if return_heatmap:
        H.append(hm)
        L.append(Lp.squeeze())

    Ua = np.concatenate(Ua, 0)
    Ue = np.concatenate(Ue, 0)
    Yp = np.concatenate(Yp, 0)
    if args.store_traj:
        H.append(hm)
        H = np.concatenate(H, 0) 
    ##
    if return_heatmap:
        H = np.concatenate(H, 0) 

    if test:
        if return_heatmap:
            return M, T, Nm, Na, Yp, Ua, Ue, H#, L
        else:
            return Yp, Ua, Ue
    else:
        if return_heatmap:
            return M, T, Nm, Na, Yp, Ua, Ue, H, Y
        else:
            if args.store_traj:
                np.savez_compressed('./logging/prediction_record/'+cl_method_name+"_buffer"+str(args.buffer_size)+'_{:.0f}tasks_learned'.format(trained_to)+'_test_on_'+scenarioname,  pred = Yp, gt = Y, heatmap = H)

            return Yp, Ua, Ue, Y