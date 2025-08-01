import numpy  
import os
import sys
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')
from cl_model import get_model
from cl_data_stream.seq_dataset import SequentialINTERACTION
from experiments.seq_training_all_task import train
from utils.args_loading import *
from traj_predictor.losses import *
from traj_predictor.utils import *
from traj_predictor.interaction_model import UQnet



def main():
    args = args_loading()
    dataset = SequentialINTERACTION(args)
    args.paralist = paralist
    args.minibatch_size = args.batch_size
    args.gss_minibatch_size  = args.minibatch_size
    backbone = UQnet(args.paralist, test=True, drivable=False).to(args.device)
    loss = OverAllLoss(args.paralist).to(device)
    model = get_model(args, backbone, loss)

    print("\nContinual Training For Each Task...\n")
    train(model, dataset, args)

if __name__ == '__main__':
    main()
    
