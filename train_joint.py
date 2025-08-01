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
from cl_data_stream.joint_dataset import Joint_INTERACTION
from experiments.joint_training import train
from utils.args_loading import *
from traj_predictor.losses import *
from traj_predictor.utils import *
from traj_predictor.interaction_model import UQnet



def main():
    args = args_loading()
    dataset = Joint_INTERACTION(args)
    args.paralist = paralist
    args.model = 'vanilla'
    args.minibatch_size = args.batch_size
    backbone = UQnet(args.paralist, test=True, drivable=False).to(args.device)
    loss = OverAllLoss(args.paralist).to(device)
    model = get_model(args, backbone, loss)

    print("\nJoint Training...\n")
    train(model, dataset, args)

if __name__ == '__main__':
    main()
    
