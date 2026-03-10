import sys
import os
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')
from argparse import ArgumentParser
import torch
import numpy  # needed (don't change it)
import importlib

#file path
root_dir = './'


def args_loading():
    torch.set_num_threads(4)
    parser = ArgumentParser(description='CL for interactive behavior learning', allow_abbrev=False)
    parser.add_argument('--dataset', type=str, default= 'seq-interaction')   # 'joint-interaction'    'seq-interaction'
    parser.add_argument('--model', type=str, 
                        default= 'der', 
                        help='Model name.')


    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')
    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])

    parser.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')

    parser.add_argument('--validation', default=0, choices=[0, 1], type=int,
                        help='Test on the validation set')
    parser.add_argument('--ignore_other_metrics', default=0, choices=[0, 1], type=int,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=1, help='Run only a few forward steps per epoch')
    parser.add_argument('--train_task_num', type=int, default=12, help='The Number of Continual Tasks for Training')
    #baseline in plastisity
    parser.add_argument('--train_task_id', type=int, default=0, help='The index of the task in Single Training')
    parser.add_argument('--test_task_id', type=int, default=1, help='The index of the task in Single Testing')
    
    parser.add_argument('--buffer_size', type=int,default= 2000,
                                help='The size of the memory buffer.')

    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')
    parser.add_argument('--lr', type=float, default= 0.001,
                        help='Learning rate.')
    parser.add_argument('--n_epochs', type=int, default= 1,
                        help='n_epochs.')
    parser.add_argument('--batch_size', type=int, default= 8,
                        help='Batch size.')
    parser.add_argument('--alpha', type=float, default= 1.0,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, default= 1.0,
                        help='Penalty weight.')
    parser.add_argument('--gamma', type=float, default= 0.5,
                        help='the added constant to solve QP in GEM')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default= device)
    
    #parameter for decide whether record the amount of replayed data in each task or not
    parser.add_argument('--replayed_rc', type=bool, default=False,
                        help='turn True for replayed data logging')
    parser.add_argument('--store_traj', type=bool, default=False,
                        help='turn True for test trajectory recording')
    parser.add_argument('--restart_training', type=bool, default=False,
                        help='True for restart')
    parser.add_argument('--restart_pre_task_num', type=int, default=1,
                        help='scenario index of restarted training')
    parser.add_argument('--num_candidate', type=int, default=8, help='sample number of the candidates for comparison during memory selectoin')

    parser.add_argument('--num_tasks', type=int, default=11, help='The number of continuous tasks to be tested.')
    parser.add_argument('--learned_tasks', type=int, default=3, help='The number of learned tasks for the model to be tested.')


    parser.add_argument('--task_order_id', type=int, default=0,
                        help='task order index for order selection')
    
    parser.add_argument('--e_lambda', type=float, default=1,
                        help='lambda weight for EWC')

    parser.add_argument('--alpha_p', default=0.1, type=float, help='plasticity scaler')


    ## For AdaBOP
    # Adaptive gradient clipping boundary (initial value)
    parser.add_argument('--adabop_clip_bound', type=float, default=1.0, 
                        help='ADABOP: Initial adaptive gradient clip boundary')
    # Decay rate for adaptive boundary adjustment
    parser.add_argument('--adabop_bound_decay', type=float, default=0.995, 
                        help='ADABOP: Decay rate for adaptive boundary')
    # Minimum value of adaptive boundary to avoid underflow
    parser.add_argument('--adabop_min_bound', type=float, default=0.1, 
                        help='ADABOP: Minimum adaptive boundary value')
    # Loss scaling factor for adaptive loss boundary
    parser.add_argument('--adabop_loss_scale', type=float, default=1.0, 
                        help='ADABOP: Loss scaling factor for boundary adjustment')


    args = parser.parse_args()  # parameters for a specific method
    mod = importlib.import_module('cl_model.' + args.model)
    get_parser = getattr(mod, 'get_parser')
    parser = get_parser(parser)
    return args



args = args_loading()




baseline_dir = root_dir+'results/baseline_weights'
data_dir = root_dir+'cl_dataset'
logging_dir = root_dir+'logging'

saved_dir = root_dir+'results/'+'order_'+str(args.task_order_id)+'/weights'
result_dir = root_dir+'results/'+'order_'+str(args.task_order_id)+'/logs'


scenario_info_set = [{0:'MA', 1:'FT', 2:'LN', 3:'ZS2', 4:'OF', 5:'EP0', 6:'GL', 7:'ZS0', 8:'EP', 9:'MT', 10:'SR', 11:'EP1'}, 
    {
    0: 'EP',
    1: 'MA',
    2: 'ZS0',
    3: 'FT',
    4: 'GL',
    5: 'EP1',
    6: 'LN',
    7: 'SR',
    8: 'OF',
    9: 'ZS2',
    10: 'MT',
    11: 'EP0'
    },
    {
    0: 'GL',
    1: 'ZS2',
    2: 'MA',
    3: 'EP0',
    4: 'SR',
    5: 'FT',
    6: 'EP',
    7: 'LN',
    8: 'MT',
    9: 'ZS0',
    10: 'OF',
    11: 'EP1'
    }
    ]

#Training setting of UQnet:
paralist = dict(xmax = 23,#23,
                ymin = -12,#-12,
                ymax = 75,
                resolution = 0.5,
                nb_map_vectors = 5,
                nb_traj_vectors = 9,
                map_dim = 5,
                traj_dim = 8,
                nb_map_gnn = 5,
                nb_traj_gnn = 5, 
                nb_mlp_layers = 3,
                c_out_half = 32,
                c_mlp = 64,
                c_out = 96,
                encoder_nb_heads = 3,
                encoder_attention_size = 128,
                encoder_agg_mode = "cat",
                decoder_attention_size = 64,
                decoder_nb_heads = 3,
                decoder_agg_mode = "cat",
                decoder_masker = False,
                sigmax = 0.6,
                sigmay = 0.6,
                r_list = [2,4,8,16],
                kf = 1,
                model = 'densetnt',
                sample_range=1,
                use_masker=False, 
                lane2agent='lanegcn',
                use_sem=False,
                mode='lanescore',
                prob_mode='ce',
                inference=False #when testing turn this into True to calculate uncertainty
                )
