# SyReM

Official codes for the paper ''_Escaping Stability-Plasticity Dilemma in Online Continual Learning for Motion Forecasting via Synergetic Memory Rehearsal_'', Yunlong Lin, Jianwei Gong, Chao Lu, Tongshuai Wu, Xiaocong Zhao, Guodong Du, Zirui Li.


# Dataset
## Original Dataset
The experiments in this work are based on [INTERACTION dataset](https://interaction-dataset.com/).
## Processed Data
- The processed data is available in this link for [Google Drive](https://drive.google.com/drive/folders/1roEeNQJFz777DbPEMf21R3j2BQdRKecp?usp=drive_link).
- Please download the processed data in the direction ```./cl_dataset/```.

# Implementations
## Enviroment
1. System: The codes can be run in **Ubuntu 22.04 LTS**.
2. **Python = 3.9**
3. **Pytorch = 2.0.0**
4. Other required packages are provided in "**requirements.txt**":
```
 pip install -r requirements.txt
```
## Configurations
- Before running codes, please revise ```root_dir``` and ```data_dir``` in ```./utils/args_loading.py``` to your local paths.
- Parameters for the networks can be also revised in ```./utils/args_loading.py```.


## Key Parameters for running the experiments
- **--model**: the method you want to train and test.
- **--buffer_size**: the memory size of the continual learning methods to run, and set as 0 when using the vanilla method.
- **--dataset**: set as "seq-interaction" when continual training, set as "joint-interaction" when joint training.
- **--train_task_num**: the number of tasks in continual training.
- **--debug_mode**: _True_ or _1_ when you are debugging, only a few batches of samples will be used in each task for a convenient check. _False_ or _0_ in the formal training.
-  **--num_tasks**: the number of continual tasks for testing.


# Usage 

## Simple Running
After adding the Executable Permissions to the provided bash file (_bash_training_and_test.sh_), you can directly run the training and testing with command:
```
./bash_training_and_test_cuda0.sh
```
## Code File Structure
```text
│  run_train_test_cuda0.sh
│  test_CL_bi_direct.py
│  test_joint.py
│  test_joint2held_out.py
│  test_single.py
│  train_CL.py
│  train_joint.py
│  train_single.py
│
├─cl_dataset
│  ├─train
│  └─val
├─cl_data_stream
│      joint_dataset.py
│      seq_dataset.py
│      traj_dataset.py
│      traj_joint_dataset.py
│
├─cl_model
│      agem.py
│      continual_model.py
│      jotr.py
│      syrem.py
│      vanilla.py
│      __init__.py
│
├─experiments
│      joint_training.py
│      seq_training_all_task.py
│      testing_1_task_bi_direct.py
│      testing_1_task_joint2held_out.py
│      testing_1_task_single.py
│      testing_joint_tasks.py
│      training_single_task.py
│
├─mapfiles
│      DR_CHN_Merging_ZS0.osm
│      DR_CHN_Merging_ZS0.osm_xy
│      DR_CHN_Merging_ZS2.osm
│      DR_CHN_Merging_ZS2.osm_xy
│      DR_CHN_Roundabout_LN.osm
│      DR_CHN_Roundabout_LN.osm_xy
│      ...
│
├─results
│  ├─logs
│  └─weights
├─traj_predictor
│      baselayers.py
│      decoder.py
│      encoder.py
│      evaluation.py
│      inference.py
│      interaction_model.py
│      losses.py
│      utils.py
│      __init__.py
│
├─utils
│      args_loading.py
│      memory_buffer.py
│      metrics.py
│
└─visualization_utils
        dictionary.py
        dict_utils.py
        extract_original_tv_info.py
        map_vis_without_lanelet.py
        traj_legend.py
        traj_plot.py
        traj_plot_single_fig.py
```
## Main Program

### Model Training

- Continual training in sequential tasks :
  ```
  python train_CL.py
  ```

- Train the model within a specific task (optional task should be chosen in hyper parameter, refering to args_loading.py):
  ```
  python train_single.py
  ```
- Joint training using multiple datasets (mixed training data among multi-tasks):
  ```
  python train_joint.py
  ```

### Model testing

- Test the model in all tasks (previously learned, current, unseen ones):
```
python test_CL_bi_direct.py
```

- Test the model in a specific test set of one task:
```
python test_single.py
```

- Test the model in a joint testing set, which covers test data from multiple tasks:
```
python test_joint.py
```

⚠ Please set your chosen hyper-parameters (the .sh file in this repo is referred to as an example of setting neccessary hyper-parameters) in args_loading.py before running.


# Contact
Please feel free to contact our main contributors if you have any questions or suggestions!

Yunlong Lin (Email: jacklyl.bit@gmail.com)
