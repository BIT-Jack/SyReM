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
## Running
After adding the Executable Permissions to the provided bash file (_bash_training_and_test.sh_), you can directly run the training and testing with command:
```
./bash_training_and_test.sh
```
## Code
