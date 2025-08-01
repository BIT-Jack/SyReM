#!/bin/bash


#training and test CL...
echo " Running Continual Training EXP"
echo $(pwd)


python train_CL.py --model gss --dataset seq-interaction --buffer_size 2000 --debug_mode 0 --train_task_num 12 --device cuda:0 & P1=$!
wait $P1

python test_CL_bi_direct.py --model gss --buffer_size 0 --num_tasks 12 --device cuda:0 & P2=$!
wait $P2

echo "All scripts are excuted."
