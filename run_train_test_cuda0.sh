#!/bin/bash


#training and test CL...
echo " Running Continual Training EXP"
echo $(pwd)


python train_CL.py --model syrem --dataset seq-interaction --buffer_size 1000 --debug_mode 0 --train_task_num 11 --device cuda:0 & P1=$!
wait $P1

python test_CL_bi_direct.py --model syrem --buffer_size 1000 --num_tasks 11 --device cuda:0 & P2=$!
wait $P2


python train_CL.py --model vanilla-gp --dataset seq-interaction --buffer_size 1000 --debug_mode 0 --train_task_num 11 --device cuda:0 & P1=$!
wait $P3

python test_CL_bi_direct.py --model vanilla-gp --buffer_size 1000 --num_tasks 11 --device cuda:0 & P2=$!
wait $P4

python train_CL.py --model vanilla --dataset seq-interaction --buffer_size 1000 --debug_mode 0 --train_task_num 11 --device cuda:0 & P1=$!
wait $P5

python test_CL_bi_direct.py --model vanilla --buffer_size 1000 --num_tasks 11 --device cuda:0 & P2=$!
wait $P6



echo "All scripts are excuted."
