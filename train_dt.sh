#!/bin/sh
model=resnet50
niter=100
batchsize=48
device=1
lr=1e-4

expname=dt_svf_exp3
optim=sgd-weight-decay
seed=3407
stop_interval=10


for fold in 0 1 2 3
do
svf=0
skipnovel=0
CUDA_VISIBLE_DEVICES=${device} python train.py \
                --backbone ${model}\
                --fold ${fold}\
                --svf ${svf}\
                --seed ${seed}\
                --lr ${lr}\
                --bsz ${batchsize}\
                --niter ${niter}\
                --optim ${optim}\
                --skipnovel ${skipnovel}\
                --stop_interval ${stop_interval}\
                --logpath ${expname}/s_${skipnovel}_f_${fold}_svf_${svf}_$(date +"%Y%m%d_%H%M%S") 2>&1
svf=1
skipnovel=0
CUDA_VISIBLE_DEVICES=${device} python train.py \
                --backbone ${model}\
                --fold ${fold}\
                --svf ${svf}\
                --seed ${seed}\
                --lr ${lr}\
                --bsz ${batchsize}\
                --niter ${niter}\
                --optim ${optim}\
                --skipnovel ${skipnovel}\
                --stop_interval ${stop_interval}\
                --logpath ${expname}/s_${skipnovel}_f_${fold}_svf_${svf}_$(date +"%Y%m%d_%H%M%S") 2>&1
svf=0
skipnovel=1
CUDA_VISIBLE_DEVICES=${device} python train.py \
                --backbone ${model}\
                --fold ${fold}\
                --svf ${svf}\
                --seed ${seed}\
                --lr ${lr}\
                --bsz ${batchsize}\
                --niter ${niter}\
                --optim ${optim}\
                --skipnovel ${skipnovel}\
                --stop_interval ${stop_interval}\
                --logpath ${expname}/s_${skipnovel}_f_${fold}_svf_${svf}_$(date +"%Y%m%d_%H%M%S") 2>&1
done




