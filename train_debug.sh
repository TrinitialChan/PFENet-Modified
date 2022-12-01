#!/bin/sh
model=resnet50
niter=30
batchsize=8
device=3
lr=1e-3
skipnovel=0
expname=debug
optim=adam

CUDA_VISIBLE_DEVICES=${device} python train.py \
                --backbone ${model}\
                --fold 0\
                --seed 3407\
                --lr ${lr}\
                --bsz ${batchsize}\
                --niter ${niter}\
                --optim ${optim}\
                --skipnovel ${skipnovel}\
                --logpath ${expname}/$(date +"%Y%m%d_%H%M%S") 2>&1