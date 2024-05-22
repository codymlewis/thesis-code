#!/bin/bash


for dataset in fmnist cifar10 cifar100 svhn tinyimagenet; do
    if [[ $dataset == "fmnist" ]]; then
        models=("CNN" "LeNet")
    else
        models=("ResNetV2" "ConvNeXt")
    fi

    if [[ $dataset == "cifar100" ]] || [[ $dataset == "tinyimagenet" ]]; then
        batch_size=32
    else
        batch_size=128
    fi

    if [[ $dataset == "cifar100" ]] || [[ $dataset == "tinyimagenet" ]] || [[ $dataset == "svhn" ]]; then
        lr='0.0001'
    else
        lr='0.001'
    fi

    if [[ $dataset == "svhn" ]]; then
        extra_flags=""
    else
        extra_flags="--pgd"
    fi

    for model in ${models[@]}; do
        python train_inversion.py --epochs 100 --dataset $dataset --optimiser secadam --model $model --batch-size $batch_size --learning-rate $lr $extra_flags
    done
done
