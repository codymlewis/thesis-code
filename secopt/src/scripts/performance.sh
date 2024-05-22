#!/bin/bash


for dataset in fmnist cifar10 cifar100 svhn tinyimagenet; do
    if [[ $dataset == "fmnist" ]]; then
        model="LeNet"
    else
        model="ResNetV2"
    fi
    if [[ $dataset == "cifar100" ]] || [[ $dataset == "tinyimagenet" ]] || [[ $dataset == "svhn" ]]; then
        batch_size=8
    else
        batch_size=32
    fi
    if [[ $dataset == "cifar100" ]] || [[ $dataset == "tinyimagenet" ]] || [[ $dataset == "svhn" ]]; then
        lr='0.0001'
    else
        lr='0.001'
    fi

    for opt in "sgd" "dpsgd" "secadam" "dpsecadam" "topk" "fedprox" "adam" "fedadam" "regularise"; do
        if [[ $opt == "dpsgd" ]] || [[ $opt == "dpsecadam" ]]; then
            # extra_flags=("-ct 10 -ns 0.00001" "-ct 10 -ns 0.001" "-ct 5 -ns 0.001" "-ct 1 -ns 0.001" "-ct 10 -ns 0.005" "-ct 5 -ns 0.005" "-ct 1 -ns 0.005" "-ct 10 -ns 0.01" "-ct 5 -ns 0.01" "-ct 1 -ns 0.01")
            extra_flags=("-ct 10 -ns 0.00001")
        else
            extra_flags=("")
        fi

        if [[ $opt == "fedprox" ]]; then
            opt_flag="--server-optimiser fedprox --client-optimiser sgd"
        elif [[ $opt == "fedadam" ]]; then
            opt_flag="--server-optimiser adam --client-optimiser sgd"
        elif [[ $opt == "regularise" ]]; then
            opt_flag="--client-optimiser sgd --regularise"
        else
            opt_flag="--client-optimiser $opt"
        fi
        
        for ef in "${extra_flags[@]}"; do
            for seed in {1..5}; do
                python performance.py -s $seed --dataset $dataset $opt_flag --model $model --batch-size $batch_size -slr $lr -clr $lr $ef
            done
        done
    done
done
