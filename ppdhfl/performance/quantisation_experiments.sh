#!/bin/bash


datasets=("mnist" "har" "nbaiot" "cifar10" "cifar100" "tinyimagenet")

for framework in "ppdhfl" "fedavg"; do
     for dataset in ${datasets[@]}; do
	if [[ $framework == "fedavg" ]] || [[ $framework == "local" ]]; then
            allocation="full"
        else
            allocation="sim"
        fi

        if [[ $dataset == "mnist" ]]; then
            rounds=50
            batch_size=128
            clients=100
        elif [[ $dataset == "nbaiot" ]]; then
            rounds=10
            batch_size=128
            clients=0
        elif [[ $dataset == "cifar100" ]] || [[ $dataset == "tinyimagenet" ]]; then
            rounds=100
            batch_size=32
            clients=10
        else
            rounds=50
            batch_size=128
            clients=0
        fi

	for seed in {1..5}; do
            python main.py --rounds $rounds --dataset $dataset --framework $framework --seed $seed --allocation $allocation --batch-size $batch_size --clients $clients -q
        done
    done
done
