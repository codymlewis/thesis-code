#!/bin/bash


frameworks=("pdhfl" "heterofl" "fjord" "feddrop" "local" "fedavg")
datasets=("mnist" "har" "nbaiot" "cifar10" "cifar100" "tinyimagenet")


for framework in ${frameworks[@]}; do
    for dataset in ${datasets[@]}; do
        if [[ $framework == "fedavg" ]] || [[ $framework == "local" ]]; then
            allocations=("full")
        else
            allocations=("cyclic" "sim")
        fi

        if [[ $dataset == "nbaiot" ]]; then
            rounds=10
            batch_size=128
            clients_list=(0 90)
        elif [[ $dataset == "cifar100" ]]; then
            rounds=100
            batch_size=32
            clients_list=(10)
        elif [[ $dataset == "tinyimagenet" ]]; then
            rounds=100
            batch_size=32
            clients_list=(10)
        else
            rounds=50
            batch_size=128
            clients_list=(0)
        fi

        for allocation in ${allocations[@]}; do
            for clients in ${clients_list[@]}; do
                for seed in {1..5}; do
                    python main.py --rounds $rounds --dataset $dataset --framework $framework --seed $seed --allocation $allocation --batch-size $batch_size --clients $clients

                    if [[ $clients -ge 90 ]]; then
                        python main.py --rounds $rounds --dataset $dataset --framework $framework --seed $seed --allocation $allocation --batch-size $batch_size --proportion-clients 0.1 --clients $clients
                    fi
                done
            done
        done
    done
done
