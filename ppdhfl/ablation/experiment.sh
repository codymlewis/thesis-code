#!/bin/sh


for epochs in 1 10; do
    if [[ $epochs -eq 1 ]]; then
        rounds=5000
    else
        rounds=500
    fi
    for avg in norm_scaling counter; do
        for local in GMS KD; do
            for i in {1..5}; do
                python main.py --id $i --avg $avg --local $local --dataset mnist --rounds $rounds --epochs $epochs
            done
        done
    done
done
