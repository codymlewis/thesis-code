#!/usr/bin/env bash


repeat_experiment () {
  for ((i = 1; i <= $2; i++)); do
    $1 -s $i
  done
}

for dataset in "mnist" "cifar10" "kddcup99"; do
  for aggregator in "fedavg" "median" "krum" "foolsgold" "contra" "stddagmm" "viceroy"; do
    for compressor in "none" "autoencoder" "fedzip" "fedmax" "fedprox" "topk"; do
      # for adversary_type in "none" "labelflipper" "scaling_labelflipper" "backdoor" "scaling_backdoor" "freerider" "onoff_labelflipper" "onoff_backdoor" "onoff_freerider" "goodmouther" "badmouther"; do
      for adversary_type in "none" "labelflipper" "backdoor" "freerider" "onoff_labelflipper" "onoff_freerider" "goodmouther" "badmouther"; do
        for percent_adversaries in "0.1" "0.3" "0.5"; do
          repeat_experiment "python main.py --clients 100 --rounds 500 --epochs 10 --dataset $dataset --aggregator $aggregator --compressor $compressor --adversary-type $adversary_type --percent-adversaries $percent_adversaries" 5
        done
      done
    done
  done
done
