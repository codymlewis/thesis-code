#!/bin/bash

for checkpoint in checkpoints/*; do
  for l1_reg in 0.0 0.1 0.01 0.001 0.0001 0.00001 0.000001; do
    for l2_reg in 0.0 0.1 0.01 0.001 0.0001 0.00001 0.000001; do
    	python attack.py -f $checkpoint -r 5 -b 8 --optimiser sgd --zinit data --l1-reg $l1_reg --l2-reg $l2_reg
  	done
	done
done

for checkpoint in precode_checkpoints/*; do
  for l1_reg in 0.0 0.1 0.01 0.001 0.0001 0.00001 0.000001; do
    for l2_reg in 0.0 0.1 0.01 0.001 0.0001 0.00001 0.000001; do
    	python precode.py --attack -f $checkpoint --runs 5 --batch-size 8 --l1-reg $l1_reg --l2-reg $l2_reg
  	done
	done
done
