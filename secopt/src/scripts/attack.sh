#!/bin/bash


for checkpoint in checkpoints/*; do
	if echo "$checkpoint" | grep -q 'cifar100'; then
		l1_reg="0.0"
		l2_reg="0.0001"
	elif echo "$checkpoint" | grep -q 'cifar10'; then
		l1_reg="0.00001"
		l2_reg="0.0001"
	elif echo "$checkpoint" | grep -q 'fmnist'; then
		l1_reg="0.000001"
		l2_reg="0.000001"
	elif echo "$checkpoint" | grep -q 'svhn'; then
		l1_reg="0.00001"
		l2_reg="0.00001"
	elif echo "$checkpoint" | grep -q 'tinyimagenet'; then
		l1_reg="0.000001"
		l2_reg="0.0001"
	fi

	for optimiser in 'sgd' 'secadam' 'dpsgd' 'dpsecadam' 'topk' 'fedprox'; do
		python attack.py -f $checkpoint -r 30 -b 8 --optimiser $optimiser --zinit data --l1-reg $l1_reg --l2-reg $l2_reg
	done

	python attack.py -f $checkpoint -r 30 -b 8 --optimiser sgd --zinit data --l1-reg $l1_reg --l2-reg $l2_reg --regularise

	for optimiser in 'sgd' 'secadam' 'dpsgd' 'dpsecadam'; do
		python attack.py -f $checkpoint -r 30 -b 8 --optimiser $optimiser --zinit data --l1-reg $l1_reg --l2-reg $l2_reg --steps 3
	done
done
