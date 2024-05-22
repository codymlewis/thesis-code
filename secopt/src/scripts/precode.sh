#!/bin/bash


echo "Evaluating performance of the precode models..."

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

  for seed in {1..5}; do
    python precode.py -s $seed --dataset $dataset --model $model --batch-size $batch_size --learning-rate $lr --performance
    python precode.py -s $seed --dataset $dataset --model $model --batch-size $batch_size --learning-rate $lr --secadam --performance
  done
done

echo "Training models to be inverted..."

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
    python precode.py --epochs 100 --dataset $dataset --model $model --batch-size $batch_size --learning-rate $lr $extra_flags --train-inversion
    python precode.py --epochs 100 --dataset $dataset --model $model --batch-size $batch_size --learning-rate $lr --secadam $extra_flags --train-inversion
  done
done

echo "Inverting models..."

for checkpoint in precode_checkpoints/*; do
  if echo "$checkpoint" | grep -q 'cifar100'; then
		l1_reg="0.00001"
		l2_reg="0.01"
	elif echo "$checkpoint" | grep -q 'cifar10'; then
		l1_reg="0.00001"
		l2_reg="0.01"
	elif echo "$checkpoint" | grep -q 'fmnist'; then
		l1_reg="0.001"
		l2_reg="0.000001"
	elif echo "$checkpoint" | grep -q 'svhn'; then
		l1_reg="0.001"
		l2_reg="0.001"
	elif echo "$checkpoint" | grep -q 'tinyimagenet'; then
		l1_reg="0.01"
		l2_reg="0.000001"
	fi

  python precode.py -f $checkpoint --runs 30 -b 8 --attack --l1-reg $l1_reg --l2-reg $l2_reg
done

echo "Done."
