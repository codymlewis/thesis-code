#!/bin/bash

for optimiser in 'sgd' 'dpsgd' 'secadam' 'dpsecadam'; do
  python attack.py -f checkpoints/seed=42-epochs=100-batch_size=128-dataset=fmnist-model=CNN-optimiser=secadam-learning_rate=0.001-pgd=True-perturb=False.safetensors -r 30 -b 1 --optimiser $optimiser --zinit data --l1-reg 0.0 --l2-reg 0.001
done
