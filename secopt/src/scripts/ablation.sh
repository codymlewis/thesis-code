#!/bin/bash


for activation in "relu" "elu" "sigmoid" "leaky_relu" "tanh"; do
	for pooling in "none" "max_pool" "avg_pool"; do
		if [[ $pooling != "none" ]]; then
			extra_flags=("--pool-size small" "--pool-size large")
		else
			extra_flags=("")
		fi
		
		for extra_flag in "${extra_flags[@]}"; do
			for normalisation in "none" "LayerNorm"; do
				for attack in "representation" "idlg" "reg_idlg"; do
					for zinit in "uniform" "repeated_pattern" "colour"; do
						for batch_size in 1 8; do
							python ablation.py -a $activation -p $pooling $extra_flag -n $normalisation --attack $attack --batch-size $batch_size --zinit $zinit
						done
					done
				done
			done
		done
	done
done

