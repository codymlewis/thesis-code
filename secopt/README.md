# Mitigation of Gradient Inversion Attacks in Federated Learning with Private Adaptive Optimization

For the original commit history, see https://github.com/codymlewis/SecOpt/

## Abstract

With the rapid advancements in machine learning as well as an increase in the awareness of privacy issues, federated learning has emerged to be an important paradigm as it greatly reduces the amount of data that need to be directly shared as part of the learning process. However, recently federated learning has been shown to be susceptible to gradient inversion attacks, where an adversary can compromise privacy by recreating the data that lead to a particular client's update. In this paper, we propose a new algorithm, SecAdam, to mitigate such emerging gradient inversion attacks and enable the clients to perform adaptive gradient based training in a federated setting while retaining client gradient privacy. We have given theoretical proofs for these properties as well as providing extensive practical experimental results, which we have carried out on five different datasets using two different neural network architectures. The results from these experiments demonstrate the effectiveness of our proposed algorithm.

## Running the Experiments

From the `src` folder, the inversion experiments can be run by first training the required models with `scripts/train_inversion_models.sh`, then attacking them with `attack.sh`. The performance experiments can be run with `performance.sh`. Both experiments generate a csv file which can be processed and summarized into tables by running `python statistics.py`.
