# Comparison of the Proposed Local Averaging Methods

In the design of the algorithm, we have considered four distinct methods of learning for the clients. This section compares these methods empirically. The methods are each composed of two orthogonal parts: (1) the "averaging" of the global model and (2) the initialization and training strategy employed by clients in each round. For the former, there we have the counter method and the norm scaling method described above. For the latter, there is the global model setting (GMS), where clients set their local parameters to the partition of the global parameters and locally train as normal. Alternatively, there is the knowledge distillation (KD) method, where clients never change the value of their local model outside the training but instead perform knowledge distillation from the global model.

The table below provides a comparison of performance of the four methods using 10 clients based federated learning system with the MNIST dataset. For each case, our environment consists of 1 epoch of local training performed over 5000 rounds to 10 epochs of local training over 500 rounds. These environments perform an equal number of steps of training, where the former shows the performance under a setting close to that of local stochastic gradient descent, while the latter shows the performance with a lower communication overhead. Due to the stochasticity of the learning process, all values in the table are taken as the average over five experiments.

The table below shows the accuracies (Acc.) achieved by the proposed methods on the MNIST dataset within a 10 client network with respect to 1 and 10 epochs of local training. We denote the global model setting as GMS, knowledge distillation as KD, norm scaling as NS, and counter as CTR. Accuracy is calculated based on the set of values achieved by the clients at the end of the final round of training.

| **Method** | **E (R)** | **Acc. Mean (STD)** | **Acc. Range** |
|------------|-----------|---------------------|----------------|
| GMS NS |  1 (5000) |  97.718\% (0.194\%) |  $[97.485, 97.927]$\% |
| GMS NS |  10 (500) |  96.761\% (0.501\%) |  $[96.111, 97.298]$\% |
| KD NS |  1 (5000) |  77.529\% (4.812\%) |  $[71.908, 83.062]$\% |
| KD NS |  10 (500) |  77.540\% (3.678\%) |  $[73.680, 82.044]$\% |
| GMS CTR |  1 (5000) |  39.037\% (40.938\%) |  $[10.090, 96.932]$\% |
| GMS CTR |  10 (500) |  10.090\% (0.000\%) |  $[10.090, 10.090]$\% |
| KD CTR |  1 (5000) |  10.090\% (0.000\%) |  $[10.090, 10.090]$\% |
| KD CTR |  10 (500) |  10.090\% (0.000\%) |  $[10.090, 10.090]$\% |

The results of our experiments show that the norm scaling technique with global model setting performs the best. The better performance of norm scaling over the counter method can be explained by the fact norm scaling removes the variance among client updates in the federated stochastic gradient descent and hence more closely approximates federated gradient descent. This can be seen in the convergence proof in our main paper's appendix. The counter method calculates specific averages, which result in a greater disparity among the client updates benefiting the clients holding intermediate size models. The method is seen to be ineffective when more than one 1 epoch is performed during each round. The global model setting performs better than knowledge distillation, as the latter attempts to use a global model that is being compiled by the clients to simultaneously influence the clients, creating a conflict resulting in local learning epochs continually attempting to mildly step the model back towards the initialization model.
