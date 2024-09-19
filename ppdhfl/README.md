# Ensuring Gradient Privacy in Personalized Device Heterogeneous Federated Learning Environment

To see the full commit history, refer to https://github.com/codymlewis/ppdhfl


## Abstract

With the increasing tension between conflicting requirements of the availability of large amounts of data for effective machine learning-based analysis, and for ensuring their privacy, the paradigm of federated learning has emerged, a distributed machine learning setting where the clients provide only the machine learning model updates to the server rather than the actual data for decision making. However, the distributed nature of federated learning raises specific challenges related to fairness in a heterogeneous setting. This motivates the focus of our article, on the heterogeneity of client devices having different computational capabilities and their impact on fairness in federated learning. Furthermore, our aim is to achieve fairness in heterogeneity while ensuring privacy. As far as we are aware there are no existing works that address all three aspects of fairness, device heterogeneity, and privacy simultaneously in federated learning. In this article, we propose a novel federated learning algorithm with personalization in the context of heterogeneous devices while maintaining compatibility with the gradient privacy preservation techniques of secure aggregation. We analyze the proposed federated learning algorithm under different environments with different datasets and show that it achieves performance close to or greater than the state-of-the-art in heterogeneous device personalized federated learning. We also provide theoretical proofs for the fairness and convergence properties of our proposed algorithm.


## Organisation

- The benchmark folder contains experiments concerning the benchmark algorithm in our proposed procedure.
- Performance contains experiments evaluating the performance of our algorithm, comparing to others state-of-the-art algorithms.
- Secure aggregation includes complete code with full gradient privacy.
- Ablation explores some alternative approaches to our proposed algorithm.


## Running the Experiments

Each folder contains an `experiments.sh` shell file that allows for the recreation of experiments that are a part of this work.
