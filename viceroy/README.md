# Viceroy

See https://github.com/codymlewis/viceroy for the original commit history. This folder contains a rewrite of the original code, with better optimisation and using the flax library instead of haiku.


## Abstract

The susceptibility of federated learning (FL) to attacks from untrustworthy endpoints has led to the design of several defense systems. FL defense systems enhance the federated optimization algorithm using anomaly detection, scaling the updates from endpoints depending on their anomalous behavior. However, the defense systems themselves may be exploited by the endpoints with more sophisticated attacks. First, this paper proposes three categories of attacks and shows that they can effectively deceive some well-known FL defense systems. In the first two categories, referred to as on-off attacks, the adversary toggles between being honest and engaging in attacks. We analyse two such on-off attacks, label flipping and free riding, and show their impact against existing FL defense systems. As a third category, we propose attacks based on “good mouthing” and “bad mouthing”, to boost or diminish influence of the victim endpoints on the global model. Secondly, we propose a new federated optimization algorithm, Viceroy, that can successfully mitigate all the proposed attacks. The proposed attacks and the mitigation strategy have been tested on a number of different experiments establishing their effectiveness in comparison with other contemporary methods. The proposed algorithm has also been made available as open source. Finally, in the appendices, we provide an induction proof for the on-off model poisoning attack, and the proof of convergence and adversarial tolerance for the new federated optimization algorithm.


## Organisation

- The main experiment code is found in the `src` folder
- A demo of a backdoor attack in the `backdoor_demo` folder


## Running the Experiments

From the `src` directory, run the `scripts/run_experiments.sh` shell script.
