# On the Security and Robustness of Federated Learning with Application to the Smart Grid Infrastructure

Source code repository for my thesis work. A link to the thesis itself will be here soon.

## Summary

Federated learning is a distributed machine learning framework designed to improve privacy by enabling clients to share model updates instead of data directly. However, federated learning is not perfect, as it is still susceptible to numerous attacks that can undermine the quality of the model, and even the privacy of the process. In this project, we have studied these flaws by first dividing them into three major archetypes: robustness, privacy, and fairness, then we studied the application of federated learning to the smart grid setting. This project is divided as follows:
- We first study robustness, where adversaries can attack the server's model by submitting adversarial model updates, and considers techniques that mitigate attacks. In our work, we propose and mitigate attacks that exploit mitigation techniques. This work can be found in the `viceroy` folder.
- Afterwards, we consider the issue of privacy, by regarding the recently proposed gradient inversion attacks where a search is performed by the server on model updates to find the data to that calculates them. We then propose an algorithm to better mitigate these attacks, while remaining orthogonal to other privacy improving techniques. This work can be found in the `secopt` folder.
- Then we consider the issue of device heterogeneity fairness, where clients can hold devices of varying power and thus taking differing amounts of time to perform equivalent computations. We propose an algorithm that better improves fairness with respect to equality of opportunity to contribute to the federated learning system, while simultaneously remaining orthoganal to privacy preserving techniques. This work can be found in the `ppdhfl` folder.
- Finally, we conclude by considering the application of federated learning to the smart grid setting. We consider the hierarchical federated learning setting since this best reflects the network structure, and study the likely robustness, fairness, and privacy issues that may occur. This work can be found in the `sghfl` folder.

We also include some demos of some important concepts to this project in the `background_demos` folder.

## Running the code

To run the code, you can use nix (https://nixos.org) to replicate my environment, in which case you will want to enable nix flakes and simply run `nix develop` from the root of this repository to install and open the environment. From there, each of the folders in this repository have execution instructions in their respective `README.md` files, and scripts to run their respective experiments.

If you do not want to use nix, this can also be run by creating a Python virtual environment and installing the requirements from `requirements.txt`, we recommend using Python 3.12 in this case.
