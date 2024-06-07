# Application of the Secure Aggregation Algorithm

In this section, we present a comparison of the time taken to perform each step of the secure aggregation algorithm using the norm scaling variation of our algorithm and the federated averaging algorithm. The secure aggregation algorithm has no impact upon the performance of the model. However, it increases the amount of time to perform updates due to introducing cryptographic exchanges. For this reason, we did not apply secure aggregation when evaluating the model performance resulting from our algorithm in this paper. In the table below, we show that our algorithm takes a reasonable amount of time to compute updates under the secure aggregation algorithm and hence can be used in practice.

| **Algorithm** | **Algorithm Step** | **Time (s)** |
|---------------|--------------------|--------------|
| PPDHFL | Step | 0.00815 |
| | Advertise Keys | 0.00532 |
| | Share Keys | 0.267 |
| | Masked Input Collection | 0.332 |
| | Consistency Check | 0.000606 |
| | Unmasking | 0.264 |
| | Receive Grads | 0.00108 |
| | **Total** | 0.878 |
| FedAVG | Step | 0.00824 |
| | Advertise Keys | 0.00535 |
| | Share Keys | 0.267 |
| | Masked Input Collection | 0.331 |
| | Consistency Check | 0.000628 |
| | Unmasking | 0.264 |
| | Receive Grads | 0.000485 |
| | **Total** | 0.876 |

In our experiments, we recorded the time taken by clients to perform each step in the secure aggregation algorithm proposed in https://research.google/pubs/pub45808/. We take the mean of 1000 measurements for each step within a 10 client system. Clients learn the MNIST task with the LeNet-300-100 model using a minibatch size of $32$, SGD optimizer with a learning rate of $0.1$ and cross-entropy loss with 10 local epochs each round. The secure aggregation algorithm has one parameter, the contributor threshold, which determines the number of clients that must contribute to the current round for clients to be confident in the security. This threshold determines the $t$ value within the $t$ out of $|\mathbb{C}|$ Shamir secret sharing used in this algorithm. We set the threshold as the recommend for an active adversary environment, $t = \lfloor\frac{2 |\mathbb{C}|}{3}\rfloor + 1$. Our experiments were performed on a laptop with an Intel i7-8565U CPU.

We see from the table above that our algorithm does not experience any significant differences from the canonical at any step of the algorithm except in the received gradients step. This is to be expected, as this step is the only step modified by our algorithm, which involves taking the norm of received gradient before adding to the local global model. Observing the total time for client computations, we see that the overhead introduced from our algorithm is relatively insignificant at $0.2$% compared to the canonical.
