"""
Module for secure aggregation servers
"""


class Server:

    def __init__(self, network, params, R=2**16 - 1):
        network.setup()
        self.network = network
        self.R = R

    def step(self):
        "Perform a step of aggregation of gradients"
        self.network.begin("grads")
        return self.network(self.R)

    def analysis(self):
        "Perform a step of aggregation of the loss values"
        self.network.begin("loss")
        loss, count = self.network(self.R)
        return loss[0] / count


class FedAvgServer(Server):
    "Standard federated averaging server"
    def step(self):
        grads, count = super().step()
        self.network.send_grads(grads / count)


class NormServer(Server):
    "Server for the PDHFL norm scaling algorithm"
    def step(self):
        grads, _ = super().step()
        self.network.send_grads(grads)
