"""
Standard client collaborators for federated learning.
"""

import tensorflow as tf

import fl
import utils.weights


class Client:
    """A client for federated learning, holds its own data and personal learning variables."""

    def __init__(
        self,
        local_model,
        global_model,
        data,
        epochs,
        eta=0.1,
        step_type="norm_scaling",
        test_data=None,
        training_type="GMS",
        classifier_layers=1
    ):
        """
        Constructor for a Scout.
        Arguments:
        - opt: optimizer to use for training
        - opt_state: initial optimizer state
        - loss: loss function to use for training
        - data: data to use for training
        - epochs: number of epochs to train for per round
        """
        self.data = data
        self.batch_size = data.batch_size
        self.epochs = epochs
        self.model = model
        self.test_data = test_data
        self.global_model = global_model
        self.gm_skeleton = utils.weights.skeleton(self.global_model)
        self.lm_skeleton = utils.weights.skeleton(self.model)
        partitioned_gweights = utils.weights.partition(self.global_model.get_weights(), self.gm_skeleton, self.lm_skeleton)
        self.classifer_layers = classifier_layers
        self.model.set_weights(partitioned_gweights)
        self.eta_0 = eta
        self.am_epochs = max(eta, 1)
        self.round = 0
        self.step = {
            'norm_scaling': self.norm_scaling_step,
            'counter': self.counter_step
        }[step_type]
        self.alpha = 0.5
        self.kd_loss = tf.keras.losses.KLDivergence()
        self.training_type = training_type
        if training_type == "KD":
            self._step = self.kd_step
        else:
            self._step = self.standard_step


    @tf.function
    def standard_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.model.loss(y, logits)
        self.model.optimizer.minimize(loss, self.model.trainable_weights, tape=tape)
        return loss

    def analytics(self):
        return self.model.test_on_batch(*next(self.test_data), return_dict=False)

    def norm_scaling_step(self, global_update, _):
        """
        Perform a single local training loop.

        Arguments:
        - global_update: the most recent aggregated update
        """
        # norm_scaling
        eta = max(self.eta_0 / (1 + 0.0001 * self.round), 0.0001)
        self.global_model.set_weights(utils.weights.sub(
            self.global_model.get_weights(),
            utils.weights.scale(global_update, min(eta / utils.norm(global_update), 1))
        ))
        self.round += 1
        partitioned_gweights = utils.partition(self.global_model.get_weights(), self.gm_skeleton, self.lm_skeleton)
        if self.training_type == "GMS":
            self.model.set_weights(partitioned_gweights)
        for _ in range(self.epochs):
            x, y = next(self.data)
            loss = self._step(x, y)
        updates = utils.expand(utils.weights.sub(partitioned_gweights, self.model.get_weights()), self.lm_skeleton, self.gm_skeleton)
        return loss, updates, self.batch_size

    def counter_step(self, global_update, _):
        global_grad, counter = global_update
        self.round += 1
        self.global_model.set_weights(utils.weights.sub(
            self.global_model.get_weights(),
            utils.weights.div(global_grad, utils.weights.maximum(counter, 1))
        ))
        partitioned_gweights = utils.partition(self.global_model.get_weights(), self.gm_skeleton, self.lm_skeleton)
        if self.training_type == "GMS":
            self.model.set_weights(partitioned_gweights)
        for _ in range(self.epochs):
            x, y = next(self.data)
            loss = self._step(x, y)
        updates = utils.expand(utils.weights.sub(partitioned_gweights, self.model.get_weights()), self.lm_skeleton, self.gm_skeleton)
        return loss, updates, utils.counter(updates)

    @tf.function
    def kd_step(self, x, y):
        global_logits = self.global_model(x, training=False)
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = (1 - self.alpha) * self.model.loss(y, logits) + self.alpha * self.kd_loss(global_logits, logits)
        self.model.optimizer.minimize(loss, self.model.trainable_weights, tape=tape)
        return loss
