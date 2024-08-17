import tensorflow as tf
import numpy as np
import sys
import math

class ValueNetwork():
    def __init__(self, num_features, hidden_size, learning_rate=.01):
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.session = tf.Session()

            self.observations = tf.placeholder(shape=[None, self.num_features], dtype=tf.float32)
            self.W = [
                tf.get_variable("W1", shape=[self.num_features, self.hidden_size]),
                tf.get_variable("W2", shape=[self.hidden_size, self.hidden_size]),
                tf.get_variable("W3", shape=[self.hidden_size, 1])
            ]
            self.B = [
                tf.get_variable("B1", [self.hidden_size]),
                tf.get_variable("B2", [self.hidden_size]),
                tf.get_variable("B3", [1]),
            ]
            self.layer_1 = tf.nn.relu(tf.matmul(self.observations, self.W[0]) + self.B[0])
            self.layer_2 = tf.nn.relu(tf.matmul(self.layer_1, self.W[1]) + self.B[1])
            self.output = tf.reshape(tf.matmul(self.layer_2, self.W[2]) + self.B[2], [-1])

            self.rollout = tf.placeholder(shape=[None], dtype=tf.float32)
            self.loss = tf.losses.mean_squared_error(self.output, self.rollout)
            self.grad_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.minimize = self.grad_optimizer.minimize(self.loss)

            init = tf.global_variables_initializer()
            self.session.run(init)

    def get(self, states):
        value = self.session.run(self.output, feed_dict={self.observations: states})
        return value

    def update(self, states, discounted_rewards):
        _, loss = self.session.run([self.minimize, self.loss], feed_dict={
            self.observations: states, self.rollout: discounted_rewards
        })


class PPOPolicyNetwork():
    def __init__(self, name, num_features, layer_size, num_actions, epsilon=.1,
                 learning_rate=9e-4, alpha=1e-3, beta=1,lamdaw = 5,lamdae = 0):
        self.name = name
        self.alpha = alpha
        self.beta = beta
        self.lamdaw = lamdaw
        self.lamdae = lamdae
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.session = tf.Session()

            self.observations = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
            self.W = [
                tf.get_variable("W1", shape=[num_features, layer_size]),
                tf.get_variable("W2", shape=[layer_size, layer_size]),
                tf.get_variable("W3", shape=[layer_size, num_actions])
            ]
            self.B = [
                tf.get_variable("B1", [layer_size]),
                tf.get_variable("B2", [layer_size]),
                tf.get_variable("B3", [num_actions]),
            ]
            trainable_vars = [item for sublist in [self.W, self.B] for item in sublist]
            self.saver = tf.train.Saver(trainable_vars, max_to_keep=3000)

            self.output = tf.nn.relu(tf.matmul(self.observations, self.W[0]) + self.B[0])
            self.output = tf.nn.relu(tf.matmul(self.output, self.W[1]) + self.B[1])
            self.output = tf.nn.softmax(tf.matmul(self.output, self.W[2]) + self.B[2])

            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            self.chosen_actions = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)
            self.old_probabilities = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)

            self.new_responsible_outputs = tf.reduce_sum(self.chosen_actions * self.output, axis=1)
            self.old_responsible_outputs = tf.reduce_sum(self.chosen_actions * self.old_probabilities, axis=1)

            self.ratio = self.new_responsible_outputs / (self.old_responsible_outputs + 1e-10)
            self.loss = tf.reshape(
                    tf.minimum(
                        tf.multiply(self.ratio, self.advantages),
                        tf.multiply(tf.clip_by_value(self.ratio, 1 - epsilon, 1 + epsilon), self.advantages)),
                    [-1]
                ) - 0.03 * self.new_responsible_outputs * tf.log(self.new_responsible_outputs + 1e-10)
            self.loss = -tf.reduce_mean(self.loss)

            self.W0_grad = tf.placeholder(dtype=tf.float32)
            self.W1_grad = tf.placeholder(dtype=tf.float32)
            self.W2_grad = tf.placeholder(dtype=tf.float32)

            self.B0_grad = tf.placeholder(dtype=tf.float32)
            self.B1_grad = tf.placeholder(dtype=tf.float32)
            self.B2_grad = tf.placeholder(dtype=tf.float32)

            self.gradient_placeholders = [self.W0_grad, self.W1_grad, self.W2_grad, self.B0_grad, self.B1_grad,
                                          self.B2_grad]
            self.trainable_vars = [item for sublist in [self.W, self.B] for item in sublist]
            self.gradients = [(np.zeros(var.get_shape()), var) for var in self.trainable_vars]

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.get_grad = self.optimizer.compute_gradients(self.loss, self.trainable_vars)
            self.apply_grad = self.optimizer.apply_gradients(zip(self.gradient_placeholders, self.trainable_vars))
            init = tf.global_variables_initializer()
            self.session.run(init)

    def get_dist(self, states):
        dist = self.session.run(self.output, feed_dict={self.observations: states})
        return dist

    def update(self, states, chosen_actions, ep_advantages):
        old_probabilities = self.session.run(self.output, feed_dict={self.observations: states})
        self.session.run(self.apply_grad, feed_dict={
            self.W0_grad: self.gradients[0][0],
            self.W1_grad: self.gradients[1][0],
            self.W2_grad: self.gradients[2][0],
            self.B0_grad: self.gradients[3][0],
            self.B1_grad: self.gradients[4][0],
            self.B2_grad: self.gradients[5][0],
        })

        self.gradients, loss = self.session.run([self.get_grad, self.output], feed_dict={
            self.observations: states,
            self.advantages: ep_advantages,
            self.chosen_actions: chosen_actions,
            self.old_probabilities: old_probabilities
        })

    def save_w(self, name):
        self.saver.save(self.session, name + '.ckpt')

    def restore_w(self, name):
        self.saver.restore(self.session, name + '.ckpt')

    def set_parameter(self, kwargs):
        dict = ["W1:0", "W2:0", "W3:0","B1:0", "B2:0", "B3:0"]
        for name in dict:
            local_parameters = self.session.run(self.tf_graph.get_tensor_by_name(name))
            if self.name == "Ego":
                uti_parameters = kwargs["Uti"][name]
                ega_parameters = kwargs["Ega"][name]
                local_parameters = local_parameters - self.alpha * self.lamdaw * (local_parameters - uti_parameters)
                local_parameters = local_parameters - self.alpha * self.lamdae * (local_parameters - ega_parameters)
            if self.name == "Uti":
                ego_parameters_list = kwargs["Ego"]
                ega_parameters = kwargs["Ega"][name]
                local_parameters = (1 - self.beta) * local_parameters
                for ego_parameters in ego_parameters_list:
                    param = ego_parameters[name]
                    local_parameters = local_parameters + self.beta * param / len(ego_parameters_list)
                local_parameters = local_parameters - self.alpha * self.lamdae * (local_parameters - ega_parameters)
            update = tf.assign(self.tf_graph.get_tensor_by_name(name), local_parameters)
            self.session.run(update)

    def broad_paramters(self):
        dict = ["W1:0", "W2:0", "W3:0", "B1:0", "B2:0", "B3:0"]
        parameters = {}
        for name in dict:
            parameters[name] = self.session.run(self.tf_graph.get_tensor_by_name(name))
        return parameters




