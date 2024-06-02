import tensorflow as tf
import numpy as np

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
    def __init__(self, num_features, layer_size, num_actions, epsilon=.1,
                 learning_rate=9e-4):
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

            self.ratio = self.new_responsible_outputs / self.old_responsible_outputs

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


class CPPOPolicyNetwork():
    def __init__(self, num_features, layer_size, num_actions, std, if_tanh=False, epsilon=.1, learning_rate=9e-4):
        self.tf_graph = tf.Graph()
        self.std = std
        self.if_tanh = if_tanh

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
            if self.if_tanh:
                self.output = tf.nn.tanh(tf.matmul(self.observations, self.W[0]) + self.B[0])
                self.output = tf.nn.tanh(tf.matmul(self.output, self.W[1]) + self.B[1])
            else:
                self.output = tf.nn.relu(tf.matmul(self.observations, self.W[0]) + self.B[0])
                self.output = tf.nn.relu(tf.matmul(self.output, self.W[1]) + self.B[1])
            self.output = tf.nn.sigmoid(tf.matmul(self.output, self.W[2]) + self.B[2])
            self.mean = self.output

            self.sampling = self.mean + self.std * tf.random_normal(tf.shape(self.mean))
            self.neglogprob = 0.5 * tf.square((self.sampling - self.mean) / self.std) + 0.5 * np.log(2.0 * np.pi) + tf.log(self.std)

            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            self.chosen_actions = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)
            self.OLDNEGLOGPAC = tf.placeholder(shape=[None], dtype=tf.float32)

            self.old_responsible_outputs = 0.5 * tf.reduce_sum(tf.square((self.chosen_actions - self.mean) / self.std), axis=-1) + 0.5 * np.log(2.0 * np.pi) + tf.log(self.std)

            #self.ratio = self.new_responsible_outputs / self.old_responsible_outputs
            self.ratio = tf.exp(self.OLDNEGLOGPAC - self.old_responsible_outputs)

            self.loss = tf.reshape(
                tf.minimum(
                    tf.multiply(self.ratio, self.advantages),
                    tf.multiply(tf.clip_by_value(self.ratio, 1 - epsilon, 1 + epsilon), self.advantages)),
                [-1]
            )
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
        #sampling, neglogprob, mean = self.session.run([self.sampling, self.neglogprob, self.mean], feed_dict={self.observations: states})
        sampling, neglogprob = self.session.run([self.sampling, self.neglogprob], feed_dict={self.observations: states})
        sampling = np.clip(sampling, 0, 1)
        return sampling, neglogprob

    def get_mean(self, states):
        mean = self.session.run(self.mean, feed_dict={self.observations: states})
        return mean

    def update(self, states, chosen_actions, neglogproba, ep_advantages):
        #is the next line mandatory?
        # self.session.run(self.output, feed_dict={self.observations: states})
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
            self.OLDNEGLOGPAC: neglogproba[:, 0]
        })

    def save_w(self, name):
        self.saver.save(self.session, name + '.ckpt')

    def restore_w(self, name):
        self.saver.restore(self.session, name + '.ckpt')
