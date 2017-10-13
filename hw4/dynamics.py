import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        
        self.sess = sess
        self.normalization = normalization
        self.batch_size = batch_size
        self.iterations = iterations
        self.eps = np.finfo(float).tiny
        
        self.obs_and_act = tf.placeholder(shape=[None, env.obs_dim+env.action_space.shape[0]], name="ob_act", dtype=tf.float32)
        self.deltas = tf.placeholder(shape=[None, env.obs_dim], name="delta", dtype=tf.float32)
        
        self.out = build_mlp(self.obs_and_act, 
              env.obs_dim,
              "dynamics", 
              n_layers=n_layers, 
              size=size, 
              activation=activation,
              output_activation=output_activation
              )
        
        self.loss = tf.nn.l2_loss(self.out-self.deltas)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        
        obs = (np.concatenate([path["observations"] for path in data]) - self.normalization[0]) / (self.normalization[1]+self.eps)

        delta = np.concatenate([path["next_observations"] for path in data]) - np.concatenate([path["observations"] for path in data])
        delta = (delta - self.normalization[2]) / (self.normalization[3]+self.eps)
        
        acts = (np.concatenate([path["actions"] for path in data]) - self.normalization[4]) / (self.normalization[5]+self.eps)
        
        for i in range(self.iterations):
            perm_ind = np.random.permutation(obs.shape[0])
            for j in range(int(obs.shape[0]/self.batch_size)):
                index = perm_ind[j*self.batch_size:((j+1)*self.batch_size-1)]
                
                l, u = self.sess.run([self.loss, self.update_op], feed_dict={self.obs_and_act: np.concatenate([obs[index], acts[index]], axis=1),
                                                        self.deltas: delta[index]})

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        
        states_n = (states - self.normalization[0]) / (self.normalization[1]+self.eps)
        actions_n = (actions - self.normalization[4]) / (self.normalization[5]+self.eps)

        deltas = self.sess.run(self.out, feed_dict={self.obs_and_act: np.concatenate([states_n, actions_n], axis=1)})
        
        return (deltas*self.normalization[3]) + self.normalization[2] + states