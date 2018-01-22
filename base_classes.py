import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

class NetworkCopier():
    def __init__(self, estimator,target):
        est_params = [variable for variable in tf.trainable_variables() if variable.name.startswith(estimator.scope)]
        est_params = sorted(est_params, key = lambda x: x.name)

        tar_params = [variable for variable in tf.trainable_variables() if variable.name.startswith(target.scope)]
        tar_params = sorted(tar_params, key = lambda x: x.name)

        self.update_ops = [tar_var.assign(est_var) for est_var,tar_var in zip(est_params,tar_params)]
        return
    
    
    def copy_and_freeze(self,sess): 
        sess.run(self.update_ops)
        return
        
        
class ReplayBuffer():
    def __init__(self, max_size = 50000):
        self.buffer = deque(maxlen = max_size)
        return
        
        
    def add_new(self, state, action, reward, next_state, done):
        entry = (state,action,reward,next_state,done)
        self.buffer.append(entry)
        return
        
        
    def batch(self, n = 100):
        if len(self.buffer) < n:
            minibatch = 0
        else:
            minibatch = random.sample(self.buffer, n)
        return minibatch


class Q_learner():
    def __init__(self, state_size, action_size, lr = 0.001, scope='default'):
        self.scope = scope
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        with tf.variable_scope(self.scope):
            self.set_placeholders(self.state_size, self.action_size)
            self.build_model()
            self.set_loss_and_opt(self.lr)
        return
    
    
    def set_placeholders(self, state_size, action_size):
                # state
        self.X_states = tf.placeholder(shape = [None] + self.state_size, dtype = tf.float32)
        
        # target values (R+maxQ)
        self.Q_targets = tf.placeholder(shape = [None] + self.action_size, dtype = tf.float32)
        
        return
    
    
    def build_model(self):
        pass
    
    
    def set_loss_and_opt(self, lr):
        self.loss = tf.losses.mean_squared_error(self.Q_targets, self.Q_est)
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss)
        
        return
    
    
    def predict(self, sess, state):
        return sess.run(self.Q_est, { self.X_states: state })
     
    
    def update(self, sess, states, targets):
        feed_dict = { self.X_states: states, self.Q_targets: targets }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss
    
   
