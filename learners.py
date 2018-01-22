import base_classes as base
import tensorflow as tf
import numpy as np
import random

class Coordinator():
    def __init__(self, model, model_params, env, buff_size = None, save_dir = None, save_time = 10):
        self.Q_estimator = model(*model_params, scope = 'estimator')
        self.Q_target = model(*model_params, scope = 'target')
        self.Freezer = base.NetworkCopier(self.Q_estimator,self.Q_target)
        self.Buffer = base.ReplayBuffer(buff_size)
        
        self.env = env
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if save_dir:
            self.saver = saver
        
        return
    
   
    def run(self, episodes):
        agent_results = []
        for e in range(episodes):
            done = False
            observation = self.env.reset()
            total_steps = 0
            while not done:
                state_reshape = np.reshape(observation,(1,4))
                q_est = self.Q_estimator.predict(self.sess,state_reshape)[0]
                action = np.argmax(q_est)
                observation, reward, done, info = self.env.step(action)
                total_steps += 1
            agent_results.append(total_steps)
        return agent_results
    
    
    def train(self, episodes, iterations = 10, discount = 0.95, epsilon_decay = 0.995, epsilon_min = 0.1, freeze_rate = 5, max_steps = 1000, batch_size = 50):
        for e in range(episodes):
            epsilon = max(epsilon_decay**e, epsilon_min)
            observation = self.env.reset()
            total_reward = 0
            loss = None

            for step in range(max_steps):
                curr_state = observation
                if np.random.rand() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    curr_state_reshape = np.reshape(curr_state,(1,4))
                    q_est = self.Q_estimator.predict(self.sess,curr_state_reshape)[0]
                    action = np.argmax(q_est)

                observation, reward, done, info = self.env.step(action)
                next_state = observation
                self.Buffer.add_new(curr_state, action, reward, next_state, done)
                total_reward += reward
                if done:
                    break

            for i in range(10):
                minibatch = self.Buffer.batch(batch_size)
                if minibatch:
                    states = []
                    targets = []
                    for state, action, reward, next_state, done in minibatch:
                        state_reshape = np.reshape(state, (1,4))
                        next_state_reshape = np.reshape(next_state, (1,4))            
                        target = self.Q_estimator.predict(self.sess,state_reshape)[0]
                        q_max = np.amax(self.Q_target.predict(self.sess,next_state_reshape)[0])

                        target[action] = reward
                        if not done:
                            target[action] += discount*q_max

                        states.append(state)
                        targets.append(target)
                    state_array = np.stack(states)
                    target_array = np.stack(targets)
                    loss = self.Q_estimator.update(self.sess, state_array, target_array)
            
            if e%freeze_rate == 0 and e > 0:
                self.Freezer.copy_and_freeze(self.sess)
                print('Episode {}, loss = {}, survival = {} steps'.format(e,loss,total_reward))
   
    def load(self):
        pass
    
    
class Q_nn(base.Q_learner):    
    def build_model(self):
        self.dense1 = tf.layers.dense(inputs = self.X_states, units=12, activation = tf.nn.relu)
        self.dense2 = tf.layers.dense(self.dense1,12, activation = tf.nn.relu)
        self.dense3 = tf.layers.dense(self.dense2,12, activation = tf.nn.relu)
        self.Q_est = tf.layers.dense(self.dense3,self.action_size[0])
        
        return