import tensorflow as tf
import numpy as np
import gym

class MLP:
    
    def __init__(self, session, hidden, lr, input_size, output_size, init_means=0, init_stds=.1):
        
        self.sess = session
        self.hidden = hidden
        if not isinstance(self.hidden, list):
            self.hidden = [self.hidden]
        self.input_size = input_size
        self.output_size = output_size
        self.architecture = [self.input_size] + self.hidden + [self.output_size]
        N = len(self.architecture)
        
        with tf.variable_scope('q-func'):
            self.input_ph = tf.placeholder('float', [None, self.input_size])
            self.actions_ph = tf.placeholder('float', [None, self.output_size])
            
            self.weights = [(tf.get_variable('W' + str(i), [self.architecture[i], self.architecture[i+1]], 
                            initializer=tf.random_normal_initializer(mean=init_means, stddev=init_stds)), 
                             tf.get_variable('B' + str(i), [self.architecture[i+1]], 
                            initializer=tf.constant_initializer(-1e-3))) for i in range(N-1)]
            
            self.activation = [tf.nn.relu(tf.matmul(self.input_ph, self.weights[0][0])) + self.weights[0][1]]
            
            for i in range(1,N-2):
                
                self.activation.append(tf.nn.relu(tf.matmul(self.activation[-1], self.weights[i][0])+self.weights[i][1]))
            
            self.activation.append(tf.matmul(self.activation[-1], self.weights[-1][0])+self.weights[-1][1])
            
            
            self.rewards_ph = tf.placeholder('float', [None, 1])
            self.loss = tf.nn.l2_loss(tf.reduce_sum(tf.mul(self.activation[-1], self.actions_ph), 
                                                    reduction_indices=1, keep_dims=True) - self.rewards_ph)
            self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)
            
    def predict(self, state):
        return self.sess.run(self.activation[-1], feed_dict={self.input_ph:state})

    def train(self, state, action, rewards):
        self.sess.run(self.optimizer, feed_dict={self.input_ph:state, self.actions_ph:action, self.rewards_ph:rewards})

class Agent:
    
    def __init__(self, session, net, env, discount=.99, batch=128, min_epsilon=.1, epsilon_decay=.98, max_memory=int(1e5), max_iter=5000, start_training=500):
        
        #machinery
        self.sess = session
        self.nn = net
        self.env = env
        self.N = self.nn.output_size
        
        #params
        self.epsilon = 1
        self.discount = discount
        self.eps_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_memory = max_memory
        self.max_iter = max_iter
        self.start_training = start_training
        self.batch = batch
        
        #database
        self.transitions = []
        self.idx_dict = {
            'state': 0,
            'action': 1,
            'reward': 2,
            'next_state': 3,
            'over': 4
        }
        self.update = self.transitions.append

        
        
    def play(self, training=True, watkins=None):
        self.all_rewards = []
        for i in range(self.max_iter):
            
            state, done = self.env.reset(), False
            total_reward = 0
            while not done:
                
                transition = [0]*5
                act_vec = np.zeros(self.N)
                if i < self.start_training or np.random.uniform() < self.epsilon:
                    random.append(i)
                    act_vec[np.random.randint(self.N)] = 1
                
                else:
                    notrandom.append(i)
                    act_vec[np.argmax( self.nn.predict(state[None]) )] = 1
                    
                transition[0], transition[1] = state, act_vec
                
                state, reward, done, _ = self.env.step(np.argmax(act_vec))
                total_reward += reward
                
                transition[2], transition[3], transition[4] = reward, state, done
                
                self.update(transition)
                if len(self.transitions) > self.max_memory:
                    self.transitions.pop(0)
                
                if i > self.start_training and training:
                    
                    indices = np.random.permutation(len(self.transitions))[:self.batch]
                    
                    current_states = np.asarray([self.transitions[i][self.idx_dict['state']] for i in indices])
                    actions = np.asarray([self.transitions[i][self.idx_dict['action']] for i in indices])
                    rewards = np.asarray([self.transitions[i][self.idx_dict['reward']] for i in indices])
                    next_states = np.asarray([self.transitions[i][self.idx_dict['next_state']] for i in indices])
                    dones = np.asarray([self.transitions[i][self.idx_dict['over']] for i in indices]).astype('int')
                    
                    ns_val = self.nn.predict(next_states)
                    if watkins:
                        cr_val = np.sum(np.multiply( self.nn.predict(current_states), actions), axis=1)
                        ns_val = watkins*(rewards + self.discount*(1-dones)*ns_val.max(axis=1)) + (1-watkins)*cr_val
                    
                    else:
                        ns_val = rewards + self.discount*(1-dones)*ns_val.max(axis=1)
                    
                    self.nn.train(current_states, actions, ns_val[None].T)
                    
            self.all_rewards.append(total_reward)
            if i%25==0:        
                print('episode nr', i, 'reward', total_reward, 'avg', np.mean(self.all_rewards[-25:]), 'bank', len(self.transitions))
                
            self.epsilon = max(self.epsilon*self.eps_decay, self.min_epsilon)
            
                    
                    
def main(envir):
    env = gym.make(envir)
    S, A = env.observation_space.shape[0], env.action_space.n
    print(S, A)
    tf.reset_default_graph()
    session = tf.InteractiveSession()

    _MLP = MLP(session, [200,200], .1e-3, S, A)

    session.run(tf.initialize_all_variables())

    a = Agent(session, _MLP, env, start_training=25, max_iter=1000)

    a.play(watkins=.5)
