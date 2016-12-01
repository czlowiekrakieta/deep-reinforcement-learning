import numpy as np
import tensorflow as tf
import gym

class LS: #linear softmax
    def __init__(self, session, state_size, actions_nr, name='policy', lr=.1):
        
        self.sess = session
        self.state_size = state_size
        self.actions = actions_nr
        assert name == 'policy', "liniowy softmax nie ma sensu jako aproksymator wartości"
        #ustawione, żeby nikt się nie pomylił i nie otrzymał bzdurnych wyników
        
        with tf.variable_scope(name):
            
            self.params = tf.get_variable('parameteres', [self.state_size, self.actions])
            self.state_ph = tf.placeholder('float', [None, self.state_size])
            self.advs_ph = tf.placeholder('float', [None, 1])
            self.actions_ph = tf.placeholder('float', [None, self.actions])
            self.probs = tf.nn.softmax(tf.matmul(self.state_ph, self.params))
            
            self.gradient = tf.log(tf.reduce_max(tf.mul(self.probs, self.actions_ph), 1))*self.advs_ph
            
            self.optimizer = tf.train.AdamOptimizer(lr).minimize(-tf.reduce_mean(self.gradient))
        
    def predict(self, state):
        return self.sess.run(self.probs, feed_dict={self.state_ph:state})
    
    def train(self, state, results, actions):
        self.sess.run(self.optimizer, feed_dict={self.state_ph:state, self.advs_ph:results, self.actions_ph:actions})

class MLP:
    
    def __init__(self, session, hidden, name, input_size, output_size, init_means=0, init_stds=0.01, lr=.1): #name of variable scope, indicating policy of value
        
        assert name == 'policy' or name == 'value', 'perceptron może przybliżać strategię albo wartość'
        self.sess = session
        self.name = name
        self.lr = lr
        self.hidden = hidden
        if not isinstance(self.hidden, list):
            self.hidden = [self.hidden]
        self.input_size = input_size
        self.output_size = output_size
        if self.name == 'value':
            self.output_size = 1
        self.architecture = [self.input_size] + self.hidden + [self.output_size]
        N = len(self.architecture)
        
        with tf.variable_scope(self.name):
            self.input_ph = tf.placeholder('float', [None, self.input_size])
            
            self.weights = [(tf.get_variable('W' + str(i), [self.architecture[i], self.architecture[i+1]], 
                            initializer=tf.random_normal_initializer(mean=init_means, stddev=init_stds)), 
                             tf.get_variable('B' + str(i), [self.architecture[i+1]], 
                            initializer=tf.constant_initializer(init_means))) for i in range(N-1)]
            
            self.activation = [tf.nn.relu(tf.matmul(self.input_ph, self.weights[0][0])) + self.weights[0][1]]
            
            for i in range(1,N-2):
                
                self.activation.append(tf.nn.relu(tf.matmul(self.activation[-1], self.weights[i][0])+self.weights[i][1]))
            
            self.activation.append(tf.nn.softmax(tf.matmul(self.activation[-1], self.weights[-1][0])+self.weights[-1][1]))
            
            if self.name == 'value':
                self.rewards = tf.placeholder('float', [None, 1])
                self.val_optimizer = tf.train.AdamOptimizer(self.lr).minimize(tf.nn.l2_loss(self.activation[-1] - self.rewards))
                
            if self.name == 'policy':
                self.actions = tf.placeholder('float', [None, self.output_size])
                self.advantages = tf.placeholder('float', [None, 1])
                
                self.probs = tf.nn.softmax(self.activation[-1])
                self.gradient = tf.log(tf.reduce_sum(tf.mul(probs, self.actions), reduction_indices=1, keep_dims=True))*self.advantages
                
                self.pol_optimizer = tf.train.AdamOptimizer(self.lr).minimize(-tf.reduce_mean(self.gradient))
                
    def predict(self, state):
        return self.sess.run(self.activation[-1], feed_dict={self.input_ph:state})
    
    def train(self, state, results, actions=None):
        #if value, results are just rewards from actions
        if self.name == 'value':
            self.sess.run(self.val_optimizer, feed_dict={self.input_ph:state, self.rewards:results})
            
        #if policy, results are advantages, which are yet to be combined with log probs
        if self.name == 'policy':
            
            assert actions is not None, 'musisz podać zerojedynkową macierz akcji'
            assert actions.shape == (state.shape[0], self.output_size)
            
            self.sess.run(self.pol_optimizer, feed_dict={self.input_ph:state, 
                                                         self.advantages:results, self.actions:actions})
            

class PGAgent:
    
    def __init__(self, env, discount, policy, value, random_play=1000, start_epsilon=1, min_epsilon=.1, epsilon_decay=.99,ifbaseline=True, ifdiscount=True):
        
        self.env = env
        self.N = env.action_space.n
        
        #params
        self.discount = discount
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.eps_decay = epsilon_decay
        self.random_play = random_play
        
        #way of learning
        self.ifbaseline = ifbaseline
        self.ifdiscount = ifdiscount
        
        #approximators
        self.policy = policy
        self.value = value
        
            
    def episode(self, global_, iters=300, training=True):
        state = self.env.reset()
        states = []
        actions = []
        rewards = []
        
        sapp = states.append
        aapp = actions.append
        rapp = rewards.append
        
        for i in range(iters):
        
            act_vec = np.zeros(self.N)
            if global_ < self.random_play or np.random.uniform() < self.epsilon:
                act_vec[np.random.randint(self.N)] = 1
                
            else:
                act_vec[np.argmax(self.policy.predict(state[None]))] = 1
            sapp(state)
            aapp(act_vec)
            state, reward, done, _ = self.env.step(np.argmax(act_vec))
            rapp(reward)
            if done:
                break
             
        if training:       
            rewards = np.asarray(rewards)

            L = len(states)
            future_rewards = rewards.copy()
            if self.ifdiscount:
                for j in range(L):

                    future_reward = np.sum(np.power(self.discount, np.arange(L-j) )*rewards[j:])
                    future_rewards[j] = future_reward

            if self.ifbaseline:
                self.values = self.value.predict(np.asarray(states))
            else:
                self.values = np.zeros(( L,1))
            advs = (future_rewards[None].T - self.values)
            self.value.train(np.asarray(states), future_rewards[None].T)
            self.policy.train(np.asarray(states), advs, np.asarray(actions))
            
        return np.sum(rewards)
        
    def play(self, session, training, max_iter=10000):
        self.totals = []
        for i in range(max_iter):
            
            self.epsilon *= self.eps_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)
            R = self.episode(global_=i)
            self.totals.append(R)
            if R >= 200:
                print('solved', i, R)
            if i%100==0:
                print('episode', i, 'reward', R, 'time', time.time() - t, 'avg', np.mean(self.totals[-100:]))
                

def main(envir):
    env = gym.make(envir)
    tf.reset_default_graph()

    session = tf.InteractiveSession()
    S, A = env.observation_space.shape[0], env.action_space.n

    _LS = LS(session, state_size=S,actions_nr=A)
    _MLP = MLP(session, [10], 'value', input_size=S, output_size=1)

    session.run(tf.initialize_all_variables())

    a = PGAgent(env, discount=.99, policy=_LS, value=_MLP, ifbaseline=False, ifdiscount=False)
    a.play(session,training=True, max_iter=1000000)
