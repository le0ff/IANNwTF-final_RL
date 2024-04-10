import gymnasium as gym
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import layers
from collections import deque

#for gpu memory mangament 
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

#DQN-AGENT
class DQN:
    def __init__(self, env_name="LunarLander-v2", replay_buffer_size=100000, learning_rate=0.0001, discount_factor=0.99, architecture=None):
        """
        input_shape: shape of the input 
        num_actions: number of actions
        replay_buffer_size: size of the replay buffer
        learning_rate: learning rate of the optimizer
        discount_factor: discount factor for the bellman equation
        architecture: architecture of the model, if None a default architecture is used
        """
        temp_env = gym.make(env_name)
        self.env_name = env_name
        self.input_shape = temp_env.observation_space.shape
        self.num_actions = temp_env.action_space.n
        # self.input_shape = input_shape
        # self.num_actions = num_actions
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.architecture = architecture
        
        #builds models
        self.model = self.build_model()
        self.target_model = self.build_model()

    #function to create and compile model
    def build_model(self):
        if self.architecture is not None:
            model = tf.keras.models.Sequential(self.architecture)
        else:
            model = tf.keras.models.Sequential([
                layers.Dense(32, activation="relu", input_shape=self.input_shape),
                layers.Dense(32, activation="relu"), 
                layers.Dense(self.num_actions, activation="linear")
            ])
        #compiles model with mean squared error loss and adam optimizer
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    #agent takes action based on epsilon-greedy policy
    def act(self, state, epsilon = 0.0):
        if np.random.rand() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            #memory error with model.predict
            # Q_values = self.model.predict(state[np.newaxis])
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            Q_values = self.model(state[np.newaxis])
            return np.argmax(Q_values[0])
    
    #agent stores experience in replay buffer
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    #agent samples from replay buffer and updates Q-values
    def replay(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        samples = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))


        #base implementation

        #Q-values
        # q_values = self.model(states)
        #next Q-values from target model
        # next_q_values = self.target_model(next_states)
        # max_next_q_values = np.max(next_q_values, axis=1)
        #calculate target Q-values using Bellman equation
        # target_q_values = (rewards + (1 - dones) * self.discount_factor * max_next_q_values)

        #quick fix for TypeError: 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment
        # q_values = q_values.numpy()
        # q_values[range(batch_size), actions] = target_q_values
        # self.model.train_on_batch(states, q_values)
    
        ################################################################################################################################
        
        #implementation with tf.function decorator
        
        #convert numpy arrays to tensors (maybe computationally more efficient?)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int64)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float64)
        dones = tf.convert_to_tensor(dones, dtype=tf.bool)
        
        #maybe more efficient to use a explicit train_step function with tf.function decorator
        
        states, q_values = self.train_step(states, actions, rewards, next_states, dones, batch_size)
        
        #train model
        self.model.train_on_batch(states, q_values)

        ################################################################################################################################


    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones, batch_size=32):

        #calculate q values and next q values
        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        #calculate max next q values
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)

        #calculate target q values using Bellman equation
        target_q_values = rewards + (1 - tf.cast(dones, dtype=tf.float64)) * self.discount_factor * tf.cast(max_next_q_values, dtype=tf.float64)

        #get the indices of the actions
        batch_indices = tf.range(batch_size)
        action_indices = tf.cast(actions, tf.int32)

        #update the q values
        target_q_values_float32 = tf.cast(target_q_values, tf.float32)
        q_values = tf.tensor_scatter_nd_update(q_values, tf.expand_dims(tf.stack([batch_indices, action_indices], axis=1), axis=1), tf.expand_dims(target_q_values_float32, axis=1))
        
        return states, q_values


    #updates target model weights with model weights
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    #TRAINING
    #if saving is set to a path, the model will save every saving_frequency episodes
    def train(self, epsilon_decay=0.99941, num_episodes=5000, batch_size=32, target_update_freq=1, output=100, saving=None, saving_frequency=500, render=False):
        """
        epsilon_decay: decay rate of epsilon
        num_episodes: number of episodes
        batch_size: batch size for replay
        target_update_freq: frequency of updating the target model
        output: number of episodes after which the function will print the episode, episode reward and cumulative reward over last output episodes
        saving: path to save the model
        saving_frequency: frequency of saving the model
        render: if True, the environment will be rendered
        """
        
        #create environment
        if render:
            env = gym.make(self.env_name, render_mode="human")
        else:
            env = gym.make(self.env_name)

        epsilon = 1.0
        min_epsilon = 0.01
        min_reward = -250

        #list to store reward per episode
        reward_history = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render:
                    env.render()

                action = self.act(state, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                self.replay(batch_size)

                #episode ends if reward is too low
                if episode_reward < min_reward:
                    done = True

            if episode % target_update_freq == 0:
                self.update_target_model()

            reward_history.append(episode_reward)
            
            #apply epsilon decay, clip epsilon to min_epsilon
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

            if episode % output == 0:
                #print(f"Episode: {episode} episode reward: {episode_reward} Epsilon: {epsilon}")
                #episode, episode_reward and cumulative reward over the last output episodes
                print(f"Episode: {episode} | Episode reward: {episode_reward} | avg. reward for last {output}: {np.mean(reward_history[-output:])}")

            #save intermediate models
            if saving and episode % saving_frequency == 0:
                self.save(f"{saving}/intermediate_results/episode_{episode}.h5")
                
        env.close()
        return reward_history
    
    #TESTING
    def test(self, num_episodes=20, output=True, render=False):
        """
        num_episodes: number of episodes to test
        output: if True, the function will print the episode and episode reward
        """
        #create environment
        if render:
            env = gym.make(self.env_name, render_mode="human")
        else:
            env = gym.make(self.env_name)

        test_rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render:
                    env.render()
                #get action from model using 0.0 as epsilon to only exploit
                action = self.act(state, 0.0)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                episode_reward += reward

            if output:
                print(f"Test Episode: {episode} | Episode reward: {episode_reward}")

            test_rewards.append(episode_reward)
        
        env.close()
        return test_rewards
    
    #saving model
    #path of the scheme "models/{NAME}/TYPE"
    def save(self, path):
        self.model.save(path)
    
    #loading model
    def load(self, path):
        self.model = tf.keras.models.load_model(f"{path}")
        #update target model
        self.update_target_model()