import gymnasium as gym
import numpy as np
import tensorflow as tf
import random, time
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import matplotlib.pyplot as plt

#for gpu memory mangament
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

#DQN-AGENT
class DQN:
    def __init__(self, input_shape, num_actions, replay_buffer_size=100000, learning_rate=0.0001, discount_factor=0.99, architecture=None):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.architecture = architecture
        
        
        self.model = self.build_model()
        self.target_model = self.build_model()

    # build model
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
            #Q_values = self.model.predict(state[np.newaxis])
            #convert state to tensor
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

        # #Q-values
        # q_values = self.model(states)
        # #next Q-values from target model
        # next_q_values = self.target_model(next_states)
        # max_next_q_values = np.max(next_q_values, axis=1)
        # #calculate target Q-values using Bellman equation
        # target_q_values = (rewards + (1 - dones) * self.discount_factor * max_next_q_values)

        # # #quick fix for TypeError: 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment
        # q_values = q_values.numpy()
        # q_values[range(batch_size), actions] = target_q_values
        # self.model.train_on_batch(states, q_values)
    
        ################################################################################################################################
        
        #alternative implementation with tf.function decorator
        
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

        #calculate target q values using bellman equation
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
    def train(self, epsilon_decay=0.99941, num_episodes=5000, batch_size=32, target_update_freq=1, output=False, saving=None, saving_frequency=500):
        env = gym.make("LunarLander-v2")

        epsilon = 1.0
        min_epsilon = 0.01
        target_update_counter = 0
        min_reward = -250

        running_reward = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                #env.render()
                action = self.act(state, epsilon)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                self.replay(batch_size)

                if episode_reward < min_reward:
                    done = True

            target_update_counter += 1
            if target_update_counter % target_update_freq == 0:
                self.update_target_model()

            running_reward.append(episode_reward)
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

            if output and episode % 100 == 0:
                #print(f"Episode: {episode} episode reward: {episode_reward} Epsilon: {epsilon}")
                #episode, episode_reward and cumulative reward over the last 100 episodes
                print(f"Episode: {episode} episode reward: {episode_reward} cumulative reward: {np.mean(running_reward[-100:])}")

            if saving and episode % saving_frequency == 0:
                self.save(f"{saving}/intermediate_results/episode_{episode}")
                

        env.close()
        return running_reward
    
    #TESTING
    def test(self, num_episodes=20, output=False):
        env = gym.make("LunarLander-v2")

        test_rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                #env.render()
                action = self.act(state, 0.0)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                episode_reward += reward

            if output:
                print(f"Test Episode: {episode} episode reward: {episode_reward}")

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



#disable interactive logging
keras.utils.disable_interactive_logging()

INPUT_SHAPE = (8,)
NUM_ACTIONS = 4

#MODEL
REPLAY_BUFFER_SIZE = 250000
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.99
ARCHITECTURE = None

#HYPERPARAMETERS
EPSILON_DECAY = 0.99941
BATCH_SIZE = [32, 64]
NUM_EPISODES = 5000
TARGET_UPDATE_FREQ = [1, 4, 8, 16, 32]

################################################################################################################################

#training for one model
def training(name="intial_model"):
    #print start time
    print(time.strftime("%H:%M:%S", time.localtime()))

    PATH = f"models/{name}"
    dqn = DQN(INPUT_SHAPE, NUM_ACTIONS, REPLAY_BUFFER_SIZE, LEARNING_RATE, DISCOUNT_FACTOR, ARCHITECTURE)
    train_reward = dqn.train(epsilon_decay=EPSILON_DECAY, num_episodes=NUM_EPISODES, batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ, output=True, saving=PATH)
    dqn.save(f"{PATH}/final_model")
    np.save(f"{PATH}/train_reward.npy", train_reward)
    test_rewards = dqn.test(num_episodes=100, output=True)
    np.save(f"{PATH}/test_rewards.npy", test_rewards)

    #saving further information
    with open(f"{PATH}/info.txt", "w") as file:
        #model summary
        dqn.model.summary(print_fn=lambda x: file.write(x + '\n'))
        file.write(f"Replay buffer size: {REPLAY_BUFFER_SIZE}\n")
        file.write(f"Learning rate: {LEARNING_RATE}\n")
        file.write(f"Discount factor: {DISCOUNT_FACTOR}\n")
        file.write(f"Epsilon decay: {EPSILON_DECAY}\n")
        file.write(f"Batch size: {BATCH_SIZE}\n")
        file.write(f"Number of episodes: {NUM_EPISODES}\n")
        file.write(f"Target update frequency: {TARGET_UPDATE_FREQ}\n")

    # #print time
    print(time.strftime("%H:%M:%S", time.localtime()))

#training and saving the models, here for different batch sizes and target update frequencies
def training_models():
    for batch_size in BATCH_SIZE:
        for target_update_freq in TARGET_UPDATE_FREQ:
            PATH = f"models/batch_size_{batch_size}/target_update_freq_{target_update_freq}"
            print(f"Model {PATH}")
            #print start time for each model
            print(time.strftime("%H:%M:%S", time.localtime()))

            dqn = DQN(INPUT_SHAPE, NUM_ACTIONS, REPLAY_BUFFER_SIZE, LEARNING_RATE, DISCOUNT_FACTOR, ARCHITECTURE)
            train_reward = dqn.train(epsilon_decay=EPSILON_DECAY, num_episodes=NUM_EPISODES, batch_size=batch_size, target_update_freq=target_update_freq, output=True, saving=PATH)
            dqn.save(f"{PATH}/final_model")
            np.save(f"{PATH}/train_reward.npy", train_reward)
            test_rewards = dqn.test(num_episodes=100, output=True)
            np.save(f"{PATH}/test_rewards.npy", test_rewards)

            #saving further information
            with open(f"{PATH}/info.txt", "w") as file:
                #model summary
                dqn.model.summary(print_fn=lambda x: file.write(x + '\n'))
                file.write(f"Replay buffer size: {REPLAY_BUFFER_SIZE}\n")
                file.write(f"Learning rate: {LEARNING_RATE}\n")
                file.write(f"Discount factor: {DISCOUNT_FACTOR}\n")
                file.write(f"Epsilon decay: {EPSILON_DECAY}\n")
                file.write(f"Batch size: {batch_size}\n")
                file.write(f"Number of episodes: {NUM_EPISODES}\n")
                file.write(f"Target update frequency: {target_update_freq}\n")
            
            #release memory
            del dqn
            tf.keras.backend.clear_session()

            #print time
            print(time.strftime("%H:%M:%S", time.localtime()))


################################################################################################################################

#testing each intermediate save of the model and saving the results
def testing(name="intial_model"):
    PATH = f"models/{name}/intermediate_results"
    for episode in range(500, NUM_EPISODES, 500):
        print(f"Model {PATH}/episode_{episode}")
        dqn = DQN(INPUT_SHAPE, NUM_ACTIONS, REPLAY_BUFFER_SIZE, LEARNING_RATE, DISCOUNT_FACTOR, ARCHITECTURE)
        dqn.load(f"{PATH}/episode_{episode}")
        test_rewards = dqn.test(num_episodes=100, output=False)
        np.save(f"{PATH}/test_rewards_episode_{episode}.npy", test_rewards)
        
        #print average test reward
        print(f"Average test reward: {np.mean(test_rewards)}")
        
        #release memory
        del dqn
        tf.keras.backend.clear_session()

#testing each intermediate save of the models and saving results in npy files
def testing_models():
    for batch_size in BATCH_SIZE:
        for target_update_freq in TARGET_UPDATE_FREQ:
            PATH = f"models/batch_size_{batch_size}/target_update_freq_{target_update_freq}/intermediate_results"
            for episode in range(500, NUM_EPISODES, 500):
                print(f"Model {PATH}/episode_{episode}")
                dqn = DQN(INPUT_SHAPE, NUM_ACTIONS, REPLAY_BUFFER_SIZE, LEARNING_RATE, DISCOUNT_FACTOR, ARCHITECTURE)
                dqn.load(f"{PATH}/episode_{episode}")
                test_rewards = dqn.test(num_episodes=100, output=False)
                np.save(f"{PATH}/test_rewards_episode_{episode}.npy", test_rewards)
                
                #print average test reward
                print(f"Average test reward: {np.mean(test_rewards)}")
                
                #release memory
                del dqn
                tf.keras.backend.clear_session()

################################################################################################################################

#plotting
# PATH = f"models/big_replay_buffer"
# train_reward = np.load(f"{PATH}/train_reward.npy")
# test_rewards = np.load(f"{PATH}/test_rewards.npy")

# plt.plot(train_reward)
# plt.title("Training Reward")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# #save plot
# plt.savefig(f"{PATH}/train_reward.png")

# plt.show()

################################################################################################################################