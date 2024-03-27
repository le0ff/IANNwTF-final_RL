import gymnasium as gym
import numpy as np
import tensorflow as tf
import random, time
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import matplotlib.pyplot as plt

#against memory leak (unsure about the effectiveness)
from tensorflow.compat.v1.keras.backend import set_session
#tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

#DQN-AGENT
class DQN:
    def __init__(self, input_shape, num_actions, replay_buffer_size=20000, learning_rate=0.001, discount_factor=0.99, architecture=None):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = deque(maxlen=self.replay_buffer_size) #a high enough replay-buffer size is needed to prevent catastrophic forgetting
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
                layers.Dense(24, activation="relu", input_shape=self.input_shape), #32 units
                layers.Dense(24, activation="relu"), #32 units
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
            Q_values = self.model(state[np.newaxis])
            #Q_values = self.model.predict(state)
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
        
        #Q-values
        q_values = self.model(states)
        #next Q-values from target model
        next_q_values = self.target_model(next_states)
        max_next_q_values = np.max(next_q_values, axis=1)
        #calculate target Q-values using Bellman equation
        target_q_values = (rewards + (1 - dones) * self.discount_factor * max_next_q_values)

        # #quick fix for TypeError: 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment
        q_values = q_values.numpy()
        q_values[range(batch_size), actions] = target_q_values
        self.model.train_on_batch(states, q_values)

    #updates target model weights with model weights
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    #TRAINING
    #if saving is set to a path, the model will save every saving_frequency episodes
    def train(self, epsilon_decay=0.995, num_episodes=500, batch_size=32, target_update_freq=10, output=False, saving=None, saving_frequency=100):
        env = gym.make('CartPole-v0')

        epsilon = 1.0
        min_epsilon = 0.01
        target_update_counter = 0

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
            target_update_counter += 1
            if target_update_counter % target_update_freq == 0:
                self.update_target_model()
            running_reward.append(episode_reward)
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

            if output and episode % 10 == 0:
                print(f"Episode: {episode} episode reward: {episode_reward} Epsilon: {epsilon}")
            
            if saving and episode % saving_frequency == 0:
                self.save(f"{saving}/intermediate_results/episode_{episode}.h5")

        env.close()
        return running_reward
    
    #TESTING
    def test(self, num_episodes=20, output=False):
        env = gym.make('CartPole-v0')

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
        #loading target model with equal weights
        self.target_model = tf.keras.models.load_model(f"{path}")


#disable interactive logging
keras.utils.disable_interactive_logging()

INPUT_SHAPE = (4,)
NUM_ACTIONS = 2

#MODEL
REPLAY_BUFFER_SIZE = 20000
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
ARCHITECTURE = None

#HYPERPARAMETERS
EPSILON_DECAY = 0.995
BATCH_SIZE = [32, 64]
NUM_EPISODES = 500
TARGET_UPDATE_FREQ = [1, 2, 4, 6, 8]

################################################################################################################################

#loop for different models:

# for batch_size in BATCH_SIZE:
#     for target_update_freq in TARGET_UPDATE_FREQ:
#         path = f"models/batch_size_{batch_size}/target_update_freq_{target_update_freq}"

#         dqn = DQN(INPUT_SHAPE, NUM_ACTIONS, REPLAY_BUFFER_SIZE, LEARNING_RATE, DISCOUNT_FACTOR, ARCHITECTURE)
#         train_reward = dqn.train(EPSILON_DECAY, NUM_EPISODES, batch_size, target_update_freq, output=True, saving=path)
#         test_rewards = dqn.test(output=True)
#         dqn.save(f"{path}/weights.h5")
#         #further saves for the plots
#         np.save(f"{path}/train_reward.npy", train_reward)
#         np.save(f"{path}/test_rewards.npy", test_rewards)

#         #saving model info
#         with open(f"{path}/info.txt", "w") as f:
#             #write model summary and other details to file
#             dqn.model.summary(print_fn=lambda x: f.write(x + '\n'))
#             f.write(f"Replay buffer length: {REPLAY_BUFFER_SIZE}\n")
#             f.write(f"Learning rate: {LEARNING_RATE}\n")
#             f.write(f"Discount factor: {DISCOUNT_FACTOR}\n")
#             f.write(f"Batch size: {batch_size}\n")
#             f.write(f"Epsilon decay: {EPSILON_DECAY}\n")
#             f.write(f"Target update frequency: {target_update_freq}\n")
#             f.write(f"Episodes: {NUM_EPISODES}\n")
        
#         #release memory
#         del dqn
#         tf.keras.backend.clear_session()

# #print time
# print(time.strftime("%H:%M:%S", time.localtime()))

################################################################################################################################

#plotting

#loading data and plot training and test rewards per model in one plot, and all the models in a grid (5 x 2) of plots
# fig, axs = plt.subplots(5, 2, figsize=(20, 20))
# for j, batch_size in enumerate(BATCH_SIZE):
#     for i, target_update_freq in enumerate(TARGET_UPDATE_FREQ):
#         path = f"models/batch_size_{batch_size}/target_update_freq_{target_update_freq}"
#         train_reward = np.load(f"{path}/train_reward.npy")
#         test_rewards = np.load(f"{path}/test_rewards.npy")
#         axs[i, j].plot(train_reward, label="Train reward")
#         axs[i, j].plot(test_rewards, label="Test reward")
#         axs[i, j].set_title(f"Batch size: {batch_size}, Target update frequency: {target_update_freq}")
#         axs[i, j].legend()
#         axs[i, j].set_xlabel("Episodes")
#         axs[i, j].set_ylabel("Reward")
# plt.show()

################################################################################################################################

#testing each intermediate save of the model with for loop and saving the results

for batch_size in BATCH_SIZE:
    for target_update_freq in TARGET_UPDATE_FREQ:
        path = f"models/batch_size_{batch_size}/target_update_freq_{target_update_freq}/intermediate_results"
        for episode in range(100, NUM_EPISODES, 100):
            print(f"Model {path}/episode_{episode}.h5")
            dqn = DQN(INPUT_SHAPE, NUM_ACTIONS, REPLAY_BUFFER_SIZE, LEARNING_RATE, DISCOUNT_FACTOR, ARCHITECTURE)
            dqn.load(f"{path}/episode_{episode}.h5")
            test_rewards = dqn.test(num_episodes=100, output=False)
            np.save(f"{path}/test_rewards_episode_{episode}.npy", test_rewards)
            
            #print average test reward
            print(f"Average test reward: {np.mean(test_rewards)}")
            
            #release memory
            del dqn
            tf.keras.backend.clear_session()

################################################################################################################################