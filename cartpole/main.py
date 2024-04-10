import sys
sys.path.append("..")
from DQN import DQN
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#training for one model
def training(name="intial_model", render=False):
    #print start time
    print(time.strftime("%H:%M:%S", time.localtime()))

    PATH = f"models/{name}"
    dqn = DQN("CartPole-v0", REPLAY_BUFFER_SIZE, LEARNING_RATE, DISCOUNT_FACTOR, ARCHITECTURE)
    train_reward = dqn.train(epsilon_decay=EPSILON_DECAY, num_episodes=NUM_EPISODES, batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ, output=10, saving=PATH, render=render)
    dqn.save(f"{PATH}/final_model.h5")
    np.save(f"{PATH}/train_rewards.npy", train_reward)
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
def training_models(render=False):
    for batch_size in BATCH_SIZES:
        for target_update_freq in TARGET_UPDATE_FREQS:
            PATH = f"models/batch_size_{batch_size}/target_update_freq_{target_update_freq}"
            print(f"Model {PATH}")
            #print start time for each model
            print(time.strftime("%H:%M:%S", time.localtime()))

            dqn = DQN("CartPole-v0", REPLAY_BUFFER_SIZE, LEARNING_RATE, DISCOUNT_FACTOR, ARCHITECTURE)
            train_reward = dqn.train(epsilon_decay=EPSILON_DECAY, num_episodes=NUM_EPISODES, batch_size=batch_size, target_update_freq=target_update_freq, output=10, saving=PATH, render=render)
            dqn.save(f"{PATH}/final_model.h5")
            np.save(f"{PATH}/train_rewards.npy", train_reward)
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

#testing final and each intermediate save of the model and saving the results
def testing(name="intial_model", intermediate=False, render=False):
    if intermediate:
        PATH = f"models/{name}/intermediate_results"
        for episode in range(100, NUM_EPISODES, 100):
            print(f"Model {PATH}/episode_{episode}")
            dqn = DQN("CartPole-v0", REPLAY_BUFFER_SIZE, LEARNING_RATE, DISCOUNT_FACTOR, ARCHITECTURE)
            dqn.load(f"{PATH}/episode_{episode}.h5")
            test_rewards = dqn.test(num_episodes=100, output=False, render=render)
            np.save(f"{PATH}/test_rewards_episode_{episode}.npy", test_rewards)
            
            #print average test reward
            print(f"Average test reward: {np.mean(test_rewards)}")
            
            #release memory
            del dqn
            tf.keras.backend.clear_session()
    
    PATH = f"models/{name}"
    print(f"Model {PATH}")
    dqn = DQN("CartPole-v0", REPLAY_BUFFER_SIZE, LEARNING_RATE, DISCOUNT_FACTOR, ARCHITECTURE)
    dqn.load(f"{PATH}/final_model.h5")
    test_rewards = dqn.test(num_episodes=100, output=True, render=render)
    np.save(f"{PATH}/test_rewards.npy", test_rewards)

    #print average test reward
    print(f"Average test reward: {np.mean(test_rewards)}")

#testing each intermediate save of the models and saving results in npy files
def testing_models(intermediate=False, render=False):
    for batch_size in BATCH_SIZES:
        for target_update_freq in TARGET_UPDATE_FREQS:
            PATH = f"models/batch_size_{batch_size}/target_update_freq_{target_update_freq}"
            if intermediate:
                for episode in range(100, NUM_EPISODES, 100):
                    print(f"Model {PATH}/intermediate_results/episode_{episode}")
                    dqn = DQN("CartPole-v0", REPLAY_BUFFER_SIZE, LEARNING_RATE, DISCOUNT_FACTOR, ARCHITECTURE)
                    dqn.load(f"{PATH}/intermediate_results/episode_{episode}.h5")
                    test_rewards = dqn.test(num_episodes=100, output=True, render=render)
                    np.save(f"{PATH}/intermediate_results/test_rewards_episode_{episode}.npy", test_rewards)
                
                    #print average test reward
                    print(f"Average test reward: {np.mean(test_rewards)}")

                    #release memory
                    del dqn
                    tf.keras.backend.clear_session()
            
            print(f"Model {PATH}")
            dqn = DQN("CartPole-v0", REPLAY_BUFFER_SIZE, LEARNING_RATE, DISCOUNT_FACTOR, ARCHITECTURE)
            dqn.load(f"{PATH}/final_model.h5")
            test_rewards = dqn.test(num_episodes=100, output=True, render=render)
            np.save(f"{PATH}/test_rewards.npy", test_rewards)

            #print average test reward
            print(f"Average test reward: {np.mean(test_rewards)}")
                
            #release memory
            del dqn
            tf.keras.backend.clear_session()

################################################################################################################################

#disable interactive logging
keras.utils.disable_interactive_logging()

#MODEL
REPLAY_BUFFER_SIZE = 20000
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
ARCHITECTURE = [
    layers.Dense(24, activation="relu", input_shape=(4,)),
    layers.Dense(24, activation="relu"),
    layers.Dense(2, activation="linear")
    ]

#HYPERPARAMETERS
EPSILON_DECAY = 0.995
NUM_EPISODES = 500
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 1
#different batch sizes and target update frequencies for comparison 
BATCH_SIZES = [32, 64]
TARGET_UPDATE_FREQS = [1, 2, 4, 6, 8]

#training and testing
# training(name="PLACEHOLDER_NAME", render=False)
# training_models(render=False)
# testing(name="PLACEHOLDER_NAME", intermediate=False, render=True)
testing_models(intermediate=False, render=True)