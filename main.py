# Imports:
# --------
import numpy as np
from env import create_env
from Q_learning import train_q_learning, visualize_q_table

# User definitions:
# -----------------
train = True
visualize_results = True

random_initialization = True  # If True, the Q-table will be initialized randomly

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate

# 
# 
# at episode 15000 it reaches epsilon 0.1
# 
# 

epsilon_decay = 0.9998465  # Decay rate for exploration
no_episodes = 20_000  # Number of episodes

# learning_rate = 0.01  # Learning rate
# gamma = 0.99  # Discount factor
# epsilon = 1.0  # Exploration rate
# epsilon_min = 0.1  # Minimum exploration rate
# epsilon_decay = 0.995  # Decay rate for exploration
# no_episodes = 1_000  # Number of episodes

goal_coordinates = np.array([1, 6])

# Define all hell state coordinates as a tuple within a list
hell_state_coordinates = [np.array([2, 6]), np.array([2, 10]), np.array([4, 1]), np.array([4, 9]), 
                          np.array([8, 4]), np.array([9, 7]), np.array([12, 7]), np.array([14, 5])]


# Execute:
# --------
if train == True:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env(random_initialization=random_initialization)
    print("main: iniciando processo")
    # Train a Q-learning agent:
    # -------------------------

    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma,
                     train = train)
    
if train == False:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env(random_initialization = False)
    print("main: iniciando processo")
    # Train a Q-learning agent:
    # -------------------------

    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon = 0,
                     epsilon_min = 0,
                     epsilon_decay = 0,
                     alpha=learning_rate,
                     gamma=gamma,
                     train = train)

if visualize_results:
    # Visualize the Q-table:
    # ----------------------
    visualize_q_table(goal_coordinates=goal_coordinates,
                      q_values_path="q_table.npy",
                      wall_states=env.wall_states)


"""
TODO:
1. plot a graph that is total reward vs episode number
2. train without random initialization - store the q values - run it wuth random initialization
3. if random initialization -- don't update the q table
"""