# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Function 1: Train Q-learning agent
# -----------
rewards = []
episodes = []

def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy",
                     train = None):

    # Initialize the Q-table:
    # -----------------------
    if train == True:
        q_table = np.zeros((env.y_size, env.x_size, env.action_space.n))
    if train == False:
        q_table = np.load("q_table.npy")
    # Q-learning algorithm:
    # ---------------------
    #! Step 1: Run the algorithm for fixed number of episodes
    #! -------
    for episode in range(no_episodes):
        state, _ = env.reset()
        state = tuple(state)
        total_reward = 0

        #! Step 2: Take actions in the environment until "Done" flag is triggered
        #! -------
        while True:
            #! Step 3: Define your Exploration vs. Exploitation
            #! -------
        #     print("vou andar no train_q_learning!")
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)
            # env.render()

            next_state = tuple(next_state)
            total_reward += reward

            #! Step 4: Update the Q-values using the Q-value update rule
            #! -------
            q_table[state][action] = q_table[state][action] + alpha * \
                (reward + gamma *
                 np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            #! Step 5: Stop the episode if the agent reaches Goal or Hell-states
            #! -------
            if done:
                # print("NOVO EP SAIU GALERA UAU")
                break

        #! Step 6: Perform epsilon decay
        #! -------
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        rewards.append(total_reward)
        episodes.append(episode)

    #! Step 7: Close the environment window
    #! -------
    env.close()
    print("Training finished.\n")

    #! Step 8: Save the trained Q-table
    #! -------
    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")
    plt.plot(episodes, rewards)
    plt.title("Reward vs Epoch")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.show()

# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(hell_state_coordinates = [(2,6), (2, 10), (4, 1), (4, 9), (8, 4), (9, 7), (12, 7), (14, 5)],
                      goal_coordinates=tuple([1,6]),
                      wall_states = [],
                      actions=["Up", "Down", "Right", "Left"],
                      q_values_path="q_table.npy"):



    # Load the Q-table:
    # -----------------
    try:
        q_table = np.load(q_values_path)

        # Create subplots for each action:
        # --------------------------------
        _, axes = plt.subplots(2, 2, figsize=(10,10))
        axes = axes.flatten() # In order to plot 2x2
        
        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            # Mask the goal state's Q-value for visualization:
            # ------------------------------------------------
            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[goal_coordinates] = False
            mask[hell_state_coordinates[0]] = True
            mask[hell_state_coordinates[1]] = True
            mask[hell_state_coordinates[2]] = True
            mask[hell_state_coordinates[3]] = True
            mask[hell_state_coordinates[4]] = True
            mask[hell_state_coordinates[5]] = True
            mask[hell_state_coordinates[6]] = True
            mask[hell_state_coordinates[7]] = True

            for t in wall_states:
                mask[tuple(t)] = True

            # Create a new figure for each action
            
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9}) #"size": 9

            # Annotate Goal and Hell states
            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='Black',
                    ha='center', va='center', weight='bold', fontsize=14)
            ax.text(hell_state_coordinates[0][1] + 0.5, hell_state_coordinates[0][0] + 0.5, 'H', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)
            ax.text(hell_state_coordinates[1][1] + 0.5, hell_state_coordinates[1][0] + 0.5, 'H', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)
            ax.text(hell_state_coordinates[2][1] + 0.5, hell_state_coordinates[2][0] + 0.5, 'H', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)
            ax.text(hell_state_coordinates[3][1] + 0.5, hell_state_coordinates[3][0] + 0.5, 'H', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)
            ax.text(hell_state_coordinates[4][1] + 0.5, hell_state_coordinates[4][0] + 0.5, 'H', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)
            ax.text(hell_state_coordinates[5][1] + 0.5, hell_state_coordinates[5][0] + 0.5, 'H', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)
            ax.text(hell_state_coordinates[6][1] + 0.5, hell_state_coordinates[6][0] + 0.5, 'H', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)
            ax.text(hell_state_coordinates[7][1] + 0.5, hell_state_coordinates[7][0] + 0.5, 'H', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f'Action: {action}')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")
