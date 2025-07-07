import numpy as np
import gymnasium as gym
import pygame
import sys

board = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0, -1], 
                      [-1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0, -1], 
                      [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1], 
                      [-1,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0, -1], 
                      [-1, -1, -1, -1,  0,  0,  0, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1,  0,  0,  0, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1],
                      [-1, -1, -1,  0,  1,  0,  0,  0,  0,  0, -1, -1, -1],
                      [-1, -1, -1,  0,  0,  0,  0,  1,  0,  0, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1],
                      [-1, -1, -1, -1,  0,  0,  0,  1,  0,  0, -1, -1, -1],
                      [-1, -1, -1, -1,  0,  0,  0,  0,  0,  0, -1, -1, -1],
                      [-1, -1, -1, -1,  0,  1,  2,  0,  0, -1, -1, -1, -1],
                      [-1, -1, -1, -1,  0,  0,  0,  0,  0, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]


board_np = np.array(board)
wall_states = np.argwhere(board_np == -1)
# print(wall_states)

walles = [tuple(t) for t in wall_states]
print(walles)

# for t in wall_states:
#     print("miau eh isso ai", wall_states[t][0])

    
# q_table = np.load("q_table.npy")


# def calc_decay(epsilon, epsilon_min, epsilon_decay):
#     n = 0
#     while epsilon > epsilon_min:
#         epsilon = epsilon*epsilon_decay
#         n += 1
#     return n