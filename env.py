# Import section:
# Add a small reward for everytime he passes through the neck
import gymnasium as gym
import numpy as np
import pygame
import sys

class SootopolisGym(gym.Env):
    def __init__(self, random_initialization=False):

        super(SootopolisGym, self).__init__()

        self.board = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
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
                      [-1, -1, -1, -1,  0,  1,  2,  0,  0,  0, -1, -1, -1],
                      [-1, -1, -1, -1,  0,  0,  0,  0,  0,  0, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]


        self.agent_states = None

        """
        Rock
        States
        """
        self.rock1 = np.array([2, 6])
        self.rock2 = np.array([2, 10])
        self.rock3 = np.array([4, 2])
        self.rock4 = np.array([4, 9])
        self.rock5 = np.array([8, 4])
        self.rock6 = np.array([9, 7])
        self.rock7 = np.array([12, 7])
        self.rock8 = np.array([14, 5])

        """
        Wall States
        """
                
        self.board_np = np.array(self.board)
        self.wall_states = np.argwhere(self.board_np == -1)

        """
        Reward
        States
        """
        self.small_reward1 = np.array([4, 5])
        self.small_reward2 = np.array([9, 8])
        self.goal_state = np.array([1, 6])

        """
        Drawing
        Board
        Size
        """
        self.x_size = np.size(self.board, 1)
        self.y_size = np.size(self.board, 0)
        self.grid_size = max([self.x_size, self.y_size])

        """
        Defining
        Environment
        Parameters
        """
        self.random_initialization = random_initialization
        self.info = {}
        self.action_space = gym.spaces.Discrete(4) # the spaces are discrete and 4 actions are possible
        self.observation_space = gym.spaces.Box(low = 0, high = self.y_size, shape=(2,), dtype=np.int32)
        self.reward = 0

        """
        Initializing
        Pygame
        """
        
        pygame.init()
        self.tile_size = 32

        self.width = self.x_size * self.tile_size
        self.height = self.y_size * self.tile_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("PADM Initiation Test")
        print("env: iniciei a classe!")

    def reset(self):
        """
        Everything must be reset
        """
        if self.random_initialization:

            board_np = np.array(self.board)
            random_start = [np.argwhere(board_np == 0)] # positions where board equals 0
            random_start = random_start[0] # because it becomes a list of a list
            random = np.random.choice(len(random_start)) # pick a random position for random initialization
            self.agent_state = [random_start[random]] # place agent in the board
            self.agent_state = self.agent_state[0]

        else:
            self.agent_state = np.array([14, 6]) # or just use the default starting position
        
        self.done = False
        self.reward = 0
        
        self.info["Distance to goal"] = self.goal_state - self.agent_state # just simple math for distance

        return self.agent_state, self.info # info to be used in Q_learning when episode ends

    def step(self, action): # it moves the agent and rewards the goal

        new_state = np.array(self.agent_state.copy())
        if action == 0:  # up
            # print("env: andei pra cima")
            new_state[1] += 1
            self.reward = -0.1

        elif action == 1:  # down
            # print("env: andei pra baixo")
            new_state[1] -= 1
            self.reward = -0.1

        elif action == 2:  # left
            # print("env: andei pra esquerda")
            new_state[0] -= 1
            self.reward = -0.1

        elif action == 3:  # right
            # print("env: andei pra direita")
            new_state[0] += 1
            self.reward = -0.1

        """
        Here no punishment for walking was implemented, because too much states - it ended up in very very negative rewards (around -500)
        """

        # Check bounds and walls
        if (0 <= new_state[0] < self.y_size and 0 <= new_state[1] < self.x_size and self.board[new_state[0]][new_state[1]] == -1):
            # stop bumping your head against the wall
            self.reward = -0.1

        else:
            self.agent_state = new_state
        
        
        # Checking the agent's new state and granting rewards:
        # ----------------------

        if any((self.agent_state == rock).all() for rock in [self.rock1, self.rock2, self.rock3, self.rock4, self.rock5, self.rock6, self.rock7, self.rock8]):
            # print("Opa, resetei! ", self.agent_state)
            self.reward = -10
            self.done = True
            # print("env: RESETEI PORRA CARALHO CU MIJO BOSTA")
            # self.reset()

        if any((self.agent_state == small_reward).all() for small_reward in [self.small_reward1, self.small_reward2]):
            # print("small reward! ", self.agent_state)
            # print("env: achei small reward!")
            """
            This part of the code was present in assignment 1. but no small reward was needed for Q learning.
            """
            self.reward = 0 #0.1
            
        elif np.array_equal(self.agent_state, self.goal_state) == True:
            # print("env: GOAL GOAL GOAL\nGOAL GOAL GOAL\nGOAL GOAL GOAL\nGOAL GOAL GOAL\n")
            self.reward = 10
            self.done = True
        
        self.info = {"Distance to goal: ": self.goal_state - self.agent_state} # can be np.linalg.norm(self.goal_state, self.agent_state) --> calculates the modulus of the vector
        
        return self.agent_state, self.reward, self.done, self.info # important info to be used afterwards
    
    def load_img(self, path):
        # cool function to load my sprites
        return pygame.transform.scale(pygame.image.load(path), (self.tile_size, self.tile_size))

    def render(self):

        self.tile_images = {
            -1: self.load_img("Assignment2/images/wall_tile.png"),
             0: self.load_img("Assignment2/images/ice_tile.png"),
             1: self.load_img("Assignment2/images/rock_tile.png"),
             2: self.load_img("Assignment2/images/small_reward.png"),
             3: self.load_img("Assignment2/images/oranberry.png")
        } # cool dict to call sprites according to their value in the board


        self.img_agent_tile = self.load_img("Assignment2/images/agent.png")
        self.img_start_tile = self.load_img("Assignment2/images/start.png")

        # Code for closing the window:
        for event in pygame.event.get():
            if event == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((25, 25, 25))

        for row in range(self.y_size):
            for col in range(self.x_size):
                tile_value = self.board[row][col] # search tile by tile the value and position to place the sprite
                image = self.tile_images.get(tile_value)
                if image:
                    self.screen.blit(image, (col * self.tile_size, row * self.tile_size)) # blit = plot dynamically

        agent_y, agent_x = self.agent_state  # Assuming agent_state = (col, row)
        self.screen.blit(self.img_agent_tile, (agent_x * self.tile_size, agent_y * self.tile_size))
        pygame.display.flip()
        

    def close(self):
        pygame.quit()
        

def create_env(random_initialization):
    # Create the environment:
    # -----------------------
    env = SootopolisGym(random_initialization = random_initialization)

    return env
