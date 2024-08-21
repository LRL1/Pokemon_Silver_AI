#### Libraries ####
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

## DQN Network Libraries ##
from stable_baselines3 import PPO

#### Gameboy Emulator ####
from pyboy import PyBoy
# Get the path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go two directories up (out of V1 and then out of Silver_KI)
base_dir = os.path.dirname(os.path.dirname(current_dir))
# Create the path to the Silver_Game directory
silver_game_dir = os.path.join(base_dir, "Pokemon_Silver_Game")
# Add the path to the game file
rom_path = os.path.join(silver_game_dir, "Pokemon_Silver.gbc")
# Add the parent directory to the Python search path
sys.path.append(base_dir)
import Pokemon_Silver_Adresses.adresses as ad
emulation_speed = int(input("What speed do you want? (0=Unlimited, 1=1x, 2=2x.....): "))

#### Global Variables ####
action_map = [" ", "a", "b", "left", "right", "up", "down", "start", "select"]
channels = 1  # Grayscale image, so only one channel is used
# Display size of the emulator: height = 144, width = 160 | Divided by 2 because it was compressed to half size
height = 72
width = 80

#### Main Program ####
## Environment Definition ##
class PokemonEnv(gym.Env):
    
    def __init__(self):
        super(PokemonEnv, self).__init__()  # Initialize the gym.Env superclass
        # Define the action space
        self.action_space = spaces.Discrete(len(action_map))
        # Low = Lowest value, High = Highest value, shape = shape of the observation space + color space, dtype = data type
        self.observation_space = spaces.Box(low=0, high=255, shape=(channels, height, width), dtype=np.uint8)

    
    def start_emulator(self):
        self.pyboy = PyBoy(rom_path, window="SDL2")  # Initialize the emulator
        self.pyboy.set_emulation_speed(emulation_speed)  # Set the emulation speed
        print("Debugging: Emulator started")

        return self.get_observation

    def stop_emulator(self):
        self.pyboy.stop()
        self.pyboy = None
        print("Debugging: Emulator stopped!")
    
    def reset(self, seed=None, options=None):
        ## The reset function acts as an initialization function for StableBaselines3 ##
        # Set environment variables
        self.done = 0    
        self.reward = 0 
        self.obs = 0
        self.positions = []
        self.info = {}
        self.counter_log = 0
        self.pyboy = PyBoy(rom_path, window="SDL2")  # Initialize the emulator
        
        # Stop the emulator
        self.stop_emulator()
        # Start the emulator
        self.start_emulator()

        print("Debugging: Reset complete!")

        self.info = {}
        self.get_observation()
        return self.observation, self.info

    def step(self, action):
        # Convert the given number to a corresponding action
        if action == 0:
            pass
        else:
            self.pyboy.button(action_map[action])
        self.pyboy.tick(1, True)

        self.get_observation()
        self.get_done()
        self.get_reward()
        self.info = {}
        self.truncated = 0

        return self.observation, self.reward, self.done, self.truncated, self.info

    def get_done(self):
        # Define the done variable
        self.done = False
        # 1 = Battle lost, end of game initiated, 2 = Ran away
        battle_result = self.pyboy.memory[ad.battle_result]
        if battle_result == 1:
            self.done = True

    
    def get_reward(self):
        self.reward = 0
        position_x = self.pyboy.memory[ad.player_x]
        position_y = self.pyboy.memory[ad.player_y]

        # Check if game over is reached, otherwise calculate reward
        if self.done == False:
            # Determine position reward
            if (position_x, position_y) in self.positions:
                pass
            elif (position_x, position_y) not in self.positions:
                self.positions.append((position_x, position_y))
                self.reward += 1
            else:
                print("What are you programming again????!!!!!")

        else:
            self.reward = -100

    # Function to capture the image from the emulator
    def get_observation(self):
        # Convert image to grayscale
        gray_image = self.convert_to_grayscale(self.pyboy.screen.ndarray)
        
        # Resize the image (half size)
        new_height = gray_image.shape[0] // 2
        new_width = gray_image.shape[1] // 2
        
        # Scale the image
        resized_observation = cv2.resize(gray_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Add an additional dimension to resized_observation to get the shape (1, new_height, new_width)
        self.observation = np.expand_dims(resized_observation, axis=0)

    
    # Function to display the observation image
    def show(self):
        self.get_observation()
        # Check if the observation is in the correct format
        if self.observation.shape[0] == 1:
            # If the image is in channel-first format (1, 144, 160),
            # then we need to convert it to channel-last format (144, 160)
            display_image = self.observation[0, :, :]
        else:
            # If it is already in the correct format, just use it
            display_image = self.observation

        # Display the image from the get_observation function in a separate window
        plt.imshow(display_image, cmap="gray")  # Colormap gray is used to visualize what the AI sees
        plt.colorbar()  # Add a color legend based on the colormap
        plt.title('Game Area Visualization')  # Set the window title
        plt.show()  # Display the window with the set settings

    def convert_to_grayscale(self, image):
        # Check if the input image has a fourth channel
        if image.shape[2] == 4:
            image = image[:, :, :3]  # Cut off the fourth channel

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return gray_image  # Return the grayscale image without an additional channel

    
    def log(self):
        self.counter_log += 1
        # Emulation speed maximum = output value every frame
        if emulation_speed == 0:
            print(f"Reward: {self.reward}")
        # Emulation speed > 0 = output value every 60 frames ~ every second
        elif self.counter_log == 60 * emulation_speed:
            print(f"Reward: {self.reward}")
            self.counter_log = 0
        
    def render(self):
        # Theoretical implementation of a render function, but not necessary due to using the emulator
        pass

    def close(self):
        self.stop_emulator()





#### Agent Settings ####
models_dir = os.path.join(current_dir, "models")
log_dir = os.path.join(current_dir, "logs")

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)




#### Main Loop ####
env = PokemonEnv()
env.reset()

timesteps = int(input("How many steps should the AI train per episode? (Integer values): "))
episodes = int(input("How many episodes should the AI train/load? (Integer values): "))
total_reward = 0

# Check whether the model should be trained or loaded #
decision = int(input("Should the model train or load the latest one? (1 = Train, 2 = Load): "))
load = False
training = False

if decision == 1:
    load = False
    training = True
elif decision == 2:
    load = True
    training = False
else:
    print("Invalid input!!!!")

if training == True:
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    for episode in range(1, episodes+1):
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{timesteps*episode}")



if load == True:
    model_path = f"{models_dir}/1000000.zip"
    model = PPO.load(model_path, env=env)

    for episode in range(1, episodes+1):
        obs, _ = env.reset()
        done = False
        while not done:
          action, _ = model.predict(obs)
          obs, reward, done, truncated, info = env.step(action)
          total_reward += reward
          print(total_reward)
          print(f"Action: {action}")
