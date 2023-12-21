import gym
import numpy as np

class TobaccoCuringEnv(gym.Env):
    def __init__(self):
        # Define the state space
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array([100, 100, 100]))  # Temperature, humidity, leaf moisture content

        # Define the action space
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]))  # Temperature adjustment, humidity adjustment, ventilation adjustment

        # Reset the environment
        self.reset()

    def step(self, action):
        # Apply the action to the environment
        moisture_loss_rate = 0.1
        temperature = self.temperature + action[0]
        humidity = self.humidity + action[1]
        ventilation = self.ventilation + action[2]

        # Update the leaf moisture content
        # leaf_moisture_content = self.leaf_moisture_content - 0.1  # Simulate moisture loss over time
        self.leaf_moisture_content -= moisture_loss_rate * (1 - temperature / 100) * (1 - humidity / 100)

        # Calculate the reward
        reward = calculate_reward(temperature, humidity, self.leaf_moisture_content)

        # Check if the episode is done
        # done = False  # Implement a termination condition based on the curing process
        done = self.leaf_moisture_content <= 20

        # Update the state
        self.temperature = temperature
        self.humidity = humidity
        self.ventilation = ventilation
        self.leaf_moisture_content = self.leaf_moisture_content


        # Return the new state, reward, and done flag
        return [temperature, humidity, self.leaf_moisture_content], reward, done, {}

    def reset(self):
        # Reset the environment to its initial state
        self.temperature = 25
        self.humidity = 60
        self.ventilation = 50
        self.leaf_moisture_content = 80

        return [self.temperature, self.humidity, self.leaf_moisture_content]

def calculate_reward(temperature, humidity, leaf_moisture_content):
        reward = 0
        # Calculate the error between the current temperature and the desired temperature
        temperature_error = (temperature - 50)

        # Calculate the error between the current humidity and the desired humidity
        humidity_error = (humidity - 70)

        # Calculate the error between the current leaf moisture content and the desired leaf moisture content
        leaf_moisture_content_error = (leaf_moisture_content - 20)**2

        # Calculate the total reward
        reward = (temperature_error + humidity_error + leaf_moisture_content_error)

        # Bonus reward for achieving the desired color
        if temperature >= 65 and temperature <= 85 and humidity >= 65 and humidity <= 75:
            reward += 100

        # Bonus reward for achieving the desired flavor
        if leaf_moisture_content >= 15 and leaf_moisture_content <= 25:
            reward += 50

        return reward
