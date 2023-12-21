import gym
import time
from stable_baselines3 import TD3


# Import the custom environment
from TobaccoEnv import TobaccoCuringEnv

# Create the environment
env = TobaccoCuringEnv()

# Initialize the agent
agent = TD3(
    policy="MlpPolicy",
    env=env,
    verbose=1,
)

# Train the agent for 100,000 steps
agent.learn(total_timesteps=500)

# Save the trained agent
agent.save("tobacco_curing_agent")


# Load the trained agent
agent = TD3.load("tobacco_curing_agent")

# Control the curing process

obs = env.reset()
episode_count = 0
while episode_count < 5:  # Run for 5 episodes
    action, _ = agent.predict(obs)
    obs, reward, done, _ = env.step(action)
    print("Step:", time.time(), obs, reward)
    if done:
        episode_count += 1
        obs = env.reset()  # Reset the environment for the next episode

# Close the environment
env.close()
