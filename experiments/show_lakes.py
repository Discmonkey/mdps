import gym
from mdps.visualize_policy import visualize_ice

# env_spec = gym.spec("FrozenLake-v0")
# env_spec._kwargs['map_name'] = '8x8'

env = gym.make("FrozenLake-v0")

visualize_ice(env.env)

