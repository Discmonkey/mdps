import gym
from mdps import solvers

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

env_spec = gym.spec('FrozenLake-v0')

env_spec._kwargs['is_slippery'] = False

env = gym.make('FrozenLake-v0')

env.reset()

print solvers.value_iteration(env.env)
