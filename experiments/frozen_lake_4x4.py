import gym
from mdps import solvers
from mdps.util import make_random_policy
from mdps.visualize_policy import visualize_ice
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

environment_name = 'Taxi-v2'

env_spec = gym.spec('Taxi-v2')
#env_spec._kwargs['is_slippery'] = False

env = gym.make('FrozenLake-v0')

env.reset()

visualize_ice(env.env)

print solvers.value_iteration(env.env, discount_factor=.9)[0]
print solvers.policy_improvement(env.env, discount_factor=.9)[0]

