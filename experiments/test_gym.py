import gym

env = gym.make('FrozenLake-v0')
env.reset()
print env.action_space
print env.reward_range
print env.observation_space

env.render()

