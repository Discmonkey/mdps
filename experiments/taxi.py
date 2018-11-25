import gym
from mdps import solvers
from experiments import get_env
from mdps.visualize_policy import visualize_solution

name, env = get_env('taxi')

p, _, _ = solvers.value_iteration(env, discount_factor=.95)

visualize_solution(env, p)