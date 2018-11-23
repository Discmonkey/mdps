"""
    Look at the effect of the discount factor in finding the optimal policy.
    Also evaluate the p
"""
from experiments import get_env
from mdps.solvers import policy_eval, policy_improvement, value_iteration
from mdps.visualize_policy import visualize_ice_policy
import numpy as np
import matplotlib.pyplot as plt


env = get_env('4x4', probabilistic=False)

pol, _, iters = value_iteration(env, discount_factor=.9)

print iters

x_axis = np.linspace(0.01, .99, 100)
for i in np.linspace(0.01, .99, 100):
    _, score, iters = value_iteration(env, discount_factor=i)
    print iters, score[0]

