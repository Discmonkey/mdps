from mdps.learners import q_learning, make_epsilon_greedy_policy, step_decay
from experiments import get_env
import matplotlib.pyplot as plt
from mdps.evaluate_policy import evaluate_solutions
import numpy as np
import pickle


EPISODE_LENGTH = 100000
TAIL = int(EPISODE_LENGTH * .8)
_, env_no_slip = get_env("8x8")
_, env_slip = get_env("8x8", slippery=True)

s = step_decay([5000, 2500, 1000])

_, stats_no_slip = q_learning(env_no_slip, EPISODE_LENGTH, 1.0, .8, .9)
_, stats_slip = q_learning(env_slip, EPISODE_LENGTH, 1.0, .8, .9)

with open("stats_no_slip.p", "wb") as f:
    pickle.dump(stats_no_slip, f)


with open("stats_slip.p", "wb") as f:
    pickle.dump(stats_slip, f)