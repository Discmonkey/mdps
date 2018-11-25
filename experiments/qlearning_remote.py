from mdps.learners import q_learning, make_epsilon_greedy_policy, step_decay
from experiments import get_env
import matplotlib.pyplot as plt
from mdps.evaluate_policy import evaluate_solutions
import numpy as np
import pickle
import random


EPISODE_LENGTH = 5
TAIL = int(EPISODE_LENGTH * .8)
_, env_no_slip = get_env("8x8")
_, env_slip = get_env("8x8", slippery=True)

s = step_decay([5000, 2500, 1000])

_, stats_no_slip = q_learning(env_no_slip, EPISODE_LENGTH, .95, .8, .9)
_, stats_slip = q_learning(env_slip, EPISODE_LENGTH, .95, .8, .9)

copy_dict_slip = {
    "ep_length": stats_slip.episode_lengths.tolist(),
    "ep_score": stats_slip.episode_scores.tolist()
}

copy_dict_no_slip = {
    "ep_length": stats_no_slip.episode_lengths.tolist(),
    "ep_score": stats_no_slip.episode_scores.tolist()
}


with open("stats_no_slip{}.p".format(random.randint(1, 200)), "wb") as f:
    pickle.dump(copy_dict_no_slip, f)


with open("stats_slip{}.p".format(random.randint(1, 200)), "wb") as f:
    pickle.dump(copy_dict_slip, f)