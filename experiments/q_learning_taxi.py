from mdps.learners import q_learning, make_epsilon_greedy_policy
from experiments import get_env
import matplotlib.pyplot as plt
from mdps.evaluate_policy import evaluate_solutions
import numpy as np

name_no_slip, env_no_slip = get_env("taxi")

plt.title("Effect of Exploration on Taxi Problem")

for i in np.linspace(0.01, 1, 5):
    _, stats_no_slip = q_learning(env_no_slip, 1000000, .85, .3, i)

    plt.plot(stats_no_slip.episode_rewards.clip(0), label="Epsilon: {}".format(i))

plt.legend()
plt.show()
