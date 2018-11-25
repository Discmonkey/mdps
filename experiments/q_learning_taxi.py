from mdps.learners import q_learning, make_epsilon_greedy_policy, linear_decrease, exponential_decrease
from experiments import get_env
import matplotlib.pyplot as plt
from mdps.evaluate_policy import evaluate_solutions
import numpy as np

name_no_slip, env_no_slip = get_env("taxi")

plt.title("Effect of Exploration on Taxi Problem")

policy, stats_no_slip = q_learning(env_no_slip, 10000, .65, .6, .1)

f, (ax1, ax2) = plt.subplots(1, 2)

f.suptitle("10000 iterations, .65 Discount, .6 Alpha, .1 Epsilon - No Decay")
ax1.plot(stats_no_slip.episode_scores, label="All Episodes")
ax2.plot(stats_no_slip.episode_scores[4000:], label="Episodes after episode 2000")
plt.legend()
plt.show()
