from mdps.learners import q_learning, make_epsilon_greedy_policy, step_decay
from experiments import get_env
import matplotlib.pyplot as plt
from mdps.evaluate_policy import evaluate_solutions
import numpy as np

EPISODE_LENGTH = 10000
TAIL = int(EPISODE_LENGTH * .8)
_, env_no_slip = get_env("8x8")
_, env_slip = get_env("8x8", slippery=True)

f, ((ax1, ax2), (ax3, ax4), (ax4, ax5)) = plt.subplots(3, 2)
f.suptitle("8x8 problem - 1000 iterations, .85 Discount, .5 LR, .1 Epsilon - No Decay")
ax1.set_title("No Slip")
ax2.set_title("Slip")

s = step_decay([5000, 2500, 1000])

_, stats_no_slip = q_learning(env_no_slip, EPISODE_LENGTH, 1.0, .8, .9)
_, stats_slip = q_learning(env_slip, EPISODE_LENGTH, 1.0, .8, .9)

ax1.plot(stats_no_slip.episode_scores, label="No Slipping")
ax2.plot(stats_no_slip.episode_scores[TAIL:], label="No Slipping > {}".format(TAIL))

ax3.plot(stats_no_slip.episode_scores, label="Slipping")
ax4.plot(stats_no_slip.episode_scores[TAIL:], label="Slipping > {}".format(TAIL))

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plt.show()
