from mdps.learners import q_learning, make_epsilon_greedy_policy, step_decay
from experiments import get_env
import matplotlib.pyplot as plt
from mdps.evaluate_policy import evaluate_solutions
from mdps.visualize_policy import visualize_solution
import numpy as np

EPISODE_LENGTH = 20000
DISCOUNT = .99
ALPHA = .9
EPSILON = .9

TAIL = int(EPISODE_LENGTH * .8)
_, env_no_slip = get_env("8x8")
_, env_slip = get_env("8x8", slippery=True)


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(3, 2)
f.suptitle("8x8 problem - {} iterations, {} Discount, {} LR, "
           "{} Epsilon - No Decay".format(EPISODE_LENGTH, DISCOUNT, ALPHA, EPSILON))
ax1.set_title("No Slip")
ax2.set_title("Slip")

q_no_slip, stats_no_slip = q_learning(env_no_slip, EPISODE_LENGTH,
                                      DISCOUNT, ALPHA, EPSILON)

q_slip, stats_slip = q_learning(env_slip, EPISODE_LENGTH, DISCOUNT, ALPHA,
                                EPSILON)

ax1.plot(stats_no_slip.episode_scores, label="No Slipping")
ax2.axis('off')

visualize_ice_policy

ax3.plot(stats_no_slip.episode_scores, label="Slipping")


ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plt.show()
