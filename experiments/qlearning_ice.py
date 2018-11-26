from mdps.learners import q_learning, make_epsilon_greedy_policy, step_decay
from experiments import get_env
import matplotlib.pyplot as plt
from mdps.evaluate_policy import evaluate_solutions
from mdps.visualize_policy import visualize_ice_policy, convert_q_to_policy
import numpy as np

EPISODE_LENGTH = 200000
DECREASE_EPSILON_POINT = 50000
ALPHA_DECREASE_POINT = 100000
DISCOUNT = .90
ALPHA = .85
EPSILON = .9

TAIL = int(EPISODE_LENGTH * .8)
_, env_no_slip = get_env("8x8")
_, env_slip = get_env("8x8", slippery=True)


f, (ax1, ax2) = plt.subplots(1, 2)
f.suptitle("8x8 problem Linear Decrease - {} iterations, {} Discount, {} LR, "
           "{} Epsilon\n Epsilon Decay after {} Alpha Decay After {}".format(EPISODE_LENGTH, DISCOUNT, ALPHA, EPSILON,
                                                DECREASE_EPSILON_POINT, ALPHA_DECREASE_POINT))
ax1.set_title("Success per hundred Episodes")
ax2.set_title("Final Policy - Slip")


q_slip, stats_slip = q_learning(env_slip, EPISODE_LENGTH, DISCOUNT, ALPHA,
                                EPSILON, epsilon_division_point=DECREASE_EPSILON_POINT, alpha_division_point=ALPHA_DECREASE_POINT)

ax1.plot(stats_slip.success_per_hundred, label="No Slipping")
pol_slip = convert_q_to_policy(q_slip, env_slip)
ax2.axis('off')
visualize_ice_policy(env_no_slip, pol_slip, ax=ax2)

ax2.plot(stats_slip.episode_scores, label="Slipping")

ax1.legend()


plt.show()
