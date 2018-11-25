from mdps.learners import q_learning, make_epsilon_greedy_policy, step_decay
from experiments import get_env
import matplotlib.pyplot as plt
from mdps.evaluate_policy import evaluate_solutions
from mdps.visualize_policy import visualize_ice_policy, convert_q_to_policy
import numpy as np

EPISODE_LENGTH = 40000
DISCOUNT = .90
ALPHA = .85
EPSILON = .7

TAIL = int(EPISODE_LENGTH * .8)
_, env_no_slip = get_env("8x8")
_, env_slip = get_env("8x8", slippery=True)


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
f.suptitle("8x8 problem - {} iterations, {} Discount, {} LR, "
           "{} Epsilon - No Decay".format(EPISODE_LENGTH, DISCOUNT, ALPHA, EPSILON))
ax1.set_title("Policy Evaluation per Episode - No Slip")
ax2.set_title("Final Policy - No Slip")
ax3.set_title("Policy Evaluation per Episode - Slip")
ax4.set_title("Final Policy - Slip")

q_no_slip, stats_no_slip = q_learning(env_no_slip, EPISODE_LENGTH,
                                      DISCOUNT, ALPHA, EPSILON)

q_slip, stats_slip = q_learning(env_slip, EPISODE_LENGTH, DISCOUNT, ALPHA,
                                EPSILON)

ax1.plot(stats_no_slip.episode_scores, label="No Slipping")
ax2.axis('off')
pol_no_slip = convert_q_to_policy(q_no_slip, env_no_slip)
visualize_ice_policy(env_no_slip, pol_no_slip, ax=ax2)

ax3.plot(stats_slip.episode_scores, label="Slipping")
ax4.axis('off')

pol_slip = convert_q_to_policy(q_slip, env_slip)
visualize_ice_policy(env_slip, pol_slip, ax=ax4)


ax1.legend()
ax3.legend()


plt.show()
