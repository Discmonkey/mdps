from mdps.learners import q_learning, make_epsilon_greedy_policy
from experiments import get_env
import matplotlib.pyplot as plt
from mdps.evaluate_policy import evaluate_solutions
import numpy as np


name_no_slip, env_no_slip = get_env("4x4")
name_slip, env_slip = get_env("4x4", slippery=True)

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle("Effect of Exploration on 4x4 Ice problem")
ax1.set_title("No Slip")
ax2.set_title("Slip")

for i in np.linspace(0.01, 1, 5):
    print i
    _, stats_no_slip = q_learning(env_no_slip, 1000, .85, .5, i)
    _, stats_slip = q_learning(env_slip, 1000, .85, .5, i)

    ax1.plot(stats_no_slip.episode_rewards, label="Eps: {}".format(i))
    ax2.plot(stats_slip.episode_rewards, label="Eps: {}".format(i))

    print "Done with ", i

ax1.legend()
ax2.legend()
plt.show()
