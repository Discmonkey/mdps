"""
    Look at the effect of the discount factor in finding the optimal policy.
    Also evaluate the p
"""
from experiments.get_experiment import env_generator
from mdps.solvers import policy_eval, policy_improvement, value_iteration
from mdps.visualize_policy import visualize_ice_policy
import numpy as np
import matplotlib.pyplot as plt
from mdps import evaluate_solutions

USE_POLICY = True

if USE_POLICY:
    eval_func = policy_improvement
    plot_title = "Policy Improvement"
else:
    eval_func = value_iteration
    plot_title = "Value Iteration"


def experiment(current_env):

    x, scores_expected, num_iters, scores_actual = [], [], [], []
    for i in np.linspace(0.01, .99, 25):
        print i
        policy, score, iters = eval_func(current_env, discount_factor=i)

        # we just grab the score from the expected starting state
        scores_expected.append(score[0])
        num_iters.append(iters)
        scores_actual.append(evaluate_solutions(current_env, policy))
        x.append(i)

    return x, num_iters, scores_expected, scores_actual

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

f.suptitle("Discount factor effect on {} Algorithm".format(plot_title))

ax1.set_title("Number of iterations")
ax2.set_title("Expected score from starting state")
ax3.set_title("Average Score after 100 Runs")

for name, env in env_generator():
    x, num_iters, scores, scores_expected = experiment(env)

    ax1.plot(x, num_iters, label=name)
    ax2.plot(x, scores, label=name)
    ax3.plot(x, scores_expected, label=name)
    print "Done with", name

ax1.legend()
ax2.legend()
ax3.legend()

plt.show()
