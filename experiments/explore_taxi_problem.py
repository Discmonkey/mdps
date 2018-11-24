from experiments.get_experiment import get_env
from mdps.solvers import policy_eval, policy_improvement, value_iteration
from mdps.visualize_policy import visualize_ice_policy
import numpy as np
import matplotlib.pyplot as plt
from mdps import evaluate_solutions


name, env = get_env('taxi')

pol, rewards, scores = value_iteration(env, discount_factor=.92)

print env


def experiment(current_env, eval_func):

    x, scores_expected, num_iters, scores_actual = [], [], [], []
    for i in np.linspace(0.05, .99, 25):
        print i
        policy, score, iters = eval_func(current_env, discount_factor=i)

        # we just grab the score from the expected starting state
        scores_expected.append(score[0])
        num_iters.append(iters)
        scores_actual.append(evaluate_solutions(current_env, policy))
        x.append(i)

    return x, num_iters, scores_actual

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)


f.suptitle("Final Ice Policies")

ax1.set_title("Iterations to Convergence")
ax2.set_title("Average Score (100 runs)")

x, num_iters_policy, scores_actual_policy = experiment(env, policy_improvement)
x, num_iters_value, scores_actual_value = experiment(env, value_iteration)

ax1.plot(x, num_iters_policy, label="Policy Improvement")
ax1.plot(x, num_iters_value, label="Value Iteration")

ax2.plot(x, scores_actual_policy, label="Policy Improvement")
ax2.plot(x, scores_actual_policy, label="Value Iteration")

plt.show()