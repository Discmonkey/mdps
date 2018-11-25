from experiments.get_experiment import get_env
from mdps.solvers import policy_eval, policy_improvement, value_iteration
from mdps.visualize_policy import visualize_ice_policy, visualize_solution
import numpy as np
import matplotlib.pyplot as plt
from mdps import evaluate_solutions


name, env = get_env('taxi')

pol, rewards, scores = value_iteration(env, discount_factor=.92)

print env


def experiment(current_env, eval_func):

    x, scores_expected, num_iters, scores_actual = [], [], [], []
    for i in np.linspace(0.3, .98, 25):
        print i
        policy, score, iters = eval_func(current_env, discount_factor=i)

        # we just grab the score from the expected starting state
        scores_expected.append(score[0])

        if iters > 9999:
            iters = 0

        num_iters.append(iters)
        score_actual = evaluate_solutions(current_env, policy)
        print score_actual
        scores_actual.append(score_actual)
        x.append(i)

    return x, num_iters, scores_actual

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)


f.suptitle("Effect of Varying Discount Factor Taxi Problem")

ax1.set_title("Iterations to Convergence Policy")
ax3.set_title("Iterations to Convergence Value")
ax2.set_title("Average Score (100 runs) Policy")
ax4.set_title("Average Score (100 runs) Value")

x, num_iters_policy, scores_actual_policy = experiment(env, policy_improvement)
x, num_iters_value, scores_actual_value = experiment(env, value_iteration)

ax1.plot(x, num_iters_policy, label="Policy Improvement")
ax3.plot(x, num_iters_value, label="Value Iteration")

ax2.plot(x, scores_actual_policy, label="Policy Improvement")
ax4.plot(x, scores_actual_value, label="Value Iteration")

plt.show()