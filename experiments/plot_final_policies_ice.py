from experiments import get_env
from mdps.solvers import policy_improvement, value_iteration
from mdps.visualize_policy import visualize_ice_policy
import matplotlib.pyplot as plt


DISCOUNT_FACTOR = .95

f, ((four_slip_policy, four_slip_value), (eight_policy, eight_value)) = plt.subplots(2, 2)

four_slip_policy.axis('off')
four_slip_value.axis('off')
eight_value.axis('off')
eight_policy.axis('off')

f.suptitle("Final Ice Policies")

_, env = get_env('4x4', slippery=True)

policy, _, iters = policy_improvement(env, discount_factor=DISCOUNT_FACTOR)

visualize_ice_policy(env, policy, ax=four_slip_policy)
four_slip_policy.set_title("Policy Improvement {} Iters".format(iters))

policy, _, iters = value_iteration(env, discount_factor=DISCOUNT_FACTOR)

visualize_ice_policy(env, policy, ax=four_slip_value)
four_slip_value.set_title("Value Iteration {} Iters".format(iters))

_, env = get_env('8x8', slippery=True)

policy, _, iters = policy_improvement(env, discount_factor=DISCOUNT_FACTOR)

visualize_ice_policy(env, policy, ax=eight_policy)
eight_policy.set_title("Policy Improvement {} Iters".format(iters))

policy, _, iters = value_iteration(env, discount_factor=DISCOUNT_FACTOR)

visualize_ice_policy(env, policy, ax=eight_value)
eight_value.set_title("Value Iteration {} Iters".format(iters))

plt.show()
