from experiments import get_env
from mdps.solvers import policy_improvement, value_iteration
from mdps.visualize_policy import visualize_ice_policy
import matplotlib.pyplot as plt


DISCOUNT_FACTOR = .97

f, ((four_slip, four_no_slip), (eight_slip, eight_no_slip)) = plt.subplots(2, 2)

f.suptitle("Final Ice Policies")

name, env = get_env('4x4', slippery=False)

