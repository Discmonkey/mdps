import random
import numpy as np


def make_random_policy(num_states, num_actions):
    """
    Generates a random policy of shape [num_states, num_actions]

    :param num_states: Number of states
    :param num_actions: Number of actions per states
    :return:
    """

    policy = np.zeros((num_states, num_actions))

    for i in range(num_actions):
        policy[i][random.randint(0, num_actions - 1)] = 1

    return policy
