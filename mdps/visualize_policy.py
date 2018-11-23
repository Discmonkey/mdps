import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

def get_ice_map(env):
    frozen_world = env.desc
    heat_map = np.zeros(frozen_world.shape)
    for y, row in enumerate(frozen_world):
        for x, col in enumerate(row):
            if col == 'S' or col == 'F':
                val = 0
            elif col == 'H':
                val = -1
            else:
                val = 1

            heat_map[y][x] = val

    return heat_map


def visualize_ice(env):

    heat_map = get_ice_map(env)

    seaborn.heatmap(heat_map, cmap="PuBuGn")
    plt.show()


def visualize_ice_policy(env, policy):
    """
        Plots the ice policy by drawing one of "U, D, R, L" on top off <- ^
    :param env: ice env
    :param policy: policy found
    :return: Simply plots the policy
    """

    heat_map = get_ice_map(env)
    annots = np.chararray(heat_map.shape)

    policy_to_int = np.argmax(policy, axis=1)
    policy_to_int = policy_to_int.reshape(heat_map.shape)

    for y, row in enumerate(policy_to_int):
        for x, col in enumerate(row):
            mx = col
            if mx == LEFT:
                cc = ('L')
            elif mx == RIGHT:
                cc = ('R')
            elif mx == DOWN:
                cc = ('D')
            elif mx == UP:
                cc = ('U')

            annots[y][x] = cc

    seaborn.heatmap(heat_map, cmap="PuBuGn", annot=annots, fmt='')
    plt.show()



def visualize_taxi_field(taxi_world):
    pass



