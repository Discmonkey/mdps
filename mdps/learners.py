"""
    Base code is attributed to:
    https://github.com/dennybritz/reinforcement-learning


"""


import numpy as np
import sys
from collections import defaultdict
import itertools
from collections import namedtuple
from evaluate_policy import evaluate_solutions


EpisodeStats = namedtuple('stats', ["episode_lengths", "episode_rewards",
                                    "episode_scores", "epsilon_values", "success_per_hundred"])


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def constant_decrease(alpha, current_episode, total_episodes):
    return alpha


def linear_decrease(alpha, current_episode, total_episodes):
    return float(total_episodes + 1 - current_episode) / total_episodes * alpha


def exponential_decrease(alpha, current_episode, total_episodes):
    if current_episode < 4000:
        return alpha
    return alpha * np.e ** (- 2.0 * current_episode / total_episodes)


class step_decay:

    def __init__(self, steps=None):
        if steps == None:
            self.func = constant_decrease
        else:
            self.steps = steps

    def __call__(self, alpha, current_episode, total_episodes):
        i = 0
        try:
            while current_episode > self.steps[i]:
                i += 1
                alpha /= 2

        except Exception as e:
            return alpha

        return alpha


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1,
               epsilon_division_point=None, alpha_division_point=None):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        :param epsilon_division_point: Divide epsilon past this number of iterations
        :param decay_func:
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_scores=np.zeros(num_episodes),
        epsilon_values=np.zeros(num_episodes),
        success_per_hundred=[]
    )

    # The policy we're following
    divided = False
    alpha_divided = False
    total_successes = 0
    for i_episode in range(num_episodes):

        if epsilon_division_point is not None and i_episode > epsilon_division_point and not divided:
            epsilon /= 2
            divided = True
            print "reduced epsilon!"

        if alpha_division_point is not None and i_episode > alpha_division_point:
            alpha /= 2 

        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        # Print out which episode we're on, useful for debugging.

        if i_episode % 100 == 0:
            stats.success_per_hundred.append(total_successes)
            total_successes = 0
            print ("\r{}".format(i_episode))
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            action_probs = policy(state)

            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            if reward > 0:
                total_successes += 1
                print "Found Reward!"
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])

            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                stats.episode_scores[i_episode] = evaluate_solutions(env, Q, num_iterations=3)
                stats.epsilon_values[i_episode] = epsilon
                break

            state = next_state

    return Q, stats

