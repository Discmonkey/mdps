import numpy as np


def evaluate_solutions(env, policy, num_iterations=100):
    current_state = env.reset()
    all_rewards = 0.0

    for _ in range(num_iterations):
        episode_over = False
        total_reward = 0
        iters = 0
        while not episode_over and iters < 1000:
            iters += 1

            action = np.argmax(policy[current_state])

            current_state, state_reward, episode_over, _ = env.step(action)

            total_reward += state_reward

        all_rewards += float(total_reward)
        env.reset()

    return all_rewards / num_iterations
