import gym


def get_env(env_name="", probabilistic=False):
    """
        Gets one of the environments used in the experiments

    :return: The environment
    """

    if env_name == '4x4':
        if probabilistic:
            return gym.make('FrozenLake-v0')
        else:
            env_spec = gym.spec('FrozenLake-v0')
            env_spec._kwargs['is_slippery'] = False

            return gym.make('FrozenLake-v0').env

    elif env_name == '8x8':
        env_spec = gym.spec('FrozenLake-v0')
        env_spec._kwargs['map_name'] = '8x8'

        if probabilistic:
            return gym.make('FrozenLake-v0')
        else:
            env_spec._kwargs['is_slippery'] = False

            return gym.make('FrozenLake-v0').env

    elif 'taxi':
        return gym.make('Taxi-v2').env
