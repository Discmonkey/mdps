import gym


def get_env(env_name="", slippery=False):
    """
        Gets one of the environments used in the experiments

    :return: The environment
    """

    if env_name == '4x4':
        env_spec = gym.spec('FrozenLake-v0')
        env_spec._kwargs['map_name'] = '4x4'

        if slippery:
            env_spec._kwargs['is_slippery'] = True
            return "4x4 frozen - slip", gym.make('FrozenLake-v0').env
        else:
            env_spec._kwargs['is_slippery'] = False

            return "4x4 frozen - no slip", gym.make('FrozenLake-v0').env

    elif env_name == '8x8':
        env_spec = gym.spec('FrozenLake-v0')
        env_spec._kwargs['map_name'] = '8x8'

        if slippery:
            env_spec._kwargs['is_slippery'] = True
            return "8x8 frozen - slip", gym.make('FrozenLake-v0').env
        else:
            env_spec._kwargs['is_slippery'] = False
            return "8x8 frozen - no slip", gym.make('FrozenLake-v0').env

    elif 'taxi':
        return 'Taxi', gym.make('Taxi-v2').env


def env_generator(include_taxi=False):
    """
        Iterates through available environments

    :yield: Next environment
    """
    specs = [
        ('4x4', False),
        ('8x8', False),
        ('4x4', True),
        ('8x8', True),
        ('taxi', False)
    ]

    if not include_taxi:
        specs.pop()

    for spec in specs:
        yield get_env(*spec)
