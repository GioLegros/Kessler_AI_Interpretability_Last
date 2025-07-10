def stay_alive_reward(game_state, prev_state, tick_rate=30):
    """
    A simple reward function that emphasizes staying alive for as long as possible
    :param game_state: The game state
    :param tick_rate:  The game tick rate
    :return: 1 / tick_rate, i.e a cumulative reward of tick_rate means the agent stayed alive for 1 second.
    """
    return 1.0 / tick_rate
