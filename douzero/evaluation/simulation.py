import multiprocessing as mp
import pickle

from douzero.env.game import GameEnv

num_landlord_wins = mp.Value('i', 0)
num_farmer_wins = mp.Value('i', 0)
num_landlord_scores = mp.Value('i', 0)
num_farmer_scores = mp.Value('i', 0)

def load_card_play_models(card_play_model_path_dict):
    players = {}

    for position in ['landlord', 'landlord_up', 'landlord_down']:
        if card_play_model_path_dict[position] == 'rlcard':
            from .rlcard_agent import RLCardAgent
            players[position] = RLCardAgent(position)
        elif card_play_model_path_dict[position] == 'random':
            from .random_agent import RandomAgent
            players[position] = RandomAgent()
        else:
            from .deep_agent import DeepAgent
            players[position] = DeepAgent(position, card_play_model_path_dict[position])
    return players

def mp_simulate(card_play_data_list, card_play_model_path_dict):

    players = load_card_play_models(card_play_model_path_dict)

    env = GameEnv(players)
    for idx, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step()
        env.reset()

    with num_landlord_wins.get_lock():
        num_landlord_wins.value += env.num_wins['landlord']

    with num_farmer_wins.get_lock():
        num_farmer_wins.value += env.num_wins['farmer']

    with num_landlord_scores.get_lock():
        num_landlord_scores.value += env.num_scores['landlord']

    with num_farmer_scores.get_lock():
        num_farmer_scores.value += env.num_scores['farmer']

def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker

def evaluate(landlord, landlord_up, landlord_down, eval_data, num_workers):

    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down}

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(mp_simulate,
                                args=(card_play_data,
                                      card_play_model_path_dict))
               for card_play_data in card_play_data_list_each_worker]

    results = [p.get() for p in results]
    num_total_wins = num_landlord_wins.value + num_farmer_wins.value
    print('WP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_wins.value / num_total_wins, num_farmer_wins.value / num_total_wins))
    print('ADP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_scores.value / num_total_wins, 2 * num_farmer_scores.value / num_total_wins)) 
