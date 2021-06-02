import argparse
import pickle
import numpy as np

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])

def get_parser():
    parser = argparse.ArgumentParser(description='DouZero: random data generator')
    parser.add_argument('--output', default='eval_data', type=str)
    parser.add_argument('--num_games', default=10000, type=int)
    return parser
    
def generate():
    _deck = deck.copy()
    np.random.shuffle(_deck)
    card_play_data = {'landlord': _deck[:20],
                      'landlord_up': _deck[20:37],
                      'landlord_down': _deck[37:54],
                      'three_landlord_cards': _deck[17:20],
                      }
    for key in card_play_data:
        card_play_data[key].sort()
    return card_play_data


if __name__ == '__main__':
    flags = get_parser().parse_args()
    output_pickle = flags.output + '.pkl'

    print("output_pickle:", output_pickle)
    print("generating data...")

    data = []
    for _ in range(flags.num_games):
        data.append(generate())

    print("saving pickle file...")
    with open(output_pickle,'wb') as g:
        pickle.dump(data,g,pickle.HIGHEST_PROTOCOL)




