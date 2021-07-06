import random

from rlcard.games.doudizhu.utils import CARD_TYPE

EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'B', 30: 'R'}
RealCard2EnvCard = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
                    'K': 13, 'A': 14, '2': 17, 'B': 20, 'R': 30}

INDEX = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4,
         '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9,
         'K': 10, 'A': 11, '2': 12, 'B': 13, 'R': 14}

class RLCardAgent(object):

    def __init__(self, position):
        self.name = 'RLCard'
        self.position = position

    def act(self, infoset):
        try:
            # Hand cards
            hand_cards = infoset.player_hand_cards
            for i, c in enumerate(hand_cards):
                hand_cards[i] = EnvCard2RealCard[c]
            hand_cards = ''.join(hand_cards)

            # Last move
            last_move = infoset.last_move.copy()
            for i, c in enumerate(last_move):
                last_move[i] = EnvCard2RealCard[c]
            last_move = ''.join(last_move)

            # Last two moves
            last_two_cards = infoset.last_two_moves
            for i in range(2):
                for j, c in enumerate(last_two_cards[i]):
                    last_two_cards[i][j] = EnvCard2RealCard[c]
                last_two_cards[i] = ''.join(last_two_cards[i])

            # Last pid
            last_pid = infoset.last_pid

            action = None
            # the rule of leading round
            if last_two_cards[0] == '' and last_two_cards[1] == '':
                chosen_action = None
                comb = combine_cards(hand_cards)
                min_card = hand_cards[0]
                for _, acs in comb.items():
                    for ac in acs:
                        if min_card in ac:
                            chosen_action = ac
                            action = [char for char in chosen_action]
                            for i, c in enumerate(action):
                                action[i] = RealCard2EnvCard[c]
                            #print('lead action:', action)
            # the rule of following cards
            else:
                the_type = CARD_TYPE[0][last_move][0][0]
                chosen_action = ''
                rank = 1000
                for ac in infoset.legal_actions:
                    _ac = ac.copy()
                    for i, c in enumerate(_ac):
                        _ac[i] = EnvCard2RealCard[c]
                    _ac = ''.join(_ac)
                    if _ac != '' and the_type == CARD_TYPE[0][_ac][0][0]:
                        if int(CARD_TYPE[0][_ac][0][1]) < rank:
                            rank = int(CARD_TYPE[0][_ac][0][1])
                            chosen_action = _ac
                if chosen_action != '':
                    action = [char for char in chosen_action]
                    for i, c in enumerate(action):
                        action[i] = RealCard2EnvCard[c]
                    #print('action:', action)
                elif last_pid != 'landlord' and self.position != 'landlord':
                    action = []

            if action is None:
                action = random.choice(infoset.legal_actions)
        except:
            action = random.choice(infoset.legal_actions)
            #import traceback
            #traceback.print_exc()

        assert action in infoset.legal_actions

        return action
        
def card_str2list(hand):
    hand_list = [0 for _ in range(15)]
    for card in hand:
        hand_list[INDEX[card]] += 1
    return hand_list

def list2card_str(hand_list):
    card_str = ''
    cards = [card for card in INDEX]
    for index, count in enumerate(hand_list):
        card_str += cards[index] * count
    return card_str

def pick_chain(hand_list, count):
    chains = []
    str_card = [card for card in INDEX]
    hand_list = [str(card) for card in hand_list]
    hand = ''.join(hand_list[:12])
    chain_list = hand.split('0')
    add = 0
    for index, chain in enumerate(chain_list):
        if len(chain) > 0:
            if len(chain) >= 5:
                start = index + add
                min_count = int(min(chain)) // count
                if min_count != 0:
                    str_chain = ''
                    for num in range(len(chain)):
                        str_chain += str_card[start+num]
                        hand_list[start+num] = int(hand_list[start+num]) - int(min(chain))
                    for _ in range(min_count):
                        chains.append(str_chain)
            add += len(chain)
    hand_list = [int(card) for card in hand_list]
    return (chains, hand_list)

def combine_cards(hand):
    '''Get optimal combinations of cards in hand
    '''
    comb = {'rocket': [], 'bomb': [], 'trio': [], 'trio_chain': [],
            'solo_chain': [], 'pair_chain': [], 'pair': [], 'solo': []}
    # 1. pick rocket
    if hand[-2:] == 'BR':
        comb['rocket'].append('BR')
        hand = hand[:-2]
    # 2. pick bomb
    hand_cp = hand
    for index in range(len(hand_cp) - 3):
        if hand_cp[index] == hand_cp[index+3]:
            bomb = hand_cp[index: index+4]
            comb['bomb'].append(bomb)
            hand = hand.replace(bomb, '')
    # 3. pick trio and trio_chain
    hand_cp = hand
    for index in range(len(hand_cp) - 2):
        if hand_cp[index] == hand_cp[index+2]:
            trio = hand_cp[index: index+3]
            if len(comb['trio']) > 0 and INDEX[trio[-1]] < 12 and (INDEX[trio[-1]]-1) == INDEX[comb['trio'][-1][-1]]:
                comb['trio'][-1] += trio
            else:
                comb['trio'].append(trio)
            hand = hand.replace(trio, '')
    only_trio = []
    only_trio_chain = []
    for trio in comb['trio']:
        if len(trio) == 3:
            only_trio.append(trio)
        else:
            only_trio_chain.append(trio)
    comb['trio'] = only_trio
    comb['trio_chain'] = only_trio_chain
    # 4. pick solo chain
    hand_list = card_str2list(hand)
    chains, hand_list = pick_chain(hand_list, 1)
    comb['solo_chain'] = chains
    # 5. pick par_chain
    chains, hand_list = pick_chain(hand_list, 2)
    comb['pair_chain'] = chains
    hand = list2card_str(hand_list)
    # 6. pick pair and solo
    index = 0
    while index < len(hand) - 1:
        if hand[index] == hand[index+1]:
            comb['pair'].append(hand[index] + hand[index+1])
            index += 2
        else:
            comb['solo'].append(hand[index])
            index += 1
    if index == (len(hand) - 1):
        comb['solo'].append(hand[index])
    return comb
