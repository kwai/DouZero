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
            # The rule of leading round
            # RULE: This gets all of the combinations of cards in the players hand.
            # Then, it chooses the action which contains your hands lowest card.add()
            # If two hands contain the lowest card, it prioritzes the one with the
            # most cards, ie a straight over a single.
            if last_two_cards[0] == '' and last_two_cards[1] == '':
                chosen_action = None
                comb = self.combine_cards(hand_cards)
                min_card = hand_cards[0]
                for _, acs in comb.items():
                    for ac in acs:
                        if min_card in ac:
                            action = getActionArr(ac)
            # The rule of following cards
            # Rule:
            else:
                the_type = CARD_TYPE[0][last_move][0][0]
                # this is a tuple of type (pair, straight etc) and its rank of that type
                chosen_action = ''
                rank = 1000

                # choose the legal action with the lowest rank ie play your pair of 5's over your
                # pair of aces
                for ac in infoset.legal_actions:
                    _ac = ac.copy()
                    for i, c in enumerate(_ac):
                        _ac[i] = EnvCard2RealCard[c]
                    _ac = ''.join(_ac)
                    if _ac != '' and the_type == CARD_TYPE[0][_ac][0][0]:
                        if int(CARD_TYPE[0][_ac][0][1]) < rank:
                            rank = int(CARD_TYPE[0][_ac][0][1])
                            chosen_action = _ac

                # if ther is an action, format it as an array
                if chosen_action != '':
                    action = [char for char in chosen_action]
                    for i, c in enumerate(action):
                        action[i] = RealCard2EnvCard[c]

                # else pass
                elif last_pid != 'landlord' and self.position != 'landlord':
                    action = []

            if action is None:
                action = random.choice(infoset.legal_actions)
        except:
            action = random.choice(infoset.legal_actions)
            # import traceback
            # traceback.print_exc()

        assert action in infoset.legal_actions

        return action

    # Made combine cards a method in class because it is basically a heuristic for definining what
    # the players optimal hand is. We have defined this optimal hand differently in V2.
    def combine_cards(self, hand):
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
        # Note: Updated pick chain to pick chains of pairs that are 3 long and trios that are 2 long.
        # This is what pick chain v2 accomplishes. weupdated this in thier implmentation because it is a bug.
        hand_list = card_str2list(hand)
        chains, hand_list = pick_chain_v2(hand_list, 1)
        comb['solo_chain'] = chains
        # 5. pick par_chain
        chains, hand_list = pick_chain_v2(hand_list, 2)
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

# Improvements over first agent:
#   1. Prioritze hard to play combinations ie long chains of singles and chains of pairs and triples.
#   2. Priorizes picking pair chains over solo chains if it results in removing more cards from the hand.


class RLCardAgentV2(RLCardAgent):
    def __init__(self, position):
        self.name = 'RLCardV2'
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

            # The rule of leading round
            if last_two_cards[0] == '' and last_two_cards[1] == '':
                chosen_action = None
                comb = self.combine_cards(hand_cards)

                # If there is a chain in the hand, play that chain as your leading move.
                # Play pair chains before solo chains and trio chains before all.
                chain_action = ([], None)
                for _, acs in comb.items():
                    for ac in acs:
                        the_type = CARD_TYPE[0][ac][0][0]
                        if 'solo_chain' in the_type:
                            action = getActionArr(ac)
                            if chain_action[1] == None:
                                chain_action = (action, 'solo_chain')
                        elif 'pair_chain' in the_type:
                            action = getActionArr(ac)
                            if chain_action[1] != 'trio_chain':
                                chain_action = (action, 'pair_chain')
                        elif 'trio_chain' in the_type:
                            action = getActionArr(ac)
                            return action

                if len(chain_action[0]) > 0:
                    return chain_action[0]

                # If there isn't a chain in the hand, play your combo with your min card in it.
                min_card = hand_cards[0]
                for _, acs in comb.items():
                    for ac in acs:
                        if min_card in ac:
                            action = getActionArr(ac)
            # The rule of following cards
            # Rule: choose the legal action with the lowest rank ie play your pair of 5's over your
            # pair of aces.
            else:
                the_type = CARD_TYPE[0][last_move][0][0]
                # this is a tuple of type (pair, straight etc) and its rank of that type
                chosen_action = ''
                rank = 1000

                # logic for choosing legal action with lowest rank
                for ac in infoset.legal_actions:
                    _ac = ac.copy()
                    for i, c in enumerate(_ac):
                        _ac[i] = EnvCard2RealCard[c]
                    _ac = ''.join(_ac)
                    if _ac != '' and the_type == CARD_TYPE[0][_ac][0][0]:
                        if int(CARD_TYPE[0][_ac][0][1]) < rank:
                            rank = int(CARD_TYPE[0][_ac][0][1])
                            chosen_action = _ac

                # if ther is an action, format it as an array
                if chosen_action != '':
                    action = [char for char in chosen_action]
                    for i, c in enumerate(action):
                        action[i] = RealCard2EnvCard[c]

                # else pass
                elif last_pid != 'landlord' and self.position != 'landlord':
                    action = []

            if action is None:
                action = random.choice(infoset.legal_actions)
        except:
            action = random.choice(infoset.legal_actions)
            import traceback
            traceback.print_exc()

        assert action in infoset.legal_actions

        return action

    # Pick pair chains which results in removing more cards from hand than by playing the solo straight
    # surrounding it.
    def pick_non_disruptive_pair_chains(self, hand_list):
        solo_chains, solo_handlist = pick_chain_v2(hand_list, 1)
        pair_chains, pair_handlist = pick_chain_v2(hand_list, 2)

        if sum(pair_handlist) < sum(solo_handlist):
            return (pair_chains, pair_handlist)

        # Return original handlist if there are no good pair chains to pick
        return ([], hand_list)

    # Difference from V1: Priorizes picking pair chains over solo chains if it
    # results in removing more cards from the hand.
    def combine_cards(self, hand):
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

        hand_list = card_str2list(hand)

        # The following two combos are the change from V1.
        # 4. Pick Non Disruptive Pair Chains
        chains, hand_list = self.pick_non_disruptive_pair_chains(hand_list)
        comb['pair_chain'] = chains

        # 5. Pick solo chains
        chains, hand_list = pick_chain_v2(hand_list, 1)
        comb['solo_chain'] = chains

        # update hand again with new hand list
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

        # 7. Add lowest solo and pairs to trios
        solosAndPairs = comb['solo'] + comb['pair']
        # sort by rank
        solosAndPairs.sort(key=lambda acs: int(CARD_TYPE[0][acs][0][1]), reverse=True)

        # reverse both lists so we can pop the lowest rank off the back of the list
        comb['pair'].reverse()
        comb['solo'].reverse()

        # add the lowest kickers to each each trio
        for i in range(len(comb['trio'])):
            if len(solosAndPairs) > 0:
                el = solosAndPairs.pop()

                # remove the kicker from the solo / pair list
                if len(el) == 2:
                    comb['pair'].pop()
                else: 
                    comb['solo'].pop()

                # sort the trio so that the lower rank cards come first
                new_acs = comb['trio'][i] + el
                new_ac = action_str2action_arr(new_acs)
                new_ac.sort()
                comb['trio'][i] = action_arr2action_str(new_ac)
            
        # put the lists back in their normal order
        comb['pair'].reverse()
        comb['solo'].reverse()
        return comb

# Helper for getting the action arr
# param ac: string of cards ex. 789TJ
# returns result arr which is arr of card indexs ex. [7, 8, 9, 10, 11]
def getActionArr(ac):
    chosen_action = ac
    result = [char for char in chosen_action]
    for i, c in enumerate(result):
        result[i] = RealCard2EnvCard[c]
    return result


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


def action_arr2action_str(ac):
    _ac = ac.copy()
    for i, c in enumerate(_ac):
        _ac[i] = EnvCard2RealCard[c]
    return ''.join(_ac)

def action_str2action_arr(ac):
    _ac = []
    for c in ac:
        _ac.append(RealCard2EnvCard[c])
    return _ac


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
                        hand_list[start +
                                  num] = int(hand_list[start+num]) - int(min(chain))
                    for _ in range(min_count):
                        chains.append(str_chain)
            add += len(chain)
    hand_list = [int(card) for card in hand_list]

    return (chains, hand_list)


def pick_chain_v2(hand_list, count):
    chains = []
    str_card = [card for card in INDEX]
    hand_list = [str(card) for card in hand_list]
    hand = ''.join(hand_list[:12])

    formatted_hand = hand

    # based on count, turn all cards that break chain to 0 so we can split up chains.lower()
    # example, one Jack breaks a pair chain for 2 10's, 1 J and 2 Q's.
    if count == 2:
        formatted_hand = hand.replace('1', '0')
    elif count == 3:
        formatted_hand = hand.replace('2', '0')
    elif count == 4:
        formatted_hand = hand.replace('4', '0')

    chain_list = formatted_hand.split('0')

    add = 0
    for index, chain in enumerate(chain_list):
        if len(chain) > 0:
            # pair chain needs to be 3 long, trio needs to be 2 long
            if len(chain) >= (5 / count):
                start = index + add
                min_count = int(min(chain)) // count
                if min_count != 0:
                    str_chain = ''
                    for num in range(len(chain)):
                        # push 2 jacks to result if pair chain
                        str_chain += (str_card[start+num] * count)
                        hand_list[start +
                                  num] = int(hand_list[start+num]) - int(min(chain))
                    for _ in range(min_count):
                        chains.append(str_chain)
            add += len(chain)

    hand_list = [int(card) for card in hand_list]

    return (chains, hand_list)
