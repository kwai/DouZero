
import random
import json
from . import util


MCTS_DATA_FILE_NAME = 'C:\\Users\\Genevieve\\Desktop\\NEU\\Fall_2021\\ai\\DouZero\\douzero\\evaluation\\mcts_agent_data.txt'  # TODO
VERBOSE = True
TREE_TARGET_DEPTH = 5


def debug(*msgs):
    global VERBOSE
    if VERBOSE:
        result = ""
        for msg in msgs:
            result += str(msg)
        print(result)

class Node:
    def __init__(self):
        self.children = {} # action: resulting node
        self.num_team_cards = 0
        self.state = None

# **Modified** MCTS Agent:
# This agent is called upon each time an action is to be selected.
# Each time it is called upon, it is trained with MCTS a certain number of times
# Each game this agent is used, it will update the saved mcts data to be used
class MctsAgent:

    ######################################################################
    ########################## UTILITIES #################################
    ######################################################################

    def __init__(self, mcts_data_file_name=MCTS_DATA_FILE_NAME):
        self.name = 'MCTS'
        self.mcts_data_file_name = mcts_data_file_name

        # Stores data about each node and the quality of each possible action
        #self.mcts_data = {}

    # # Loads stores MCTS data from the save file and stores in self.mcts_data as a dictionary
    # def load_mcts_data(self):
    #     # with open(self.mcts_data_file_name, 'r') as file:
    #     #     mcts_data = file.read()
    #     # mcts_data_dict = json.loads(mcts_data)
    #     # return mcts_data_dict
    #     return {}
    #
    # # Updates the MCTS Data file with the updated contents of self.mcts_data
    # def save_mcts_data(self):
    #     with open(self.mcts_data_file_name, 'w') as file:
    #         file.write(json.dumps(self.mcts_data))

    ######################################################################
    ########################## ALGORITHM #################################
    ######################################################################
    ### Follwing pseudo code from https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/

    # Returns the best action from the curr_hand based on self.mcts_data
    # (The action that will lead to the child with the highest number of visits)
    def get_best_action(self, curr_hand, infoset):
        legal_moves = infoset.legal_actions
        if len(legal_moves[0]) == 0:
            return []

        # Picks the action that will lead to the child with the highest number of visits
        card_str_list = util.list2card_str_v2(curr_hand)
        best_action_tuple = util.get_best_actions(card_str_list)
        best_legal_moves = []
        for action in best_action_tuple:
            action = action[0]
            if legal_moves.__contains__(action):
                best_legal_moves.append(action)
        if len(best_legal_moves) == 0:
            if len(legal_moves) == 0:
                return []
            else:
                return legal_moves[0] # we didn't generate an action when there are legal moves
        else:
            return best_legal_moves[0] # MCTS implementation

    # New state = our cards after playing the action, the opponents' cards after playing their next best action
    # The opponents right now choose the first action from the heuritic narrowed-down list
    def build_new_state(self, action, infoset):
        curr_hand = infoset.player_hand_cards
        curr_hand_str = util.list2card_str_v2(curr_hand)
        action_str = util.list2card_str_v2(action)
        new_player_cards = util.getNextHandTupleArrV2(curr_hand_str, action_str) # TODO check


        new_all_player_cards = []
        for other_player_card_list in infoset.all_handcards.values():
            card_list_str = util.list2card_str_v2(other_player_card_list)
            best_action_tuple = util.get_best_actions(card_list_str)
            best_opponent_action = random.choice(best_action_tuple)# for opponnet, for now pick random suggested move
            if best_opponent_action == None or len(best_opponent_action) == 0:
                best_opponent_action = []
            else:
                best_opponent_action = best_opponent_action[0]
            action_str = util.list2card_str_v2(best_opponent_action)
            new_opponent_hand = util.getNextHandTupleArrV2(card_list_str, action_str)
            new_all_player_cards.append(new_opponent_hand)

        our_last_move = None

        state = (new_player_cards, new_all_player_cards, our_last_move)
        node = Node()
        node.state = state
        return node

    def fill_node(self, root_node, depth):
        if depth == 0:
            return root_node

        for child in root_node.children:

            self.fill_node(self, child, depth - 1)

    def create_tree(self, infoset, best_action_tuple):
        global TREE_TARGET_DEPTH
        root_node = Node()
        for action in best_action_tuple:
            action_move = action[0]
            child_node = self.build_new_state(action_move, infoset)
            root_node.children[tuple(action_move)] = child_node

        self.fill_node(root_node, TREE_TARGET_DEPTH)

        return root_node

    # TODO remove this eventually
    def handle_invalid_result(self, result, infoset):
        legal_moves = infoset.legal_actions
        if legal_moves.__contains__(result):
            return result

        if len(legal_moves) == 0:
            return []
        else:
            return random.choice(legal_moves)

    def choose_best_action(self, tree, infoset):
        lowest_num_cards_left = float('inf')
        best_action = None
        for action in tree.children:
            child_node = tree.children[action]
            if child_node.num_team_cards < lowest_num_cards_left:
                lowest_num_cards_left = child_node.num_team_cards
                best_action = action

        result = list(list(best_action))
        result = self.handle_invalid_result(result, infoset)

        legal_moves = infoset.legal_actions
        return result


    # Runs the given number of iterations PER action taken
    # Saves the updated data to the save file
    # Returns the best action
    def act(self, infoset, num_iterations=10):
        curr_hand = infoset.player_hand_cards
        legal_moves = infoset.legal_actions

        card_str_list = util.list2card_str_v2(curr_hand)
        best_action_tuple = util.get_best_actions(card_str_list)

        tree = self.create_tree(infoset, best_action_tuple)

        return self.choose_best_action(tree, infoset)

    # def traverse_round_tree(self, infoset):
    #     target_leaf_hand = None  # TODO
    #
    #     # while fully_expanded(node):
    #     #     node = best_uct(node)
    #     #
    #     # # in case no children are present / node is terminal
    #     # return pick_univisted(node.children) or node
    #
    #     return target_leaf_hand
    #
    # def round_rollout_policy(self, legal_actions_from_hand):
    #     return random.choice(legal_actions_from_hand)
    #
    # def rollout_round(self, target_leaf_hand):
    #     simulation_result = None  # TODO
    #     #     while non_terminal(node):
    #     #         node = rollout_policy(node)
    #     #     return result(node)
    #     return simulation_result
    #
    # def backpropagate(self, target_leaf_hand, simulation_result):
    #     # TODO update the mcts data??
    #
    #     #     if is_root(node) return
    #     #     node.stats = update_stats(node, result)
    #     #     backpropagate(node.parent)
    #     return None  # TODO nothing to return
