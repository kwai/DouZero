import random

class ExpectimaxAgent():
    def __init__(self, depth=2):
        self.name = 'Expectimax'
        self.depth = depth


    def act(self, infoset):
        actions = infoset.legal_actions
        my_role = infoset.player_position # one of 'landlord', 'landlord_up', 'landlord_down'

        return None

"""
NEED:
Eval func
Predict probability other player will make a certain play.
"""
