import random

class ExpectimaxAgent():

    def __init__(self):
        self.name = 'Expectimax'

    def act(self, infoset):
        return random.choice(infoset.legal_actions)
