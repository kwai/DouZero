import random

class MctsAgent():

    def __init__(self):
        self.name = 'MCTS'

    def act(self, infoset):
        return random.choice(infoset.legal_actions)
