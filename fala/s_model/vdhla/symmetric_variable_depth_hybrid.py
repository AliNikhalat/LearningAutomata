import random
from numpy.random import choice
from matplotlib import pyplot as plt


class SymmetricVariableDepthHybrid:
    def __init__(self, action_number, reward_rate, penalty_rate):
        self.action_number = action_number
        self.action_probaility = [(1 / action_number)
                                  for _ in range(self.action_number)]

        self.reward_rate = reward_rate
        self.penalty_rate = penalty_rate
