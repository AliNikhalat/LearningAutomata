import numpy
import random


class Environment:
    def __init__(self, action_number, action_probability_list):
        self.action_number = action_number
        self.action_probability_list = action_probability_list

    # *****************************************************************************************
    def evaluate_action(self, action):
        random_number = random.uniform(0, 1)

        if random_number <= self.action_probability_list[action]:
            return 0  # give automata reward
        else:
            return 1  # punish automata
