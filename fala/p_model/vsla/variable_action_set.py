import random
from numpy.random import choice


class VariableActionSet:
    def __init__(self, action_number, reward_rate, penalty_rate):
        self.action_number = action_number
        self.action_probaility = [(1 / action_number)
                                  for i in range(self.action_number)]

        self.reward_rate = reward_rate
        self.penalty_rate = penalty_rate

        self.last_action = 0
        self.last_sub_action_list = None
        self.sub_action_probability = None
        self.sub_action_probability_sum = 0

    # *****************************************************************************************
    def choose_action(self, sub_action_list):
        self.sub_action_probability_sum = 0
        self.sub_action_probability = [0 for i in range(self.action_number)]
        self.last_sub_action_list = sub_action_list

        for action in sub_action_list:
            self.sub_action_probability_sum += self.action_probaility[action]

        for action in sub_action_list:
            self.sub_action_probability[action] = self.action_probaility[action] / \
                self.sub_action_probability_sum

        # self.last_action = self.sub_action_probability.index(
        #     max(self.sub_action_probability))

        self.last_action = VariableActionSet.select_index_by_probability(
            self.sub_action_probability)

        return self.last_action

    # *****************************************************************************************
    def receive_environment_signal(self, beta):
        if beta == 1:
            self.__punish_automata()
        else:
            self.__surprise_automata()

        self.__rescale_probability_vector()

        return

    # *****************************************************************************************
    @staticmethod
    def select_index_by_probability(probability_list):
        return choice(range(len(probability_list)), p=probability_list)

    # *****************************************************************************************
    def __punish_automata(self):
        for action in self.sub_action_probability:
            if action != self.last_action:
                self.sub_action_probability[action] = (self.penalty_rate / (self.action_number - 1)) + (
                    1 - self.penalty_rate) * self.sub_action_probability[action]
            else:
                self.sub_action_probability[action] = (
                    1 - self.penalty_rate) * self.sub_action_probability[action]

        return

    # *****************************************************************************************
    def __surprise_automata(self):
        for action in self.sub_action_probability:
            if action != self.last_action:
                self.sub_action_probability[action] = (1 - self.reward_rate) * self.sub_action_probability[action]  # NOQA
            else:
                self.sub_action_probability[action] = self.sub_action_probability[action] + \
                    self.reward_rate * \
                    (1 - self.sub_action_probability[action])

        return

    # *****************************************************************************************
    def __rescale_probability_vector(self):
        for action in self.sub_action_probability:
            self.action_probaility[action] = self.sub_action_probability[action] * \
                self.sub_action_probability_sum

        return
