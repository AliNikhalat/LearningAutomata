import random
from numpy.random import choice


class VariableStructure:
    def __init__(self, action_number, reward_rate, penalty_rate):
        self.action_number = action_number
        self.action_probaility = [(1 / action_number)
                                  for i in range(self.action_number)]

        self.reward_rate = reward_rate
        self.penalty_rate = penalty_rate

        self.last_action = 0

    def choose_action(self):
        self.probability_sum = sum(self.action_probaility)

        self.last_action = self.__roulette_wheel_selection(
            self.action_probaility)
        return self.last_action

    def receive_environment_signal(self, beta):
        if beta == 1:
            self.__punish_automata()
        else:
            self.__surprise_automata()

        # self.__rescale_probability_vector()

        return

    # *****************************************************************************************
    def __roulette_wheel_selection(self, probability_list):
        sum = 0
        random_number = random.uniform(0, 1)

        for index, probability in enumerate(probability_list):
            sum += probability

            if random_number <= sum:
                return index

        return probability_list.len()

    # *****************************************************************************************
    def __punish_automata(self):
        for action in self.action_probaility:
            if action != self.last_action:
                self.action_probaility[action] = (self.penalty_rate / (self.action_number - 1)) + (
                    1 - self.penalty_rate) * self.action_probaility[action]
            else:
                self.action_probaility[action] = (
                    1 - self.penalty_rate) * self.action_probaility[action]

        return

    # *****************************************************************************************
    def __surprise_automata(self):
        for action in self.action_probaility:
            if action != self.last_action:
                self.action_probaility[action] = (1 - self.reward_rate) * self.action_probaility[action]  # NOQA
            else:
                self.action_probaility[action] = self.action_probaility[action] + \
                    self.reward_rate * \
                    (1 - self.action_probaility[action])

        return

    # *****************************************************************************************
    # def __rescale_probability_vector(self):
    #     for action in self.action_probaility:
    #         self.action_probaility[action] = self.action_probaility[action] * \
    #             self.action_probaility

    #     return
