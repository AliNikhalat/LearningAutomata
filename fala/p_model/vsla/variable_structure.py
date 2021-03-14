import random
import math
from numpy.random import choice
from matplotlib import pyplot as plt


class VariableStructure:
    def __init__(self, action_number, reward_rate, penalty_rate):
        self.action_number = action_number
        self.action_probaility = [(1 / action_number)
                                  for i in range(self.action_number)]

        self.reward_rate = reward_rate
        self.penalty_rate = penalty_rate

        self.last_action = 0

        self.entropy = []
        self.sum_probability = []
        self.visual_action_probability = [[]
                                          for _ in range(self.action_number)]

        self.total_number_of_rewards = []
        self.total_number_of_action_switching = []

        self.action_selection_status = [[]
                                        for _ in range(self.action_number)]

    # *****************************************************************************************
    def choose_action(self):
        previous_last_action = self.last_action

        self.last_action = self.__roulette_wheel_selection(
            self.action_probaility)

        if len(self.total_number_of_action_switching) > 0:
            if previous_last_action != self.last_action:
                self.total_number_of_action_switching.append(
                    1 + self.total_number_of_action_switching[-1])
            else:
                self.total_number_of_action_switching.append(
                    0 + self.total_number_of_action_switching[-1])
        else:
            self.total_number_of_action_switching.append(0)

        return self.last_action

    # *****************************************************************************************
    def receive_environment_signal(self, beta):
        if beta == 1:
            self.__punish_automata()
            self.total_number_of_rewards.append(
                0 + self.total_number_of_rewards[-1] if len(self.total_number_of_rewards) > 0 else 0)
        else:
            self.__surprise_automata()
            self.total_number_of_rewards.append(
                1 + self.total_number_of_rewards[-1] if len(self.total_number_of_rewards) > 0 else 1)

        return

    # *****************************************************************************************
    def visualization_calculations(self):
        self.entropy.append(self.__calculate_entropy())
        self.sum_probability.append(sum(self.action_probaility))

        for action in range(self.action_number):
            self.visual_action_probability[action].append(
                self.action_probaility[action])

            if action == self.last_action:
                self.action_selection_status[action].append(
                    1 + self.action_selection_status[action][-1] if len(self.action_selection_status[action]) > 0 else 1)
            else:
                self.action_selection_status[action].append(
                    0 + self.action_selection_status[action][-1] if len(self.action_selection_status[action]) > 0 else 0)

    # *****************************************************************************************

    def visualize_entropy_data(self, iteration_number):
        x_values = [i for i in range(iteration_number)]

        plt.title('Entropy')
        plt.xlabel('iteration')
        plt.ylabel('entropy')

        plt.plot(x_values, self.entropy)
        plt.show()

    # *****************************************************************************************
    def visualize_sum_probability_data(self, iteration_number):
        x_values = [i for i in range(iteration_number)]

        plt.title('Sum Probability')
        plt.xlabel('iteration')
        plt.ylabel('sum')

        plt.plot(x_values, self.sum_probability)
        plt.show()

    # *****************************************************************************************
    def visualize_action_probability_data(self, iteration_number):
        x_values = [i for i in range(iteration_number)]

        plt.plot(
            x_values, self.visual_action_probability[0], color='r', label='action 0')
        plt.plot(
            x_values, self.visual_action_probability[1], color='b', label='action 1')

        plt.title('Action Probability')
        plt.xlabel('iteration')
        plt.ylabel('probability')

        plt.legend(loc="upper left")

        plt.show()

     # *****************************************************************************************
    @property
    def get_total_number_of_rewards(self):
        return self.total_number_of_rewards

    # *****************************************************************************************
    @property
    def get_total_number_of_action_switching(self):
        return self.total_number_of_action_switching

    # *****************************************************************************************
    def get_action_selection_status(self, action_number):
        return self.action_selection_status[action_number]

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
        for action in range(self.action_number):
            if action != self.last_action:
                self.action_probaility[action] = (self.penalty_rate / (self.action_number - 1)) + (
                    1 - self.penalty_rate) * self.action_probaility[action]
            else:
                self.action_probaility[action] = (
                    1 - self.penalty_rate) * self.action_probaility[action]

        return

    # *****************************************************************************************
    def __surprise_automata(self):
        for action in range(self.action_number):
            if action != self.last_action:
                self.action_probaility[action] = (1 - self.reward_rate) * self.action_probaility[action]  # NOQA
            else:
                self.action_probaility[action] = self.action_probaility[action] + \
                    self.reward_rate * \
                    (1 - self.action_probaility[action])

        return

    # *****************************************************************************************
    def __calculate_entropy(self):
        return -1 * sum([p * math.log(p, self.action_number) for p in self.action_probaility if p != 0])
