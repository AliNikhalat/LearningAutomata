import numpy
import random


class StateDependentAEnvironment:
    def __init__(self, action_number, action_probability_list, favorable_decrement_rate, unfavorable_increment_rate):
        self.action_number = action_number
        self.action_probability_list = action_probability_list

        self.favorable_decrement_rate = favorable_decrement_rate
        self.unfavorable_increment_rate = unfavorable_increment_rate

    # *****************************************************************************************
    def evaluate_action(self, action):
        random_number = random.uniform(0, 1)

        action_evaluation = 0
        if random_number <= self.action_probability_list[action]:
            action_evaluation = 0  # give automata reward
        else:
            action_evaluation = 1  # punish automata

        self.__update_action_probability(action)

        return action_evaluation

    # *****************************************************************************************
    def __update_action_probability(self, action):
        for index in range(len(self.action_probability_list)):
            if index != action:
                self.__update_unfavorable_action(index)
            else:
                self.__update_favorable_action(index)

        return

    # *****************************************************************************************
    def __update_favorable_action(self, index):
        if self.action_probability_list[index] - self.favorable_decrement_rate >= 0:
            self.action_probability_list[index] -= self.favorable_decrement_rate
        else:
            self.action_probability_list[index] = 0

        return

    # *****************************************************************************************
    def __update_unfavorable_action(self, index):
        if self.action_probability_list[index] + self.unfavorable_increment_rate <= 1:
            self.action_probability_list[index] += self.unfavorable_increment_rate
        else:
            self.action_probability_list[index] = 1

        return
