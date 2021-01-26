import numpy as np
import random


class Tsetlin:
    def __init__(self, state_number, action_number):
        self.state_number = state_number
        self.action_number = action_number

        self.chosen_action = 0
        self.chosen_action_depth_status = 0

    # *****************************************************************************************
    def choose_action(self):
        return self.chosen_action

    # *****************************************************************************************
    def choose_random_action(self):
        # Choose random action for first running
        self.chosen_action = random.randint(0, self.action_number-1)
        self.chosen_action_depth_status = 1

        return self.chosen_action

    # *****************************************************************************************
    def receive_environment_signal(self, beta):
        if beta == 1:
            self.__punish_automata()
        else:
            self.__surprise_automata()

        return

    # *****************************************************************************************
    def __surprise_automata(self):
        if self.chosen_action_depth_status < self.state_number:
            self.chosen_action_depth_status += 1

        return

    # *****************************************************************************************
    def __punish_automata(self):
        if self.chosen_action_depth_status > 1:
            self.chosen_action_depth_status -= 1
        else:
            self.__choose_new_action_clockwise()

        return

    # *****************************************************************************************
    def __choose_new_action_randomly(self):
        random_state = random.choice(range(0, self.chosen_action) + range(
            self.chosen_action + 1, self.chosen_action))

        self.chosen_action = random_state
        self.chosen_action_depth_status = 1

        return

    # *****************************************************************************************
    def __choose_new_action_clockwise(self):
        if self.chosen_action < self.action_number - 1:
            self.chosen_action += 1
        else:
            self.chosen_action = 0

        self.chosen_action_depth_status = 1

        return
