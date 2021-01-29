import numpy as np
import random


class Tsetlin:
    def __init__(self, state_number, action_number):
        self.state_number = state_number
        self.action_number = action_number

        self.chosen_action = 0
        self.chosen_action_depth_status = 0

        self.total_number_of_rewards = []
        self.total_number_of_action_switching = []

        self.action_selection_status = [[]
                                        for _ in range(self.action_number)]

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
            self.total_number_of_rewards.append(
                0 + self.total_number_of_rewards[-1] if len(self.total_number_of_rewards) > 0 else 0)
        else:
            self.__surprise_automata()

            self.total_number_of_rewards.append(
                1 + self.total_number_of_rewards[-1] if len(self.total_number_of_rewards) > 0 else 1)
            self.total_number_of_action_switching.append(
                0 + self.total_number_of_action_switching[-1] if len(self.total_number_of_action_switching) > 0 else 0)

        return

    # *****************************************************************************************
    def visualization_calculations(self):
        for action in range(self.action_number):
            if action == self.chosen_action:
                self.action_selection_status[action].append(
                    1 + self.action_selection_status[action][-1] if len(self.action_selection_status[action]) > 0 else 1)
            else:
                self.action_selection_status[action].append(
                    0 + self.action_selection_status[action][-1] if len(self.action_selection_status[action]) > 0 else 0)

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

            self.total_number_of_action_switching.append(
                1 + self.total_number_of_action_switching[-1] if len(self.total_number_of_action_switching) > 0 else 1)

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
