import numpy as np
import random


class Tsetlin:
    def __init__(self, state_number, action_number):
        self.state_number = state_number
        self.action_number = action_number

        self.state_status = [0 for i in range(self.state_number)]
        self.last_state = 0

    def choose_action(self):
        self.last_state = self.state_status.index(max(self.state_status))

    def choose_random_action(self):
        self.last_state = random.randint(0, self.action_number-1)
        self.state_number[self.last_state] = self.state_number - 1
        return self.last_state

    def receive_environment_signal(self, beta):
        if beta == 1:
            self.__punish_automata()
        else:
            self.__surprise_automata()

        return

    def __punish_automata(self):
        if self.state_status[self.action_number] < self.state_number-1:
            self.state_status[self.action_number] += 1
        else:
            self.state_status[self.action_number] = 0
            self.__choose_new_state()

        return

    def __surprise_automata(self):
        if self.state_status[self.action_number] > 1:
            self.state_status[self.action_number] -= 1

        return

    def __choose_new_state(self):
        random_state = random.choice(range(0, self.last_state) + range(
            self.last_state + 1, self.last_state))
        self.state_status[random_state] = self.state_number - 1
        return
