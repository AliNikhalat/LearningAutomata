import random


class MarkovianEnvironment:
    def __init__(self, action_number, state_probability, transition_probability):
        self.action_number = action_number
        self.state_probability = state_probability
        self.transition_probability = transition_probability

        self.current_state = 0

    def evaluate_action(self, action):
        random_envionment_number = MarkovianEnvironment.generate_random_number()

        evaluate = 0
        if random_envionment_number <= self.state_probability[self.current_state][action]:
            evaluate = 0  # give automata reward
        else:
            evaluate = 1  # punish automata

        self.__update_current_state()

        return evaluate

    def __update_current_state(self):
        random_transition_number = MarkovianEnvironment.generate_random_number()

        sum = 0
        for index, probability in enumerate(self.transition_probability[self.current_state]):
            sum += probability

            if random_transition_number <= sum:
                self.current_state = index
                return

    @staticmethod
    def generate_random_number():
        return random.uniform(0, 1)
