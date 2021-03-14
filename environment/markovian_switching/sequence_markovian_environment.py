import random


class SequenceMarkovianEnvironment:
    def __init__(self, action_number, state_probability, transition_probability, sequence_number):
        self.action_number = action_number
        self.state_probability = state_probability
        self.transition_probability = transition_probability
        self.sequence_number = sequence_number

        self.sequence = [0]
        self.episode = 0

        self.__create_seqeunce()
        print(len(self.sequence))

    def evaluate_action(self, action):
        random_envionment_number = SequenceMarkovianEnvironment.generate_random_number()

        evaluate = 0
        if random_envionment_number <= self.state_probability[self.sequence[self.episode]][action]:
            evaluate = 0  # give automata reward
        else:
            evaluate = 1  # punish automata

        return evaluate

    def goto_next_episode(self):
        self.episode += 1

        return

    def __create_seqeunce(self):
        for _ in range(self.sequence_number - 1):
            random_transition_number = SequenceMarkovianEnvironment.generate_random_number()

            sum = 0
            for index, probability in enumerate(self.transition_probability[self.sequence[-1]]):
                sum += probability

                if random_transition_number <= sum:
                    self.sequence.append(index)
                    break

    @staticmethod
    def generate_random_number():
        return random.uniform(0, 1)
