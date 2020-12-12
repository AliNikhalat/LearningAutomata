import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from fala.vsla.variable_structure import *  # NOQA
from environment.environment import *  # NOQA

iteration_number = 1000

action_number = 2
reward_rate = 0
penalty_rate = 1

learning_automata = VariableStructure(
    action_number, reward_rate, penalty_rate)

action_probability_list = [0.8, 0.2]
environment = Environment(action_number, action_probability_list)


for _ in range(iteration_number):
    chosen_action = learning_automata.choose_action()
    evaluated_action = environment.evaluate_action(chosen_action)
    learning_automata.receive_environment_signal(evaluated_action)
    learning_automata.visualization_calculations()


learning_automata.visualize_entropy_data(iteration_number)
learning_automata.visualize_sum_probability_data(iteration_number)
learning_automata.visualize_action_probability_data(iteration_number)
