import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../..')))

from fala.s_model.vsla.variable_action_set import *  # NOQA
from environment.environment import *  # NOQA

iteration_number = 1000

action_number = 3
reward_rate = 0.1
penalty_rate = 0.001

learning_automata = VariableActionSet(
    action_number, reward_rate, penalty_rate)


action_probability_list = [0.5, 0.5, 0.7]
environment = Environment(action_number, action_probability_list)

for _ in range(iteration_number):
    chosen_action = learning_automata.choose_action([0, 1])
    evaluated_action = environment.evaluate_action(chosen_action)
    learning_automata.receive_environment_signal(evaluated_action)
    learning_automata.visualization_calculations()

learning_automata.visualize_sum_probability_data(iteration_number)
learning_automata.visualize_action_probability_data(iteration_number)
