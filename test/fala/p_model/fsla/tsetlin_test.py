import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../..')))

from fala.p_model.fsla.tsetlin import *  # NOQA
from environment.environment import *  # NOQA

iteration_number = 100
state_number = 2
action_number = 2

learning_automata = Tsetlin(state_number, action_number)

action_probability_list = [0.8, 0.2]
environment = Environment(action_number, action_probability_list)

for i in range(iteration_number):
    chosen_action = 0
    if i != 0:
        chosen_action = learning_automata.choose_action()
    else:
        chosen_action = learning_automata.choose_random_action()

    evaluated_action = environment.evaluate_action(chosen_action)
    learning_automata.receive_environment_signal(evaluated_action)
