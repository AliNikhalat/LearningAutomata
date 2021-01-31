import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../..')))

from fala.s_model.vdhla.asymmetric_variable_depth_hybrid import *  # NOQA
from environment.environment import *  # NOQA


iteration_number = 20000

action_number = 2
state_number = 2
reward_rate = 0.1
penalty_rate = 0.01

learning_automata = AsymmetricVariableDepthHybrid(
    action_number, state_number, reward_rate, penalty_rate)

action_probability_list = [0.5, 0.5]
environment = Environment(action_number, action_probability_list)

for i in range(iteration_number):
    chosen_action_vdhla = 0
    if i != 0:
        chosen_action_vdhla = learning_automata.choose_action()
    else:
        chosen_action_vdhla = learning_automata.choose_random_action()

    evaluated_action_vdhla = environment.evaluate_action(chosen_action_vdhla)
    learning_automata.receive_environment_signal(evaluated_action_vdhla)


print(learning_automata.choose_action)
