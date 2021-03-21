import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../..')))

from fala.s_model.vdhla.symmetric_variable_depth_hybrid import *  # NOQA
from fala.p_model.vsla.variable_structure import *  # NOQA
from environment.static import *  # NOQA
from environment.markovian_switching import *  # NOQA
from environment.state_dependent.state_dependent_a_environment import *  # NOQA

iteration_number = 1000

action_number = 2
state_number = 2
reward_rate = 0.01
penalty_rate = 0

learning_automata = SymmetricVariableDepthHybrid(
    action_number, state_number, reward_rate, penalty_rate)

action_probability_list = [0.2, 0.8]
# environment = Environment(action_number, action_probability_list)
favorable_decrement_rate = 0.0001
unfavorable_increment_rate = 0.0001
environment = StateDependentAEnvironment(
    action_number, action_probability_list, favorable_decrement_rate, unfavorable_increment_rate)

for i in range(iteration_number):
    chosen_action_vdhla = 0
    if i != 0:
        chosen_action_vdhla = learning_automata.choose_action()
    else:
        chosen_action_vdhla = learning_automata.choose_random_action()

    evaluated_action_vdhla = environment.evaluate_action(chosen_action_vdhla)
    learning_automata.receive_environment_signal(evaluated_action_vdhla)


print(learning_automata.fsla_state_number)
print(environment.action_probability_list)
# print(learning_automata.total_number_of_rewards)
