from matplotlib import pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../../../../..')))

from fala.s_model.vdhla.asymmetric_variable_depth_hybrid import *  # NOQA
from fala.p_model.vsla.variable_structure import *  # NOQA
from environment.state_dependent import *  # NOQA

iteration_number = 1000


action_number = 5
reward_rate = 0.01
penalty_rate = 0.001

a_vdhla5 = AsymmetricVariableDepthHybrid(
    action_number, [4 for _ in range(action_number)], reward_rate, penalty_rate)
vsla = VariableStructure(action_number, reward_rate, penalty_rate)

action_probability_list_1 = [0.8, 0.05, 0.05, 0.05, 0.05]
action_probability_list_2 = [0.8, 0.05, 0.05, 0.05, 0.05]

favorable_decrement_rate = 0.00002
unfavorable_increment_rate = 0.0002

environment_1 = StateDependentAEnvironment(
    action_number, action_probability_list_1, favorable_decrement_rate, unfavorable_increment_rate)
environment_2 = StateDependentAEnvironment(
    action_number, action_probability_list_2, favorable_decrement_rate, unfavorable_increment_rate)

for i in range(iteration_number):
    # VDHLA
    chosen_action_vdhla = 0
    if i != 0:
        chosen_action_vdhla = a_vdhla5.choose_action()
    else:
        chosen_action_vdhla = a_vdhla5.choose_random_action()

    evaluated_action_vdhla_2 = environment_1.evaluate_action(
        chosen_action_vdhla)
    a_vdhla5.receive_environment_signal(evaluated_action_vdhla_2)
    a_vdhla5.visualization_calculations()

    # VSLA
    chosen_action_vsla = vsla.choose_action()

    evaluated_action_vsla = environment_2.evaluate_action(chosen_action_vsla)
    vsla.receive_environment_signal(evaluated_action_vsla)
    vsla.visualization_calculations()

print('VDHLA 5 : TNR {}'.format(a_vdhla5.total_number_of_rewards[-1]))
print('VDHLA 5 : TNAS {}'.format(
    a_vdhla5.total_number_of_action_switching[-1]))

print('VSLA : TNR {}'.format(vsla.total_number_of_rewards[-1]))
print('VSLA : TNAS {}'.format(
    vsla.total_number_of_action_switching[-1]))

print(a_vdhla5.depth_vector)
print(environment_1.action_probability_list)
print(environment_2.action_probability_list)
