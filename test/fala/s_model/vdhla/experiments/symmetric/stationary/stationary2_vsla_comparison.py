from matplotlib import pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../../../../..')))

from fala.s_model.vdhla.symmetric_variable_depth_hybrid import *  # NOQA
from fala.p_model.vsla.variable_structure import *  # NOQA
from environment.environment import *  # NOQA

iteration_number = 1000

action_number = 5
reward_rate = 0.01
penalty_rate = 0.001

s_vdhla = SymmetricVariableDepthHybrid(
    action_number, 4, reward_rate, penalty_rate)

vsla = VariableStructure(action_number, reward_rate, penalty_rate)

action_probability_list = [0.8, 0.05, 0.05, 0.05, 0.05]
environment = Environment(action_number, action_probability_list)

favorable_vdhla_action_probability = []
favorable_vsla_action_probability = []


for i in range(iteration_number):
    # VDHLA
    chosen_action_vdhla = 0
    if i != 0:
        chosen_action_vdhla = s_vdhla.choose_action()
    else:
        chosen_action_vdhla = s_vdhla.choose_random_action()

    evaluated_action_vdhla = environment.evaluate_action(
        chosen_action_vdhla)
    s_vdhla.receive_environment_signal(evaluated_action_vdhla)
    s_vdhla.visualization_calculations()

    favorable_vdhla_action_probability.append(
        s_vdhla.get_action_selection_status(0)[-1] / (i + 1))

    # VSLA
    chosen_action = 0
    chosen_action = vsla.choose_action()

    evaluated_action = environment.evaluate_action(chosen_action)
    vsla.receive_environment_signal(evaluated_action)
    vsla.visualization_calculations()

    favorable_vsla_action_probability.append(
        vsla.get_action_selection_status(0)[-1] / (i + 1))

print('VDHLA 4 : TNR {}'.format(s_vdhla.total_number_of_rewards[-1]))
print('VDHLA 4 : TNAS {}'.format(
    s_vdhla.total_number_of_action_switching[-1]))

print('VSLA : TNR {}'.format(vsla.total_number_of_rewards[-1]))
print('VSLA : TNAS {}'.format(
    vsla.total_number_of_action_switching[-1]))

print('VDHLA : Depth {}'.format(s_vdhla.fsla_state_number))
# print(vsla.get_action_selection_status(0))


# Plots
x_values = [i for i in range(iteration_number)]

plt.plot(x_values, favorable_vdhla_action_probability,
         color='r', label='SVDHLA(N=4)')

plt.plot(x_values, favorable_vsla_action_probability,
         color='b', label='VSLA', linestyle='dashed')

plt.title('VSLA Comparison-Ex1.2.2.9')
plt.xlabel('iteration')
plt.ylabel('favorable')

plt.legend(loc="lower right")

plt.show()
