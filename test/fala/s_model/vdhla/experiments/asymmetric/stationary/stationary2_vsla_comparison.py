from matplotlib import pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../../../../..')))

from fala.s_model.vdhla.asymmetric_variable_depth_hybrid import *  # NOQA
from fala.p_model.vsla.variable_structure import *  # NOQA
from environment.static.static_environment import *  # NOQA

iteration_number = 1000

action_number = 5

reward_rate_1 = 0.1
penalty_rate_1 = 0.01

reward_rate_2 = 0.01
penalty_rate_2 = 0.001

a_vdhla_1 = AsymmetricVariableDepthHybrid(
    action_number, [4 for _ in range(action_number)], reward_rate_1, penalty_rate_1)
a_vdhla_2 = AsymmetricVariableDepthHybrid(
    action_number, [4 for _ in range(action_number)], reward_rate_2, penalty_rate_2)

vsla_1 = VariableStructure(action_number, reward_rate_1, penalty_rate_1)
vsla_2 = VariableStructure(action_number, reward_rate_2, penalty_rate_2)

action_probability_list = [0.8, 0.05, 0.05, 0.05, 0.05]
environment = StaticEnvironment(action_number, action_probability_list)

favorable_vdhla_action_probability_1 = []
favorable_vsla_action_probability_1 = []

favorable_vdhla_action_probability_2 = []
favorable_vsla_action_probability_2 = []


for i in range(iteration_number):
    # VDHLA_1
    chosen_action_vdhla_1 = 0
    if i != 0:
        chosen_action_vdhla_1 = a_vdhla_1.choose_action()
    else:
        chosen_action_vdhla_1 = a_vdhla_1.choose_random_action()

    evaluated_action_vdhla_1 = environment.evaluate_action(
        chosen_action_vdhla_1)
    a_vdhla_1.receive_environment_signal(evaluated_action_vdhla_1)
    a_vdhla_1.visualization_calculations()

    favorable_vdhla_action_probability_1.append(
        a_vdhla_1.get_action_selection_status(0)[-1] / (i + 1))

    # VDHLA_2
    chosen_action_vdhla_2 = 0
    if i != 0:
        chosen_action_vdhla_2 = a_vdhla_2.choose_action()
    else:
        chosen_action_vdhla_2 = a_vdhla_2.choose_random_action()

    evaluated_action_vdhla_2 = environment.evaluate_action(
        chosen_action_vdhla_2)
    a_vdhla_2.receive_environment_signal(evaluated_action_vdhla_2)
    a_vdhla_2.visualization_calculations()

    favorable_vdhla_action_probability_2.append(
        a_vdhla_2.get_action_selection_status(0)[-1] / (i + 1))

    # VSLA_1
    chosen_action_vsla_1 = 0
    chosen_action_vsla_1 = vsla_1.choose_action()

    evaluated_action_vsla_1 = environment.evaluate_action(chosen_action_vsla_1)
    vsla_1.receive_environment_signal(evaluated_action_vsla_1)
    vsla_1.visualization_calculations()

    favorable_vsla_action_probability_1.append(
        vsla_1.get_action_selection_status(0)[-1] / (i + 1))

    # VSLA_2
    chosen_action_vsla_2 = 0
    chosen_action_vsla_2 = vsla_2.choose_action()

    evaluated_action_vsla_2 = environment.evaluate_action(chosen_action_vsla_2)
    vsla_2.receive_environment_signal(evaluated_action_vsla_2)
    vsla_2.visualization_calculations()

    favorable_vsla_action_probability_2.append(
        vsla_2.get_action_selection_status(0)[-1] / (i + 1))

print('###########One############')
print('VDHLA 4 : TNR {}'.format(a_vdhla_1.total_number_of_rewards[-1]))
print('VDHLA 4 : TNAS {}'.format(
    a_vdhla_1.total_number_of_action_switching[-1]))

print('VSLA : TNR {}'.format(vsla_1.total_number_of_rewards[-1]))
print('VSLA : TNAS {}'.format(
    vsla_1.total_number_of_action_switching[-1]))

print('###########Two############')
print('VDHLA 4 : TNR {}'.format(a_vdhla_2.total_number_of_rewards[-1]))
print('VDHLA 4 : TNAS {}'.format(
    a_vdhla_2.total_number_of_action_switching[-1]))

print('VSLA : TNR {}'.format(vsla_2.total_number_of_rewards[-1]))
print('VSLA : TNAS {}'.format(
    vsla_2.total_number_of_action_switching[-1]))


# Plots
x_values = [i for i in range(iteration_number)]


plt.plot(x_values, favorable_vdhla_action_probability_1,
         color='r', label='Asymmetric VDHLA(Ex1.2.3.8)')
plt.plot(x_values, favorable_vsla_action_probability_1,
         color='r', label='VSLA(Ex1.2.3.8)', linestyle='dashed')

plt.plot(x_values, favorable_vdhla_action_probability_2,
         color='b', label='Asymmetric VDHLA(Ex1.2.3.9)')
plt.plot(x_values, favorable_vsla_action_probability_2,
         color='b', label='VSLA(Ex1.2.3.9)', linestyle='dashed')

plt.title('VSLA Comparison-Ex1.2.3.8,Ex1.2.3.9')
plt.xlabel('Iteration')
plt.ylabel('Probability of choosing favorable action(a1)')

plt.legend(loc="lower right")

plt.show()
