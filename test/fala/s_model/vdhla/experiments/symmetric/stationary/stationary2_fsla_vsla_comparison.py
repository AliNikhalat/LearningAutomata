from matplotlib import pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../../../../..')))

from fala.s_model.vdhla.symmetric_variable_depth_hybrid import *  # NOQA
from fala.p_model.fsla.tsetlin import *  # NOQA
from fala.p_model.vsla.variable_structure import *  # NOQA
from environment.static.static_environment import *  # NOQA

iteration_number = 1000

reward_rate = 0.01
penalty_rate = 0

s_vdhla5_2 = SymmetricVariableDepthHybrid(
    2, 5, reward_rate, penalty_rate)
tsetlin_5state_2 = Tsetlin(5, 2)
vsla_2 = VariableStructure(2, reward_rate, penalty_rate)

s_vdhla5_5 = SymmetricVariableDepthHybrid(
    5, 5, reward_rate, penalty_rate)
tsetlin_5state_5 = Tsetlin(5, 5)
vsla_5 = VariableStructure(5, reward_rate, penalty_rate)

s_vdhla5_9 = SymmetricVariableDepthHybrid(
    9, 5, reward_rate, penalty_rate)
tsetlin_5state_9 = Tsetlin(5, 9)
vsla_9 = VariableStructure(9, reward_rate, penalty_rate)

action_probability_list_9 = [0.8, 0.02,
                             0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
action_probability_list_5 = [0.8, 0.05, 0.05, 0.05, 0.05]
action_probability_list_2 = [0.8, 0.2]

environment_2 = StaticEnvironment(2, action_probability_list_2)
environment_5 = StaticEnvironment(5, action_probability_list_5)
environment_9 = StaticEnvironment(9, action_probability_list_9)

favorable5_vdhla_action_probability_2 = []
favorable5_action_probability_2 = []
favorable_vsla_action_probability_2 = []

favorable5_vdhla_action_probability_5 = []
favorable5_action_probability_5 = []
favorable_vsla_action_probability_5 = []

favorable5_vdhla_action_probability_9 = []
favorable5_action_probability_9 = []
favorable_vsla_action_probability_9 = []

for i in range(iteration_number):
    # 2 Action
    # vdhla 5 state
    chosen_action_5_vdhla_2 = 0
    if i != 0:
        chosen_action_5_vdhla_2 = s_vdhla5_2.choose_action()
    else:
        chosen_action_5_vdhla_2 = s_vdhla5_2.choose_random_action()

    evaluated_action_vdhla5_2 = environment_2.evaluate_action(
        chosen_action_5_vdhla_2)
    s_vdhla5_2.receive_environment_signal(evaluated_action_vdhla5_2)
    s_vdhla5_2.visualization_calculations()

    favorable5_vdhla_action_probability_2.append(
        s_vdhla5_2.get_action_selection_status(0)[-1] / (i + 1))

    # FSLA 5 state
    chosen_action_5_2 = 0
    if i != 0:
        chosen_action_5_2 = tsetlin_5state_2.choose_action()
    else:
        chosen_action_5_2 = tsetlin_5state_2.choose_random_action()

    evaluated_action_5_2 = environment_2.evaluate_action(chosen_action_5_2)
    tsetlin_5state_2.receive_environment_signal(evaluated_action_5_2)
    tsetlin_5state_2.visualization_calculations()

    favorable5_action_probability_2.append(
        tsetlin_5state_2.get_action_selection_status(0)[-1] / (i + 1))

    # VSLA
    chosen_action_vsla_2 = vsla_2.choose_action()

    evaluated_action_vsla_2 = environment_2.evaluate_action(
        chosen_action_vsla_2)
    vsla_2.receive_environment_signal(evaluated_action_vsla_2)
    vsla_2.visualization_calculations()

    favorable_vsla_action_probability_2.append(
        vsla_2.get_action_selection_status(0)[-1] / (i + 1))

    # 5 Action
    # vdhla 5 state
    chosen_action_5_vdhla_5 = 0
    if i != 0:
        chosen_action_5_vdhla_5 = s_vdhla5_5.choose_action()
    else:
        chosen_action_5_vdhla_5 = s_vdhla5_5.choose_random_action()

    evaluated_action_vdhla5_5 = environment_5.evaluate_action(
        chosen_action_5_vdhla_5)
    s_vdhla5_5.receive_environment_signal(evaluated_action_vdhla5_5)
    s_vdhla5_5.visualization_calculations()

    favorable5_vdhla_action_probability_5.append(
        s_vdhla5_5.get_action_selection_status(0)[-1] / (i + 1))

    # FSLA 5 state
    chosen_action_5_5 = 0
    if i != 0:
        chosen_action_5_5 = tsetlin_5state_5.choose_action()
    else:
        chosen_action_5_5 = tsetlin_5state_5.choose_random_action()

    evaluated_action_5_5 = environment_5.evaluate_action(chosen_action_5_5)
    tsetlin_5state_5.receive_environment_signal(evaluated_action_5_5)
    tsetlin_5state_5.visualization_calculations()

    favorable5_action_probability_5.append(
        tsetlin_5state_5.get_action_selection_status(0)[-1] / (i + 1))

    # VSLA
    chosen_action_vsla_5 = vsla_5.choose_action()

    evaluated_action_vsla_5 = environment_5.evaluate_action(
        chosen_action_vsla_5)
    vsla_5.receive_environment_signal(evaluated_action_vsla_5)
    vsla_5.visualization_calculations()

    favorable_vsla_action_probability_5.append(
        vsla_5.get_action_selection_status(0)[-1] / (i + 1))

    # 9 Action
    # vdhla 5 state
    chosen_action_5_vdhla_9 = 0
    if i != 0:
        chosen_action_5_vdhla_9 = s_vdhla5_9.choose_action()
    else:
        chosen_action_5_vdhla_9 = s_vdhla5_9.choose_random_action()

    evaluated_action_vdhla5_9 = environment_9.evaluate_action(
        chosen_action_5_vdhla_9)
    s_vdhla5_9.receive_environment_signal(evaluated_action_vdhla5_9)
    s_vdhla5_9.visualization_calculations()

    favorable5_vdhla_action_probability_9.append(
        s_vdhla5_9.get_action_selection_status(0)[-1] / (i + 1))

    # FSLA 5 state
    chosen_action_5_9 = 0
    if i != 0:
        chosen_action_5_9 = tsetlin_5state_9.choose_action()
    else:
        chosen_action_5_9 = tsetlin_5state_9.choose_random_action()

    evaluated_action_5_9 = environment_9.evaluate_action(chosen_action_5_9)
    tsetlin_5state_9.receive_environment_signal(evaluated_action_5_9)
    tsetlin_5state_9.visualization_calculations()

    favorable5_action_probability_9.append(
        tsetlin_5state_9.get_action_selection_status(0)[-1] / (i + 1))

    # VSLA
    chosen_action_vsla_9 = vsla_9.choose_action()

    evaluated_action_vsla_9 = environment_9.evaluate_action(
        chosen_action_vsla_9)
    vsla_9.receive_environment_signal(evaluated_action_vsla_9)
    vsla_9.visualization_calculations()

    favorable_vsla_action_probability_9.append(
        vsla_9.get_action_selection_status(0)[-1] / (i + 1))


print('VDHLA 2 : TNR {}'.format(s_vdhla5_2.total_number_of_rewards[-1]))
print('VDHLA 2 : TNAS {}'.format(
    s_vdhla5_2.total_number_of_action_switching[-1]))
print('FSLA 2 : TNR {}'.format(tsetlin_5state_2.total_number_of_rewards[-1]))
print('FSLA 2 : TNAS {}'.format(
    tsetlin_5state_2.total_number_of_action_switching[-1]))
print('VSLA 2 : TNR {}'.format(vsla_2.total_number_of_rewards[-1]))
print('VSLA 2 : TNAS {}'.format(
    vsla_2.total_number_of_action_switching[-1]))

print('*******************************************************************')

print('VDHLA 5 : TNR {}'.format(s_vdhla5_5.total_number_of_rewards[-1]))
print('VDHLA 5 : TNAS {}'.format(
    s_vdhla5_5.total_number_of_action_switching[-1]))
print('FSLA 5 : TNR {}'.format(tsetlin_5state_5.total_number_of_rewards[-1]))
print('FSLA 5 : TNAS {}'.format(
    tsetlin_5state_5.total_number_of_action_switching[-1]))
print('VSLA 5 : TNR {}'.format(vsla_5.total_number_of_rewards[-1]))
print('VSLA 5 : TNAS {}'.format(
    vsla_5.total_number_of_action_switching[-1]))

print('*******************************************************************')

print('VDHLA 9 : TNR {}'.format(s_vdhla5_9.total_number_of_rewards[-1]))
print('VDHLA 9 : TNAS {}'.format(
    s_vdhla5_9.total_number_of_action_switching[-1]))
print('FSLA 9 : TNR {}'.format(tsetlin_5state_9.total_number_of_rewards[-1]))
print('FSLA 9 : TNAS {}'.format(
    tsetlin_5state_9.total_number_of_action_switching[-1]))
print('VSLA 9 : TNR {}'.format(vsla_9.total_number_of_rewards[-1]))
print('VSLA 9 : TNAS {}'.format(
    vsla_9.total_number_of_action_switching[-1]))

print('*******************************************************************')

# Plots
x_values = [i for i in range(iteration_number)]

plt.plot(x_values, favorable5_vdhla_action_probability_2,
         color='r', label='Symmetric VDHLA(K=2)')
plt.plot(x_values, favorable5_action_probability_2,
         color='r', label='FSLA(K=2)', linestyle='dashed')
plt.plot(x_values, favorable_vsla_action_probability_2,
         color='r', label='VSLA(K=2)', linestyle='dotted')

plt.plot(x_values, favorable5_vdhla_action_probability_5,
         color='b', label='Symmetric VDHLA(K=5)')
plt.plot(x_values, favorable5_action_probability_5,
         color='b', label='FSLA(K=5)', linestyle='dashed')
plt.plot(x_values, favorable_vsla_action_probability_5,
         color='b', label='VSLA(K=5)', linestyle='dotted')

plt.plot(x_values, favorable5_vdhla_action_probability_9,
         color='g', label='Symmetric VDHLA(K=9)')
plt.plot(x_values, favorable5_action_probability_9,
         color='g', label='FSLA(K=9)', linestyle='dashed')
plt.plot(x_values, favorable_vsla_action_probability_9,
         color='g', label='VSLA(K=9)', linestyle='dotted')

plt.title('FSLA, VSLA Comparison-Ex1.2.3')
plt.xlabel('Iteration')
plt.ylabel('Probability of choosing favorable action(a1)')

plt.legend(loc="lower right")

plt.show()
