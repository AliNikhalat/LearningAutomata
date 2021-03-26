from matplotlib import pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../../../../..')))

from fala.s_model.vdhla.symmetric_variable_depth_hybrid import *  # NOQA
from fala.p_model.fsla.tsetlin import *  # NOQA
from environment.state_dependent import *  # NOQA

iteration_number = 1000000


action_number = 2
reward_rate = 0.01
penalty_rate = 0

s_vdhla1 = SymmetricVariableDepthHybrid(
    action_number, 1, reward_rate, penalty_rate)
s_vdhla3 = SymmetricVariableDepthHybrid(
    action_number, 3, reward_rate, penalty_rate)
s_vdhla5 = SymmetricVariableDepthHybrid(
    action_number, 5, reward_rate, penalty_rate)
s_vdhla7 = SymmetricVariableDepthHybrid(
    action_number, 7, reward_rate, penalty_rate)

tsetlin_1state = Tsetlin(1, action_number)
tsetlin_3state = Tsetlin(3, action_number)
tsetlin_5state = Tsetlin(5, action_number)
tsetlin_7state = Tsetlin(7, action_number)


favorable1_vdhla_action_probability = []
favorable3_vdhla_action_probability = []
favorable5_vdhla_action_probability = []
favorable7_vdhla_action_probability = []

favorable1_action_probability = []
favorable3_action_probability = []
favorable5_action_probability = []
favorable7_action_probability = []

get_status = 0


action_probability_list_1 = [0.9, 0.1]
action_probability_list_2 = [0.9, 0.1]
action_probability_list_3 = [0.9, 0.1]
action_probability_list_4 = [0.9, 0.1]
action_probability_list_5 = [0.9, 0.1]
action_probability_list_6 = [0.9, 0.1]
action_probability_list_7 = [0.9, 0.1]
action_probability_list_8 = [0.9, 0.1]

favorable_decrement_rate = 0.00002
unfavorable_increment_rate = 0.0002

environment_1 = StateDependentAEnvironment(
    action_number, action_probability_list_1, favorable_decrement_rate, unfavorable_increment_rate)
environment_2 = StateDependentAEnvironment(
    action_number, action_probability_list_2, favorable_decrement_rate, unfavorable_increment_rate)
environment_3 = StateDependentAEnvironment(
    action_number, action_probability_list_3, favorable_decrement_rate, unfavorable_increment_rate)
environment_4 = StateDependentAEnvironment(
    action_number, action_probability_list_4, favorable_decrement_rate, unfavorable_increment_rate)
environment_5 = StateDependentAEnvironment(
    action_number, action_probability_list_5, favorable_decrement_rate, unfavorable_increment_rate)
environment_6 = StateDependentAEnvironment(
    action_number, action_probability_list_6, favorable_decrement_rate, unfavorable_increment_rate)
environment_7 = StateDependentAEnvironment(
    action_number, action_probability_list_7, favorable_decrement_rate, unfavorable_increment_rate)
environment_8 = StateDependentAEnvironment(
    action_number, action_probability_list_8, favorable_decrement_rate, unfavorable_increment_rate)

for i in range(iteration_number):
    # vdhla 1 state
    chosen_action_1_vdhla = 0
    if i != 0:
        chosen_action_1_vdhla = s_vdhla1.choose_action()
    else:
        chosen_action_1_vdhla = s_vdhla1.choose_random_action()

    evaluated_action_vdhla1 = environment_1.evaluate_action(
        chosen_action_1_vdhla)
    s_vdhla1.receive_environment_signal(evaluated_action_vdhla1)
    s_vdhla1.visualization_calculations()

    favorable1_vdhla_action_probability.append(
        s_vdhla1.get_action_selection_status(get_status)[-1] / (i + 1))

    # vdhla 3 state
    chosen_action_3_vdhla = 0
    if i != 0:
        chosen_action_3_vdhla = s_vdhla3.choose_action()
    else:
        chosen_action_3_vdhla = s_vdhla3.choose_random_action()

    evaluated_action_vdhla3 = environment_2.evaluate_action(
        chosen_action_3_vdhla)
    s_vdhla3.receive_environment_signal(evaluated_action_vdhla3)
    s_vdhla3.visualization_calculations()

    favorable3_vdhla_action_probability.append(
        s_vdhla3.get_action_selection_status(get_status)[-1] / (i + 1))

    # vdhla 5 state
    chosen_action_5_vdhla = 0
    if i != 0:
        chosen_action_5_vdhla = s_vdhla5.choose_action()
    else:
        chosen_action_5_vdhla = s_vdhla5.choose_random_action()

    evaluated_action_vdhla5 = environment_3.evaluate_action(
        chosen_action_5_vdhla)
    s_vdhla5.receive_environment_signal(evaluated_action_vdhla5)
    s_vdhla5.visualization_calculations()

    favorable5_vdhla_action_probability.append(
        s_vdhla5.get_action_selection_status(get_status)[-1] / (i + 1))

    # vdhla 7 state
    chosen_action_7_vdhla = 0
    if i != 0:
        chosen_action_7_vdhla = s_vdhla7.choose_action()
    else:
        chosen_action_7_vdhla = s_vdhla7.choose_random_action()

    evaluated_action_vdhla7 = environment_4.evaluate_action(
        chosen_action_7_vdhla)
    s_vdhla7.receive_environment_signal(evaluated_action_vdhla7)
    s_vdhla7.visualization_calculations()

    favorable7_vdhla_action_probability.append(
        s_vdhla7.get_action_selection_status(get_status)[-1] / (i + 1))

    # 1 state
    chosen_action_1 = 0
    if i != 0:
        chosen_action_1 = tsetlin_1state.choose_action()
    else:
        chosen_action_1 = tsetlin_1state.choose_random_action()

    evaluated_action_1 = environment_5.evaluate_action(chosen_action_1)
    tsetlin_1state.receive_environment_signal(evaluated_action_1)
    tsetlin_1state.visualization_calculations()

    favorable1_action_probability.append(
        tsetlin_1state.get_action_selection_status(get_status)[-1] / (i + 1))

    # 3 state
    chosen_action_3 = 0
    if i != 0:
        chosen_action_3 = tsetlin_3state.choose_action()
    else:
        chosen_action_3 = tsetlin_3state.choose_random_action()

    evaluated_action_3 = environment_6.evaluate_action(chosen_action_3)
    tsetlin_3state.receive_environment_signal(evaluated_action_3)
    tsetlin_3state.visualization_calculations()

    favorable3_action_probability.append(
        tsetlin_3state.get_action_selection_status(get_status)[-1] / (i + 1))

    # 5 state
    chosen_action_5 = 0
    if i != 0:
        chosen_action_5 = tsetlin_5state.choose_action()
    else:
        chosen_action_5 = tsetlin_5state.choose_random_action()

    evaluated_action_5 = environment_7.evaluate_action(chosen_action_5)
    tsetlin_5state.receive_environment_signal(evaluated_action_5)
    tsetlin_5state.visualization_calculations()

    favorable5_action_probability.append(
        tsetlin_5state.get_action_selection_status(get_status)[-1] / (i + 1))

    # 7 state
    chosen_action_7 = 0
    if i != 0:
        chosen_action_7 = tsetlin_7state.choose_action()
    else:
        chosen_action_7 = tsetlin_7state.choose_random_action()

    evaluated_action_7 = environment_8.evaluate_action(chosen_action_7)
    tsetlin_7state.receive_environment_signal(evaluated_action_7)
    tsetlin_7state.visualization_calculations()

    favorable7_action_probability.append(
        tsetlin_7state.get_action_selection_status(get_status)[-1] / (i + 1))


print('VDHLA 1 : TNR {}'.format(s_vdhla1.total_number_of_rewards[-1]))
print('VDHLA 1 : TNAS {}'.format(
    s_vdhla1.total_number_of_action_switching[-1]))
print('VDHLA 3 : TNR {}'.format(s_vdhla3.total_number_of_rewards[-1]))
print('VDHLA 3 : TNAS {}'.format(
    s_vdhla3.total_number_of_action_switching[-1]))
print('VDHLA 5 : TNR {}'.format(s_vdhla5.total_number_of_rewards[-1]))
print('VDHLA 5 : TNAS {}'.format(
    s_vdhla5.total_number_of_action_switching[-1]))
print('VDHLA 7 : TNR {}'.format(s_vdhla7.total_number_of_rewards[-1]))
print('VDHLA 7 : TNAS {}'.format(
    s_vdhla7.total_number_of_action_switching[-1]))

print('FSLA 1 : TNR {}'.format(tsetlin_1state.total_number_of_rewards[-1]))
print('FSLA 1 : TNAS {}'.format(
    tsetlin_1state.total_number_of_action_switching[-1]))
print('FSLA 3 : TNR {}'.format(tsetlin_3state.total_number_of_rewards[-1]))
print('FSLA 3 : TNAS {}'.format(
    tsetlin_3state.total_number_of_action_switching[-1]))
print('FSLA 5 : TNR {}'.format(tsetlin_5state.total_number_of_rewards[-1]))
print('FSLA 5 : TNAS {}'.format(
    tsetlin_5state.total_number_of_action_switching[-1]))
print('FSLA 7 : TNR {}'.format(tsetlin_7state.total_number_of_rewards[-1]))
print('FSLA 7 : TNAS {}'.format(
    tsetlin_7state.total_number_of_action_switching[-1]))

print(s_vdhla1.fsla_state_number)
print(s_vdhla3.fsla_state_number)
print(s_vdhla5.fsla_state_number)
print(s_vdhla7.fsla_state_number)

print(environment_1.action_probability_list)
print(environment_2.action_probability_list)
print(environment_3.action_probability_list)
print(environment_4.action_probability_list)
print(environment_5.action_probability_list)
print(environment_6.action_probability_list)
print(environment_7.action_probability_list)
print(environment_8.action_probability_list)

# Plots
# x_values = [i for i in range(iteration_number)]

# plt.plot(x_values, favorable1_vdhla_action_probability,
#          color='g', label='VDHLA(N=1)')
# plt.plot(x_values, favorable3_vdhla_action_probability,
#          color='b', label='VDHLA(N=3)')
# plt.plot(x_values, favorable5_vdhla_action_probability,
#          color='r', label='VDHLA(N=5)')
# plt.plot(x_values, favorable7_vdhla_action_probability,
#          color='y', label='VDHLA(N=7)')

# plt.plot(x_values, favorable1_action_probability,
#          color='g', label='Tsetlin(N=1)', linestyle='dashed')
# plt.plot(x_values, favorable3_action_probability,
#          color='b', label='Tsetlin(N=3)', linestyle='dashed')
# plt.plot(x_values, favorable5_action_probability,
#          color='r', label='Tsetlin(N=5)', linestyle='dashed')
# plt.plot(x_values, favorable7_action_probability,
#          color='y', label='Tsetlin(N=7)', linestyle='dashed')

# plt.title('FSLA Comparison-Ex2.1.1')
# plt.xlabel('iteration')
# plt.ylabel('favorable')

# plt.legend(loc="lower right")

# plt.show()