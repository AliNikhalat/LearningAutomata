from matplotlib import pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../../../../..')))

from fala.s_model.vdhla.asymmetric_variable_depth_hybrid import *  # NOQA
from fala.p_model.fsla.tsetlin import *  # NOQA
from environment.static.static_environment import *  # NOQA

iteration_number = 10000

action_number = 5
reward_rate = 0.1
penalty_rate = 0.01

a_vdhla1 = AsymmetricVariableDepthHybrid(
    action_number, [3, 10, 10, 10, 10], reward_rate, penalty_rate)
# a_vdhla3 = AsymmetricVariableDepthHybrid(
#     action_number, [3 for _ in range(action_number)], reward_rate, penalty_rate)
# a_vdhla5 = AsymmetricVariableDepthHybrid(
#     action_number, [5 for _ in range(action_number)], reward_rate, penalty_rate)
# a_vdhla7 = AsymmetricVariableDepthHybrid(
#     action_number, [7 for _ in range(action_number)], reward_rate, penalty_rate)

tsetlin_1state = Tsetlin(1, action_number)
tsetlin_3state = Tsetlin(3, action_number)
tsetlin_5state = Tsetlin(5, action_number)
tsetlin_7state = Tsetlin(7, action_number)

# action_probability_list = [0.8, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
action_probability_list = [0.8, 0.05, 0.05, 0.05, 0.05]
# action_probability_list = [0.8, 0.2]
environment = StaticEnvironment(action_number, action_probability_list)

favorable1_vdhla_action_probability = []
favorable3_vdhla_action_probability = []
favorable5_vdhla_action_probability = []
favorable7_vdhla_action_probability = []

favorable1_action_probability = []
favorable3_action_probability = []
favorable5_action_probability = []
favorable7_action_probability = []

for i in range(iteration_number):
    # vdhla 1 state
    chosen_action_1_vdhla = 0
    if i != 0:
        chosen_action_1_vdhla = a_vdhla1.choose_action()
    else:
        chosen_action_1_vdhla = a_vdhla1.choose_random_action()

    evaluated_action_vdhla1 = environment.evaluate_action(
        chosen_action_1_vdhla)
    a_vdhla1.receive_environment_signal(evaluated_action_vdhla1)
    a_vdhla1.visualization_calculations()

    favorable1_vdhla_action_probability.append(
        a_vdhla1.get_action_selection_status(0)[-1] / (i + 1))

    # # vdhla 3 state
    # chosen_action_3_vdhla = 0
    # if i != 0:
    #     chosen_action_3_vdhla = a_vdhla3.choose_action()
    # else:
    #     chosen_action_3_vdhla = a_vdhla3.choose_random_action()

    # evaluated_action_vdhla3 = environment.evaluate_action(
    #     chosen_action_3_vdhla)
    # a_vdhla3.receive_environment_signal(evaluated_action_vdhla3)
    # a_vdhla3.visualization_calculations()

    # favorable3_vdhla_action_probability.append(
    #     a_vdhla3.get_action_selection_status(0)[-1] / (i + 1))

    # # vdhla 5 state
    # chosen_action_5_vdhla = 0
    # if i != 0:
    #     chosen_action_5_vdhla = a_vdhla5.choose_action()
    # else:
    #     chosen_action_5_vdhla = a_vdhla5.choose_random_action()

    # evaluated_action_vdhla5 = environment.evaluate_action(
    #     chosen_action_5_vdhla)
    # a_vdhla5.receive_environment_signal(evaluated_action_vdhla5)
    # a_vdhla5.visualization_calculations()

    # favorable5_vdhla_action_probability.append(
    #     a_vdhla5.get_action_selection_status(0)[-1] / (i + 1))

    # # vdhla 5 state
    # chosen_action_7_vdhla = 0
    # if i != 0:
    #     chosen_action_7_vdhla = a_vdhla7.choose_action()
    # else:
    #     chosen_action_7_vdhla = a_vdhla7.choose_random_action()

    # evaluated_action_vdhla7 = environment.evaluate_action(
    #     chosen_action_7_vdhla)
    # a_vdhla7.receive_environment_signal(evaluated_action_vdhla7)
    # a_vdhla7.visualization_calculations()

    # favorable7_vdhla_action_probability.append(
    #     a_vdhla7.get_action_selection_status(0)[-1] / (i + 1))

    # 1 state
    chosen_action_1 = 0
    if i != 0:
        chosen_action_1 = tsetlin_1state.choose_action()
    else:
        chosen_action_1 = tsetlin_1state.choose_random_action()

    evaluated_action_1 = environment.evaluate_action(chosen_action_1)
    tsetlin_1state.receive_environment_signal(evaluated_action_1)
    tsetlin_1state.visualization_calculations()

    favorable1_action_probability.append(
        tsetlin_1state.get_action_selection_status(0)[-1] / (i + 1))

    # 3 state
    chosen_action_3 = 0
    if i != 0:
        chosen_action_3 = tsetlin_3state.choose_action()
    else:
        chosen_action_3 = tsetlin_3state.choose_random_action()

    evaluated_action_3 = environment.evaluate_action(chosen_action_3)
    tsetlin_3state.receive_environment_signal(evaluated_action_3)
    tsetlin_3state.visualization_calculations()

    favorable3_action_probability.append(
        tsetlin_3state.get_action_selection_status(0)[-1] / (i + 1))

    # 5 state
    chosen_action_5 = 0
    if i != 0:
        chosen_action_5 = tsetlin_5state.choose_action()
    else:
        chosen_action_5 = tsetlin_5state.choose_random_action()

    evaluated_action_5 = environment.evaluate_action(chosen_action_5)
    tsetlin_5state.receive_environment_signal(evaluated_action_5)
    tsetlin_5state.visualization_calculations()

    favorable5_action_probability.append(
        tsetlin_5state.get_action_selection_status(0)[-1] / (i + 1))

    # 7 state
    chosen_action_7 = 0
    if i != 0:
        chosen_action_7 = tsetlin_7state.choose_action()
    else:
        chosen_action_7 = tsetlin_7state.choose_random_action()

    evaluated_action_7 = environment.evaluate_action(chosen_action_7)
    tsetlin_7state.receive_environment_signal(evaluated_action_7)
    tsetlin_7state.visualization_calculations()

    favorable7_action_probability.append(
        tsetlin_7state.get_action_selection_status(0)[-1] / (i + 1))


print('VDHLA 1 : TNR {}'.format(a_vdhla1.total_number_of_rewards[-1]))
print('VDHLA 1 : TNAS {}'.format(
    a_vdhla1.total_number_of_action_switching[-1]))
# print('VDHLA 3 : TNR {}'.format(a_vdhla3.total_number_of_rewards[-1]))
# print('VDHLA 3 : TNAS {}'.format(
#     a_vdhla3.total_number_of_action_switching[-1]))
# print('VDHLA 5 : TNR {}'.format(a_vdhla5.total_number_of_rewards[-1]))
# print('VDHLA 5 : TNAS {}'.format(
#     a_vdhla5.total_number_of_action_switching[-1]))
# print('VDHLA 7 : TNR {}'.format(a_vdhla7.total_number_of_rewards[-1]))
# print('VDHLA 7 : TNAS {}'.format(
#     a_vdhla7.total_number_of_action_switching[-1]))

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

# Plots
x_values = [i for i in range(iteration_number)]

plt.plot(x_values, favorable1_vdhla_action_probability,
         color='g', label='Asymmetric VDHLA(N=[3, 10, 10, 10, 10])')
# plt.plot(x_values, favorable3_vdhla_action_probability,
#          color='b', label='Asymmetric VDHLA(N=3)')
# plt.plot(x_values, favorable5_vdhla_action_probability,
#          color='r', label='Asymmetric VDHLA(N=5)')
# plt.plot(x_values, favorable7_vdhla_action_probability,
#          color='y', label='Asymmetric VDHLA(N=7)')

plt.plot(x_values, favorable1_action_probability,
         color='gray', label='Tsetlin(N=1)', linestyle='dashed')
plt.plot(x_values, favorable3_action_probability,
         color='b', label='Tsetlin(N=3)', linestyle='dashed')
plt.plot(x_values, favorable5_action_probability,
         color='r', label='Tsetlin(N=5)', linestyle='dashed')
plt.plot(x_values, favorable7_action_probability,
         color='y', label='Tsetlin(N=7)', linestyle='dashed')

plt.title('FSLA Comparison-Ex1.2.2.3')
plt.xlabel('Iteration')
plt.ylabel('Probability of choosing favorable action(a1)')

plt.legend(loc="lower right")

plt.show()
