from matplotlib import pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../../../../..')))

from fala.s_model.vdhla.asymmetric_variable_depth_hybrid import *  # NOQA
from fala.s_model.vdhla.symmetric_variable_depth_hybrid import *  # NOQA
from environment.static.static_environment import *  # NOQA

iteration_number = 10000

action_number = 9
reward_rate = 0.1
penalty_rate = 0

a_vdhla1 = AsymmetricVariableDepthHybrid(
    action_number, [1 for _ in range(action_number)], reward_rate, penalty_rate)
a_vdhla3 = AsymmetricVariableDepthHybrid(
    action_number, [3 for _ in range(action_number)], reward_rate, penalty_rate)
a_vdhla5 = AsymmetricVariableDepthHybrid(
    action_number, [5 for _ in range(action_number)], reward_rate, penalty_rate)
a_vdhla7 = AsymmetricVariableDepthHybrid(
    action_number, [7 for _ in range(action_number)], reward_rate, penalty_rate)

s_vdhla1 = SymmetricVariableDepthHybrid(
    action_number, 1, reward_rate, penalty_rate)
s_vdhla3 = SymmetricVariableDepthHybrid(
    action_number, 3, reward_rate, penalty_rate)
s_vdhla5 = SymmetricVariableDepthHybrid(
    action_number, 5, reward_rate, penalty_rate)
s_vdhla7 = SymmetricVariableDepthHybrid(
    action_number, 7, reward_rate, penalty_rate)

action_probability_list = [0.8, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
# action_probability_list = [0.8, 0.05, 0.05, 0.05, 0.05]
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

    # vdhla 3 state
    chosen_action_3_vdhla = 0
    if i != 0:
        chosen_action_3_vdhla = a_vdhla3.choose_action()
    else:
        chosen_action_3_vdhla = a_vdhla3.choose_random_action()

    evaluated_action_vdhla3 = environment.evaluate_action(
        chosen_action_3_vdhla)
    a_vdhla3.receive_environment_signal(evaluated_action_vdhla3)
    a_vdhla3.visualization_calculations()

    favorable3_vdhla_action_probability.append(
        a_vdhla3.get_action_selection_status(0)[-1] / (i + 1))

    # vdhla 5 state
    chosen_action_5_vdhla = 0
    if i != 0:
        chosen_action_5_vdhla = a_vdhla5.choose_action()
    else:
        chosen_action_5_vdhla = a_vdhla5.choose_random_action()

    evaluated_action_vdhla5 = environment.evaluate_action(
        chosen_action_5_vdhla)
    a_vdhla5.receive_environment_signal(evaluated_action_vdhla5)
    a_vdhla5.visualization_calculations()

    favorable5_vdhla_action_probability.append(
        a_vdhla5.get_action_selection_status(0)[-1] / (i + 1))

    # vdhla 5 state
    chosen_action_7_vdhla = 0
    if i != 0:
        chosen_action_7_vdhla = a_vdhla7.choose_action()
    else:
        chosen_action_7_vdhla = a_vdhla7.choose_random_action()

    evaluated_action_vdhla7 = environment.evaluate_action(
        chosen_action_7_vdhla)
    a_vdhla7.receive_environment_signal(evaluated_action_vdhla7)
    a_vdhla7.visualization_calculations()

    favorable7_vdhla_action_probability.append(
        a_vdhla7.get_action_selection_status(0)[-1] / (i + 1))

    # 1 state
    chosen_action_1 = 0
    if i != 0:
        chosen_action_1 = s_vdhla1.choose_action()
    else:
        chosen_action_1 = s_vdhla1.choose_random_action()

    evaluated_action_1 = environment.evaluate_action(chosen_action_1)
    s_vdhla1.receive_environment_signal(evaluated_action_1)
    s_vdhla1.visualization_calculations()

    favorable1_action_probability.append(
        s_vdhla1.get_action_selection_status(0)[-1] / (i + 1))

    # 3 state
    chosen_action_3 = 0
    if i != 0:
        chosen_action_3 = s_vdhla3.choose_action()
    else:
        chosen_action_3 = s_vdhla3.choose_random_action()

    evaluated_action_3 = environment.evaluate_action(chosen_action_3)
    s_vdhla3.receive_environment_signal(evaluated_action_3)
    s_vdhla3.visualization_calculations()

    favorable3_action_probability.append(
        s_vdhla3.get_action_selection_status(0)[-1] / (i + 1))

    # 5 state
    chosen_action_5 = 0
    if i != 0:
        chosen_action_5 = s_vdhla5.choose_action()
    else:
        chosen_action_5 = s_vdhla5.choose_random_action()

    evaluated_action_5 = environment.evaluate_action(chosen_action_5)
    s_vdhla5.receive_environment_signal(evaluated_action_5)
    s_vdhla5.visualization_calculations()

    favorable5_action_probability.append(
        s_vdhla5.get_action_selection_status(0)[-1] / (i + 1))

    # 7 state
    chosen_action_7 = 0
    if i != 0:
        chosen_action_7 = s_vdhla7.choose_action()
    else:
        chosen_action_7 = s_vdhla7.choose_random_action()

    evaluated_action_7 = environment.evaluate_action(chosen_action_7)
    s_vdhla7.receive_environment_signal(evaluated_action_7)
    s_vdhla7.visualization_calculations()

    favorable7_action_probability.append(
        s_vdhla7.get_action_selection_status(0)[-1] / (i + 1))


print('AVDHLA 1 : TNR {}'.format(a_vdhla1.total_number_of_rewards[-1]))
print('AVDHLA 1 : TNAS {}'.format(
    a_vdhla1.total_number_of_action_switching[-1]))
print('AVDHLA 3 : TNR {}'.format(a_vdhla3.total_number_of_rewards[-1]))
print('AVDHLA 3 : TNAS {}'.format(
    a_vdhla3.total_number_of_action_switching[-1]))
print('AVDHLA 5 : TNR {}'.format(a_vdhla5.total_number_of_rewards[-1]))
print('AVDHLA 5 : TNAS {}'.format(
    a_vdhla5.total_number_of_action_switching[-1]))
print('AVDHLA 7 : TNR {}'.format(a_vdhla7.total_number_of_rewards[-1]))
print('AVDHLA 7 : TNAS {}'.format(
    a_vdhla7.total_number_of_action_switching[-1]))

print('SVDHLA 1 : TNR {}'.format(s_vdhla1.total_number_of_rewards[-1]))
print('SVDHLA 1 : TNAS {}'.format(
    s_vdhla1.total_number_of_action_switching[-1]))
print('SVDHLA 3 : TNR {}'.format(s_vdhla3.total_number_of_rewards[-1]))
print('SVDHLA 3 : TNAS {}'.format(
    s_vdhla3.total_number_of_action_switching[-1]))
print('SVDHLA 5 : TNR {}'.format(s_vdhla5.total_number_of_rewards[-1]))
print('SVDHLA 5 : TNAS {}'.format(
    s_vdhla5.total_number_of_action_switching[-1]))
print('SVDHLA 7 : TNR {}'.format(s_vdhla7.total_number_of_rewards[-1]))
print('SVDHLA 7 : TNAS {}'.format(
    s_vdhla7.total_number_of_action_switching[-1]))

# Plots
x_values = [i for i in range(iteration_number)]

plt.plot(x_values, favorable1_vdhla_action_probability,
         color='g', label='Asymmetric VDHLA(N=1)')
plt.plot(x_values, favorable3_vdhla_action_probability,
         color='b', label='Asymmetric VDHLA(N=3)')
plt.plot(x_values, favorable5_vdhla_action_probability,
         color='r', label='Asymmetric VDHLA(N=5)')
plt.plot(x_values, favorable7_vdhla_action_probability,
         color='y', label='Asymmetric VDHLA(N=7)')

plt.plot(x_values, favorable1_action_probability,
         color='g', label='Symmetric VDHLA(N=1)', linestyle='dashed')
plt.plot(x_values, favorable3_action_probability,
         color='b', label='Symmetric VDHLA(N=3)', linestyle='dashed')
plt.plot(x_values, favorable5_action_probability,
         color='r', label='Symmetric VDHLA(N=5)', linestyle='dashed')
plt.plot(x_values, favorable7_action_probability,
         color='y', label='Symmetric VDHLA(N=7)', linestyle='dashed')

plt.title('SVDHLA Comparison-Ex1.2.4.6')
plt.xlabel('Iteration')
plt.ylabel('Probability of choosing favorable action(a1)')

plt.legend(loc="lower right")

plt.show()
