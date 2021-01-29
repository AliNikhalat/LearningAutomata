from matplotlib import pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../../..')))

from fala.s_model.vdhla.symmetric_variable_depth_hybrid import *  # NOQA
from fala.p_model.fsla.tsetlin import *  # NOQA
from environment.environment import *  # NOQA

iteration_number = 1000

action_number = 2
reward_rate = 0.1
penalty_rate = 0

s_vdhla2 = SymmetricVariableDepthHybrid(
    action_number, 2, reward_rate, penalty_rate)
s_vdhla4 = SymmetricVariableDepthHybrid(
    action_number, 4, reward_rate, penalty_rate)
s_vdhla6 = SymmetricVariableDepthHybrid(
    action_number, 6, reward_rate, penalty_rate)

tsetlin_2state = Tsetlin(2, action_number)
tsetlin_4state = Tsetlin(4, action_number)
tsetlin_6state = Tsetlin(6, action_number)

action_probability_list = [0.8, 0.2]
environment = Environment(action_number, action_probability_list)

favorable2_vdhla_action_probability = []
favorable4_vdhla_action_probability = []
favorable6_vdhla_action_probability = []

favorable2_action_probability = []
favorable4_action_probability = []
favorable6_action_probability = []

for i in range(iteration_number):
    # vdhla 2 state
    chosen_action_2_vdhla = 0
    if i != 0:
        chosen_action_2_vdhla = s_vdhla2.choose_action()
    else:
        chosen_action_2_vdhla = s_vdhla2.choose_random_action()

    evaluated_action_vdhla2 = environment.evaluate_action(
        chosen_action_2_vdhla)
    s_vdhla2.receive_environment_signal(evaluated_action_vdhla2)
    s_vdhla2.visualization_calculations()

    favorable2_vdhla_action_probability.append(
        s_vdhla2.get_action_selection_status(0)[-1] / (i + 1))

    # vdhla 4 state
    chosen_action_4_vdhla = 0
    if i != 0:
        chosen_action_4_vdhla = s_vdhla4.choose_action()
    else:
        chosen_action_4_vdhla = s_vdhla4.choose_random_action()

    evaluated_action_vdhla4 = environment.evaluate_action(
        chosen_action_4_vdhla)
    s_vdhla4.receive_environment_signal(evaluated_action_vdhla4)
    s_vdhla4.visualization_calculations()

    favorable4_vdhla_action_probability.append(
        s_vdhla4.get_action_selection_status(0)[-1] / (i + 1))

    # vdhla 6 state
    chosen_action_6_vdhla = 0
    if i != 0:
        chosen_action_6_vdhla = s_vdhla6.choose_action()
    else:
        chosen_action_6_vdhla = s_vdhla6.choose_random_action()

    evaluated_action_vdhla6 = environment.evaluate_action(
        chosen_action_6_vdhla)
    s_vdhla6.receive_environment_signal(evaluated_action_vdhla6)
    s_vdhla6.visualization_calculations()

    favorable6_vdhla_action_probability.append(
        s_vdhla6.get_action_selection_status(0)[-1] / (i + 1))

    # 2 state
    chosen_action_2 = 0
    if i != 0:
        chosen_action_2 = tsetlin_2state.choose_action()
    else:
        chosen_action_2 = tsetlin_2state.choose_random_action()

    evaluated_action_2 = environment.evaluate_action(chosen_action_2)
    tsetlin_2state.receive_environment_signal(evaluated_action_2)
    tsetlin_2state.visualization_calculations()

    favorable2_action_probability.append(
        tsetlin_2state.get_action_selection_status(0)[-1] / (i + 1))

    # 4 state
    chosen_action_4 = 0
    if i != 0:
        chosen_action_4 = tsetlin_4state.choose_action()
    else:
        chosen_action_4 = tsetlin_4state.choose_random_action()

    evaluated_action_4 = environment.evaluate_action(chosen_action_4)
    tsetlin_4state.receive_environment_signal(evaluated_action_4)
    tsetlin_4state.visualization_calculations()

    favorable4_action_probability.append(
        tsetlin_4state.get_action_selection_status(0)[-1] / (i + 1))

    # 6 state
    chosen_action_6 = 0
    if i != 0:
        chosen_action_6 = tsetlin_6state.choose_action()
    else:
        chosen_action_6 = tsetlin_6state.choose_random_action()

    evaluated_action_6 = environment.evaluate_action(chosen_action_6)
    tsetlin_6state.receive_environment_signal(evaluated_action_6)
    tsetlin_6state.visualization_calculations()

    favorable6_action_probability.append(
        tsetlin_6state.get_action_selection_status(0)[-1] / (i + 1))


# Plots
x_values = [i for i in range(iteration_number)]

plt.plot(x_values, favorable2_vdhla_action_probability,
         color='g', label='VDHLA(N=2)')
plt.plot(x_values, favorable4_vdhla_action_probability,
         color='b', label='VDHLA(N=4)')
plt.plot(x_values, favorable6_vdhla_action_probability,
         color='r', label='VDHLA(N=6)')

plt.plot(x_values, favorable2_action_probability,
         color='g', label='Tsetlin(N=2)', linestyle='dashed')
plt.plot(x_values, favorable4_action_probability,
         color='b', label='Tsetlin(N=4)', linestyle='dashed')
plt.plot(x_values, favorable6_action_probability,
         color='r', label='Tsetlin(N=6)', linestyle='dashed')

plt.title('FSLA Comparison-Ex3.2')
plt.xlabel('iteration')
plt.ylabel('favorable')

plt.legend(loc="lower right")

plt.show()


print('VDHLA 2 : TNR {}'.format(s_vdhla2.total_number_of_rewards[-1]))
print('VDHLA 2 : TNAS {}'.format(
    s_vdhla2.total_number_of_action_switching[-1]))
print('VDHLA 4 : TNR {}'.format(s_vdhla4.total_number_of_rewards[-1]))
print('VDHLA 4 : TNAS {}'.format(
    s_vdhla4.total_number_of_action_switching[-1]))
print('VDHLA 6 : TNR {}'.format(s_vdhla6.total_number_of_rewards[-1]))
print('VDHLA 6 : TNAS {}'.format(
    s_vdhla6.total_number_of_action_switching[-1]))

print('FSLA 2 : TNR {}'.format(tsetlin_2state.total_number_of_rewards[-1]))
print('FSLA 2 : TNAS {}'.format(
    tsetlin_2state.total_number_of_action_switching[-1]))
print('FSLA 4 : TNR {}'.format(tsetlin_4state.total_number_of_rewards[-1]))
print('FSLA 4 : TNAS {}'.format(
    tsetlin_4state.total_number_of_action_switching[-1]))
print('FSLA 6 : TNR {}'.format(tsetlin_6state.total_number_of_rewards[-1]))
print('FSLA 6 : TNAS {}'.format(
    tsetlin_6state.total_number_of_action_switching[-1]))
