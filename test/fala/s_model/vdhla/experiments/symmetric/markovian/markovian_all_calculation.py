from matplotlib import pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../../../../..')))

from fala.s_model.vdhla.symmetric_variable_depth_hybrid import *  # NOQA
from fala.p_model.fsla.tsetlin import *  # NOQA
from environment.markovian_switching import *  # NOQA

iteration_number = 100000


action_number = 5
reward_rate = 0.001
penalty_rate = 0.01

s_vdhla1 = SymmetricVariableDepthHybrid(
    action_number, 2, reward_rate, penalty_rate)
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

state_probability = [[0.9, 0.1, 0.1, 0.1, 0.1],
                     [0.1, 0.9, 0.1, 0.1, 0.1],
                     [0.1, 0.1, 0.9, 0.1, 0.1],
                     [0.1, 0.1, 0.1, 0.9, 0.1],
                     [0.5, 0.5, 0.5, 0.5, 0.5],
                     [0.1, 0.1, 0.1, 0.1, 0.1]]

transition_probability = [[0.1, 0.4, 0.1, 0.1, 0.15, 0.15],
                          [0.1, 0.1, 0.4, 0.1, 0.15, 0.15],
                          [0.1, 0.1, 0.1, 0.4, 0.15, 0.15],
                          [0.4, 0.1, 0.1, 0.1, 0.15, 0.15],
                          [0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
                          [0.2, 0.1, 0.1, 0.2, 0.1, 0.3]]


environment = SequenceMarkovianEnvironment(
    action_number, state_probability, transition_probability, iteration_number)
environment_sequence = environment.get_seqeunce()

dominant_actions = [state.index(max(state))
                    for state in state_probability]

dominant_chosen_vdhla_1 = []
dominant_chosen_vdhla_3 = []
dominant_chosen_vdhla_5 = []
dominant_chosen_vdhla_7 = []

dominant_chosen_fsla_1 = []
dominant_chosen_fsla_3 = []
dominant_chosen_fsla_5 = []
dominant_chosen_fsla_7 = []


def calculate_dominants_all(dominant_chosen, environment_state, action, evaluation):
    for state in range(len(state_probability)):
        if dominant_actions[state] == action and state == environment_state and evaluation == 0:
            dominant_chosen.append(
                1 + dominant_chosen[-1] if len(dominant_chosen) > 0 else 1)
            return

    dominant_chosen.append(
        0 + dominant_chosen[-1] if len(dominant_chosen) > 0 else 0)

    return


for i in range(iteration_number):
    # vdhla 1 state
    chosen_action_1_vdhla = 0
    if i != 0:
        chosen_action_1_vdhla = s_vdhla1.choose_action()
    else:
        chosen_action_1_vdhla = s_vdhla1.choose_random_action()

    evaluated_action_vdhla1 = environment.evaluate_action(
        chosen_action_1_vdhla)
    s_vdhla1.receive_environment_signal(evaluated_action_vdhla1)
    s_vdhla1.visualization_calculations()

    calculate_dominants_all(dominant_chosen_vdhla_1,
                            environment_sequence[i], chosen_action_1_vdhla, evaluated_action_vdhla1)

    # vdhla 3 state
    chosen_action_3_vdhla = 0
    if i != 0:
        chosen_action_3_vdhla = s_vdhla3.choose_action()
    else:
        chosen_action_3_vdhla = s_vdhla3.choose_random_action()

    evaluated_action_vdhla3 = environment.evaluate_action(
        chosen_action_3_vdhla)
    s_vdhla3.receive_environment_signal(evaluated_action_vdhla3)
    s_vdhla3.visualization_calculations()

    calculate_dominants_all(dominant_chosen_vdhla_3,
                            environment_sequence[i], chosen_action_3_vdhla, evaluated_action_vdhla3)

    # vdhla 5 state
    chosen_action_5_vdhla = 0
    if i != 0:
        chosen_action_5_vdhla = s_vdhla5.choose_action()
    else:
        chosen_action_5_vdhla = s_vdhla5.choose_random_action()

    evaluated_action_vdhla5 = environment.evaluate_action(
        chosen_action_5_vdhla)
    s_vdhla5.receive_environment_signal(evaluated_action_vdhla5)
    s_vdhla5.visualization_calculations()

    calculate_dominants_all(dominant_chosen_vdhla_5,
                            environment_sequence[i], chosen_action_5_vdhla, evaluated_action_vdhla5)

    # vdhla 7 state
    chosen_action_7_vdhla = 0
    if i != 0:
        chosen_action_7_vdhla = s_vdhla7.choose_action()
    else:
        chosen_action_7_vdhla = s_vdhla7.choose_random_action()

    evaluated_action_vdhla7 = environment.evaluate_action(
        chosen_action_7_vdhla)
    s_vdhla7.receive_environment_signal(evaluated_action_vdhla7)
    s_vdhla7.visualization_calculations()

    calculate_dominants_all(dominant_chosen_vdhla_7,
                            environment_sequence[i], chosen_action_7_vdhla, evaluated_action_vdhla7)

    # 1 state
    chosen_action_1 = 0
    if i != 0:
        chosen_action_1 = tsetlin_1state.choose_action()
    else:
        chosen_action_1 = tsetlin_1state.choose_random_action()

    evaluated_action_1 = environment.evaluate_action(chosen_action_1)
    tsetlin_1state.receive_environment_signal(evaluated_action_1)
    tsetlin_1state.visualization_calculations()

    calculate_dominants_all(dominant_chosen_fsla_1,
                            environment_sequence[i], chosen_action_1, evaluated_action_1)

    # 3 state
    chosen_action_3 = 0
    if i != 0:
        chosen_action_3 = tsetlin_3state.choose_action()
    else:
        chosen_action_3 = tsetlin_3state.choose_random_action()

    evaluated_action_3 = environment.evaluate_action(chosen_action_3)
    tsetlin_3state.receive_environment_signal(evaluated_action_3)
    tsetlin_3state.visualization_calculations()

    calculate_dominants_all(dominant_chosen_fsla_3,
                            environment_sequence[i], chosen_action_3, evaluated_action_3)

    # 5 state
    chosen_action_5 = 0
    if i != 0:
        chosen_action_5 = tsetlin_5state.choose_action()
    else:
        chosen_action_5 = tsetlin_5state.choose_random_action()

    evaluated_action_5 = environment.evaluate_action(chosen_action_5)
    tsetlin_5state.receive_environment_signal(evaluated_action_5)
    tsetlin_5state.visualization_calculations()

    calculate_dominants_all(dominant_chosen_fsla_5,
                            environment_sequence[i], chosen_action_5, evaluated_action_5)

    # 7 state
    chosen_action_7 = 0
    if i != 0:
        chosen_action_7 = tsetlin_7state.choose_action()
    else:
        chosen_action_7 = tsetlin_7state.choose_random_action()

    evaluated_action_7 = environment.evaluate_action(chosen_action_7)
    tsetlin_7state.receive_environment_signal(evaluated_action_7)
    tsetlin_7state.visualization_calculations()

    calculate_dominants_all(dominant_chosen_fsla_7,
                            environment_sequence[i], chosen_action_7, evaluated_action_7)

    environment.goto_next_episode()

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

#############################All Calculation ################################

# Plots
x_values = [i for i in range(iteration_number)]

plt.plot(x_values, dominant_chosen_vdhla_1,
         color='g', label='VDHLA(N=1)')
plt.plot(x_values, dominant_chosen_vdhla_3,
         color='b', label='VDHLA(N=3)')
plt.plot(x_values, dominant_chosen_vdhla_5,
         color='r', label='VDHLA(N=5)')
plt.plot(x_values, dominant_chosen_vdhla_7,
         color='y', label='VDHLA(N=7)')

plt.plot(x_values, dominant_chosen_fsla_1,
         color='g', label='Tsetlin(N=1)', linestyle='dashed')
plt.plot(x_values, dominant_chosen_fsla_3,
         color='b', label='Tsetlin(N=3)', linestyle='dashed')
plt.plot(x_values, dominant_chosen_fsla_5,
         color='r', label='Tsetlin(N=5)', linestyle='dashed')
plt.plot(x_values, dominant_chosen_fsla_7,
         color='y', label='Tsetlin(N=7)', linestyle='dashed')

plt.title('VDHLA-FSLA Dominant Action Comparison')
plt.xlabel('iteration')
plt.ylabel('dominant')

plt.legend(loc="lower right")

plt.show()

#################################################################################
