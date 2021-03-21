from matplotlib import pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../../../../..')))

from fala.s_model.vdhla.symmetric_variable_depth_hybrid import *  # NOQA
from fala.p_model.vsla.variable_structure import *  # NOQA
from environment.static.static_environment import *  # NOQA

iteration_number = 1000

action_number = 2
state_number_1 = 1
state_number_4 = 4
state_number_6 = 6
reward_rate = 0.1
penalty_rate = 0.01

# # *****************************************Ex1.1*****************************************
# s_vdhla_1 = SymmetricVariableDepthHybrid(
#     action_number, state_number_1, reward_rate, penalty_rate)
# s_vdhla_4 = SymmetricVariableDepthHybrid(
#     action_number, state_number_4, reward_rate, penalty_rate)
# s_vdhla_6 = SymmetricVariableDepthHybrid(
#     action_number, state_number_6, reward_rate, penalty_rate)

# pure_chance_automata = VariableStructure(action_number, 0, 0)


# action_probability_list = [0.1, 0.9]
# environment = StaticEnvironment(action_number, action_probability_list)

# for i in range(iteration_number):
#     # Symmetric VDHLA Tests --> N=1
#     chosen_action_vdhla_1 = 0
#     if i != 0:
#         chosen_action_vdhla_1 = s_vdhla_1.choose_action()
#     else:
#         chosen_action_vdhla_1 = s_vdhla_1.choose_random_action()

#     evaluated_action_vdhla_1 = environment.evaluate_action(
#         chosen_action_vdhla_1)
#     s_vdhla_1.receive_environment_signal(evaluated_action_vdhla_1)

#     # Symmetric VDHLA Tests --> N=4
#     chosen_action_vdhla_4 = 0
#     if i != 0:
#         chosen_action_vdhla_4 = s_vdhla_4.choose_action()
#     else:
#         chosen_action_vdhla_4 = s_vdhla_4.choose_random_action()

#     evaluated_action_vdhla_4 = environment.evaluate_action(
#         chosen_action_vdhla_4)
#     s_vdhla_4.receive_environment_signal(evaluated_action_vdhla_4)

#     # Symmetric VDHLA Tests --> N=6
#     chosen_action_vdhla_6 = 0
#     if i != 0:
#         chosen_action_vdhla_6 = s_vdhla_6.choose_action()
#     else:
#         chosen_action_vdhla_6 = s_vdhla_6.choose_random_action()

#     evaluated_action_vdhla_6 = environment.evaluate_action(
#         chosen_action_vdhla_6)
#     s_vdhla_6.receive_environment_signal(evaluated_action_vdhla_6)

#     # Pure Chance Automata Tests
#     chosen_action_pure_chance = pure_chance_automata.choose_action()
#     evaluated_action_pure_chance = environment.evaluate_action(chosen_action_pure_chance)  # NOQA
#     pure_chance_automata.receive_environment_signal(evaluated_action_pure_chance)  # NOQA


# *****************************************Ex1.2*****************************************
# s_vdhla_1 = SymmetricVariableDepthHybrid(
#     action_number, state_number_1, reward_rate, penalty_rate)
# s_vdhla_4 = SymmetricVariableDepthHybrid(
#     action_number, state_number_4, reward_rate, penalty_rate)
# s_vdhla_6 = SymmetricVariableDepthHybrid(
#     action_number, state_number_6, reward_rate, penalty_rate)

# pure_chance_automata = VariableStructure(action_number, 0, 0)


# action_probability_list = [0.3, 0.7]
# environment = StaticEnvironment(action_number, action_probability_list)

# for i in range(iteration_number):
#     # Symmetric VDHLA Tests --> N=1
#     chosen_action_vdhla_1 = 0
#     if i != 0:
#         chosen_action_vdhla_1 = s_vdhla_1.choose_action()
#     else:
#         chosen_action_vdhla_1 = s_vdhla_1.choose_random_action()

#     evaluated_action_vdhla_1 = environment.evaluate_action(
#         chosen_action_vdhla_1)
#     s_vdhla_1.receive_environment_signal(evaluated_action_vdhla_1)

#     # Symmetric VDHLA Tests --> N=4
#     chosen_action_vdhla_4 = 0
#     if i != 0:
#         chosen_action_vdhla_4 = s_vdhla_4.choose_action()
#     else:
#         chosen_action_vdhla_4 = s_vdhla_4.choose_random_action()

#     evaluated_action_vdhla_4 = environment.evaluate_action(
#         chosen_action_vdhla_4)
#     s_vdhla_4.receive_environment_signal(evaluated_action_vdhla_4)

#     # Symmetric VDHLA Tests --> N=6
#     chosen_action_vdhla_6 = 0
#     if i != 0:
#         chosen_action_vdhla_6 = s_vdhla_6.choose_action()
#     else:
#         chosen_action_vdhla_6 = s_vdhla_6.choose_random_action()

#     evaluated_action_vdhla_6 = environment.evaluate_action(
#         chosen_action_vdhla_6)
#     s_vdhla_6.receive_environment_signal(evaluated_action_vdhla_6)

#     # Pure Chance Automata Tests
#     chosen_action_pure_chance = pure_chance_automata.choose_action()
#     evaluated_action_pure_chance = environment.evaluate_action(chosen_action_pure_chance)  # NOQA
#     pure_chance_automata.receive_environment_signal(evaluated_action_pure_chance)  # NOQA


# *****************************************Ex1.3*****************************************
s_vdhla_1 = SymmetricVariableDepthHybrid(
    action_number, state_number_1, reward_rate, penalty_rate)
s_vdhla_4 = SymmetricVariableDepthHybrid(
    action_number, state_number_4, reward_rate, penalty_rate)
s_vdhla_6 = SymmetricVariableDepthHybrid(
    action_number, state_number_6, reward_rate, penalty_rate)

pure_chance_automata = VariableStructure(action_number, 0, 0)


action_probability_list = [0.5, 0.5]
environment = StaticEnvironment(action_number, action_probability_list)

for i in range(iteration_number):
    # Symmetric VDHLA Tests --> N=1
    chosen_action_vdhla_1 = 0
    if i != 0:
        chosen_action_vdhla_1 = s_vdhla_1.choose_action()
    else:
        chosen_action_vdhla_1 = s_vdhla_1.choose_random_action()

    evaluated_action_vdhla_1 = environment.evaluate_action(
        chosen_action_vdhla_1)
    s_vdhla_1.receive_environment_signal(evaluated_action_vdhla_1)

    # Symmetric VDHLA Tests --> N=4
    chosen_action_vdhla_4 = 0
    if i != 0:
        chosen_action_vdhla_4 = s_vdhla_4.choose_action()
    else:
        chosen_action_vdhla_4 = s_vdhla_4.choose_random_action()

    evaluated_action_vdhla_4 = environment.evaluate_action(
        chosen_action_vdhla_4)
    s_vdhla_4.receive_environment_signal(evaluated_action_vdhla_4)

    # Symmetric VDHLA Tests --> N=6
    chosen_action_vdhla_6 = 0
    if i != 0:
        chosen_action_vdhla_6 = s_vdhla_6.choose_action()
    else:
        chosen_action_vdhla_6 = s_vdhla_6.choose_random_action()

    evaluated_action_vdhla_6 = environment.evaluate_action(
        chosen_action_vdhla_6)
    s_vdhla_6.receive_environment_signal(evaluated_action_vdhla_6)

    # Pure Chance Automata Tests
    chosen_action_pure_chance = pure_chance_automata.choose_action()
    evaluated_action_pure_chance = environment.evaluate_action(chosen_action_pure_chance)  # NOQA
    pure_chance_automata.receive_environment_signal(evaluated_action_pure_chance)  # NOQA


print(s_vdhla_1.fsla_state_number)
print(s_vdhla_4.fsla_state_number)
print(s_vdhla_6.fsla_state_number)
# print(learning_automata.total_number_of_rewards)

# Plots
x_values = [i for i in range(iteration_number)]

plt.plot(x_values, s_vdhla_1.get_total_number_of_rewards,
         color='r', label='Symmetric VDHLA(N=1)', linestyle='dashed')
plt.plot(x_values, s_vdhla_4.get_total_number_of_rewards,
         color='g', label='Symmetric VDHLA(N=4)', linestyle='dashed')
plt.plot(x_values, s_vdhla_6.get_total_number_of_rewards,
         color='purple', label='Symmetric VDHLA(N=6)', linestyle='dashed')
plt.plot(x_values, pure_chance_automata.get_total_number_of_rewards,
         color='b', label='Pure Chance')

plt.title('Pure Chance Comparison-Ex1.1.3')
plt.xlabel('Iteration')
plt.ylabel('TNR')

plt.legend(loc="upper left")

plt.show()
