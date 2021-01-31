from matplotlib import pyplot as plt
import sys
import os

from asymmetric.asymmetric_arm_manager import *  # NOQA
from asymmetric.asymmetric_arm import *  # NOQA


class AsymmetricVariableDepthHybrid:
    def __init__(self, action_number, state_number, reward_rate, penalty_rate):
        self.__arm_number = action_number
        self.__init_state_number = state_number

        self.__reward_rate = reward_rate
        self.__penalty_rate = penalty_rate

        self.__chosen_arm = 0
        self.__arm_manager = AsymmetricArmManager(self.__arm_number)

        self.__arm_list = [AsymmetricArm(
            self.__init_state_number, self.__reward_rate, self.__penalty_rate, self.__arm_manager) for _ in range(self.__arm_number)]

        self.__asymmetric_arm_manager = None

    # *****************************************************************************************
    def choose_action(self):
        self.__chosen_arm = self.__arm_manager.chosen_arm

        return self.__chosen_arm

    # *****************************************************************************************
    def choose_random_action(self):
        self.__chosen_arm = self.__arm_manager.random_chosen_arm
        self.__arm_list[self.__chosen_arm].set_depth_status(1)

        return self.__chosen_arm

    # *****************************************************************************************
    def receive_environment_signal(self, beta):
        self.__arm_list[self.__chosen_arm].receive_environment_signal(beta)

        return
