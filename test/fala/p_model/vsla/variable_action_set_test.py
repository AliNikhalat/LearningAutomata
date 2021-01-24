import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../..')))

from fala.p_model.vsla.variable_action_set import *  # NOQA

reward_rate = 0.8
penalty_rate = 0.1

variable_action_set = VariableActionSet(5, reward_rate, penalty_rate)
