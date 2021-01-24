import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from fala.p_model.fsla.tsetlin import Tsetlin  # NOQA
from fala.p_model.vsla.variable_action_set import VariableActionSet  # NOQA
