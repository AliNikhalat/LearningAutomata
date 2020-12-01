import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from fala.fsla.tsetlin import Tsetlin  # NOQA
from fala.vsla.variable_action_set import VariableActionSet  # NOQA
