import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from markovian_switching.markovian_environment import MarkovianEnvironment  # NOQA
from markovian_switching.sequence_markovian_environment import SequenceMarkovianEnvironment  # NOQA
