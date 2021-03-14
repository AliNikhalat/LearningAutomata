import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from environment.static import *  # NOQA
from environment.markovian_switching.markovian_environment import *  # NOQA
