import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../..')))

from fala.p_model.fsla.tsetlin import *  # NOQA

tsetlin = Tsetlin(1, 2)
