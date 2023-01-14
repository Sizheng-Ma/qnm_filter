__all__ = ['Data', 'Filter']

import qnm
import pandas as pd
import numpy as np
import lal
from collections import namedtuple
import scipy.signal as ss

T_MSUN = lal.MSUN_SI * lal.G_SI / lal.C_SI**3

ModeIndex = namedtuple('ModeIndex', ['l', 'm', 'n'])

def construct_mode_list(modes):
    if modes is None:
        modes = []
    elif isinstance(modes, str):
        from ast import literal_eval
        modes = literal_eval(modes)
    mode_list = []
    for (l, m, n) in modes:
        mode_list.append(ModeIndex(l, m, n))
    return mode_list