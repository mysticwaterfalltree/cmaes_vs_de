import numpy as np
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE


def get_cmaes(n):
    return CMAES(x0=np.ones(n) * 3.0, sigma=0.5)


def get_de(n):
    return DE(pop_size=40)