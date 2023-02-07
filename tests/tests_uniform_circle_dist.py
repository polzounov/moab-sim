import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.getcwd() + '/..')

from moab_sim import uniform_circle


def display_uniform_circle():
    d = np.array([uniform_circle(0.75) for _ in range(1000)])
    plt.scatter(d[:, 0], d[:, 1]), plt.show()


display_uniform_circle()
