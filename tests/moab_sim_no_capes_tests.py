import matplotlib.pyplot as plt
import numpy as np
from moab_sim_no_capes import MoabSim


def uniform_circle():
    d = np.array([MoabSim._uniform_circle(0.75) for _ in range(1000)])
    plt.scatter(d[:, 0], d[:, 1]), plt.show()


uniform_circle()
