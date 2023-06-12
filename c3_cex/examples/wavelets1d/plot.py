import numpy as np
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    filename = "psi_non_orth.dat"
    data = np.loadtxt(filename)
    N = data.shape[0]
    x = np.linspace(0, 1, N)

    plt.figure()
    plt.plot(x, data)
    plt.show()
