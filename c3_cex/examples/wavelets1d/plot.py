import numpy as np
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    filename1 = "psi_non_orth.dat"
    data = np.loadtxt(filename1)
    N = data.shape[0]
    x = np.linspace(0, 1, N)

    plt.figure()
    plt.plot(x, data)
    plt.title("Non orthogonal")

    Q = np.dot(data.T, data) / N
    print("Q = \n", Q)

    
    filename2 = "psi_orth.dat"
    data = np.loadtxt(filename2)
    N = data.shape[0]
    x = np.linspace(0, 1, N)

    R = np.dot(data.T, data) / N
    print("R = \n", R)
    
    plt.figure()
    plt.plot(x, data)
    plt.title("Should be orthogonal")
    plt.show()
