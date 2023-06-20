import numpy as np
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4)
def inner_mat(data, data2, x):

    # trapezoid rule
    out = np.zeros((data.shape[1], data2.shape[1]))
    for ii in range(data.shape[1]):
        for jj in range(data2.shape[1]):
            dd = data[:, ii] * data2[:, jj]
            out[ii, jj] = np.sum(0.5 * (dd[1:] + dd[:-1]) * (x[1:] - x[:-1]))
    
    return out;

    
if __name__ == "__main__":

    filename1 = "psi_non_orth.dat"
    data = np.loadtxt(filename1)
    N = data.shape[0]
    x = np.linspace(-1, 1, N)
    # x = np.linspace(0, 1, N)
    p = np.zeros((N, data.shape[1]))
    for ii in range(data.shape[1]):
        p[:, ii] = x**ii

    qtilde = np.copy(p)
    for ii in range(data.shape[1]):
        qtilde[x<0.0, ii] = -qtilde[x<0.0, ii]
    
    A = inner_mat(p, p, x)
    B = inner_mat(qtilde, p, x)
    # B = np.dot(qtilde.T, p).T / N
    # X = np.linalg.solve(A, -B)
    print("A: \n", A)
    print("B: \n", B)
    # print("X: \n", X)
    # data_should = qtilde + np.dot(X.T, p.T).T

    # plt.figure()
    # plt.plot(x, data)
    # # plt.plot(x, data_should)
    # # plt.plot(x, p)
    # plt.title("Non orthogonal")
    
    # Q = np.dot(data.T, data) / N
    Q = inner_mat(data, data, x)
    # Qshould = np.dot(data_should.T, data_should) / N
    print("Q = \n", Q)
    # print("Q_should = \n", Qshould)

    qtp = inner_mat(data, p, x)
    print("q^Tp: \n", qtp )
    
    # print("q^Tr: \n", np.dot(data_should.T, p) / N )
    # plt.show()
    # exit(1)
    # QTR =
    
    # beta11 = -Q[-2, -1] / Q[-1, -1]
    # print("beta11 = ", beta11)
    
    filename2a = "psi_orth.dat"
    filename2b = "psi_orth2.dat"
    
    datar1 = np.loadtxt(filename2a)
    datar2 = np.loadtxt(filename2b)
    N = datar1.shape[0]
    x1 = np.linspace(-1, -1e-15, N)
    x2 = np.linspace(1e-15, 1, N)

    R1 = inner_mat(datar1, datar1, x1)
    R2 = inner_mat(datar2, datar2, x2)
    R = R1 + R2
    print("R = \n", R)

    # rtp = inner_mat(datar, p, x)
    # # print("q^Tp: \n", np.dot(data.T, p) / N )
    # print("r^tp = \n", rtp)
    # print("q^tp = \n", np.dot(data.T, p) / N)
    
    # datar_should = np.zeros((N, datar.shape[1]))
    # datar_should[:, -1] = data_should[:, -1]
    # datar_should[:, -2] = data_should[:, -2] + beta11 * datar_should[:, -1]

    
    
    # plt.figure()
    # for ii in range(data.shape[1]):
    #     plt.plot(x, datar[:, ii], label=f"{ii}")
    #     # plt.plot(x, datar_should[:, ii], label=f"{ii}-should")
    # plt.title("Should be orthogonal")
    
    # plt.legend()
    # plt.show()
