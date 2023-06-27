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

def mother_wavelet():
    filename1 = "psi_non_orth.dat"
    loaded = False
    try: 
        data = np.loadtxt(filename1)
        loaded = True
    except:
        pass
    if loaded:
        N = data.shape[0]
        x = np.linspace(-1, 1, N)
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

        Q = inner_mat(data, data, x)

        qtp = inner_mat(data, p, x)
        print("q^Tp: \n", qtp )
    
    filename2a = "psi_orth.dat"
    
    datar1 = np.loadtxt(filename2a)
    N = datar1.shape[0]
    x1 = np.linspace(-1, 1, N)
    R = inner_mat(datar1, datar1, x1)
    print("R = \n", R)

    plt.figure()
    for ii in range(datar1.shape[1]):
        plt.plot(x1, datar1[:, ii], label=f"{ii}")
        # plt.plot(x, datar_should[:, ii], label=f"{ii}-should")
    plt.title("Should be orthogonal")
    
    plt.legend()
    plt.show()
    
if __name__ == "__main__":

    mother_wavelet()
    # a = True
    # if a:
    #     fit = np.loadtxt("mrb_evals.dat")
    #     test_func = np.loadtxt("testfunc.dat")
    #     plt.figure()
    #     plt.plot(test_func[:, 0], test_func[:, 1], '--', label='Test Func')
    #     plt.plot(fit[:, 0], fit[:, 1], '-x', label='Fit')
    #     plt.legend()
    #     plt.show()
