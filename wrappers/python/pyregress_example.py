import c3py # import the python interface to the c3 library
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return x[:, 0]**2 + x[:, 1]**2 + x[:, 0] * x[:, 1] + np.sin(np.sum(x, axis=1)) \
        + x[:,0] * x[:,-1] + x[:, -1]**2.0 * np.cos(x[:, 0])


LB = -1.0                              # lower bounds of features
UB = 1.0                               # upper bounds of features
DIM = 20                               # number of features

Ntrain = 100
X = np.random.rand(Ntrain, 2)
Y = func(X)

Ntest = 1000
Xtest = np.random.rand(Ntest, 2)
Ytest = func(Xtest)


def build_ft_regress(xdata, ydata, nparam=2, init_rank=5, adaptrank=1, verbose=0):
    dim = xdata.shape[1]
    ndata = xdata.shape[0]
    
    ft = c3py.FunctionTrain(dim)
    for ii in range(dim):
        ft.set_dim_opts(ii, "legendre", LB, UB, nparam)

    ranks = [init_rank]*(dim+1)
    ranks[0] = 1
    ranks[dim] = 1
    ft.set_ranks(ranks)
    ft.build_data_model(ndata, xdata, ydata,
                        alg="AIO", obj="LS", adaptrank=adaptrank,
                        kickrank=1, roundtol=1e-10, verbose=verbose)
    return ft

print("X shape ", X.shape)
print("Y shape ", Y.shape)
ft_regress = build_ft_regress(X, Y, nparam=5, init_rank=1, adaptrank=1, verbose=0)

print("Xtest.shape ", Xtest.shape)
print("Ytest.shape ", Ytest.shape)
ft_evals = ft_regress.eval(Xtest)

err = np.linalg.norm(ft_evals - Ytest) / np.linalg.norm(Ytest)

print("Relmse = ", err)
plt.figure()
plt.plot(ft_evals, Ytest, 'o')
plt.show()

