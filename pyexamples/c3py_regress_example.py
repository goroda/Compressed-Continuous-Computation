import c3py # import the python interface to the c3 library
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

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


print("X shape ", X.shape)
print("Y shape ", Y.shape)

ft = c3py.FunctionTrain(X.shape[1])

ft.build_model(X, Y, LB, UB, nparam=5, init_rank=1, basis="legendre",
                    alg="AIO", obj="LS", adaptrank=1,
                    kickrank=1, roundtol=1e-10, verbose=0)
ft.run_regression(seed=10)

print("Xtest.shape ", Xtest.shape)
print("Ytest.shape ", Ytest.shape)
ft_evals = ft.eval(Xtest)

err = np.linalg.norm(ft_evals - Ytest) / np.linalg.norm(Ytest)

print("Relmse = ", err)
# plt.figure()
# plt.plot(ft_evals, Ytest, 'o')
# plt.show()

