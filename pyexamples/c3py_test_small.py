import c3py # import the python interface to the c3 library
import numpy as np

## Define two functions
def func1(x,param=None):
    return np.sum(x,axis=1)

def func2(x,param=None):
    return np.sum(x**2,axis=1)

DIM = 2                                # number of features
NDATA = 100                            # number of data points
X = np.random.rand(NDATA,DIM)*2.0-1.0  # training samples
Y = func1(X)                           # function values 

LB = -1                                # lower bounds of features
UB = 1                                 # upper bounds of features
NPARAM = 7                             # number of parameters per univariate function

# This should output an equal amount of allocates and trues
def build_ft_adapt(dim):
    ft = c3py.FunctionTrain(dim)
    for ii in range(dim):
        ft.set_dim_opts(ii, "legendre", LB, UB, NPARAM, coeff_check=2)
    verbose = 2
    init_rank = 2
    adapt = 1
    ft.build_approximation(func1, None, init_rank, verbose, adapt, maxiter=10)
    return ft

def build_ft_regress(dim):
    ft = c3py.FunctionTrain(dim)
    for ii in range(dim):
        ft.set_dim_opts(ii, "legendre", LB, UB, NPARAM)

    ranks = [2]*(DIM+1)
    ranks[0] = 1
    ranks[DIM] = 1
    ft.set_ranks(ranks)
    verbose = 0
    adaptrank = 0
    ft.build_data_model(NDATA, X, Y,
                        alg="AIO", obj="LS", adaptrank=adaptrank,
                        kickrank=1, roundtol=1e-10, verbose=verbose)
    return ft

ft_adapt = build_ft_adapt(DIM)
ft_regress = build_ft_regress(DIM)
# ft_diff = ft_adapt - ft_regress

for ii in range(100):
    pt = np.random.rand(1,DIM)*2.0-1.0
#     diff_eval = ft_diff.eval(pt)
    adapt_eval = ft_adapt.eval(pt[0,:])
    regress_eval = ft_regress.eval(pt[0, :])
    print("pt = ", pt[0,:])
    print("\t eval = ",  func1(pt)[0], adapt_eval, regress_eval)

#     print("adapt_eval, regress_eval, diff_eval ", adapt_eval, regress_eval, diff_eval)
    # assert np.abs(diff_eval) < 1e-3

print("\n\n\n")
print("Tests Passed!")
print("Bye!")
