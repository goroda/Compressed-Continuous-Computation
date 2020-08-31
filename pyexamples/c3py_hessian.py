"Example evaluating a hessian of a function"
import c3py # import the python interface to the c3 library
import numpy as np
import matplotlib.pyplot as plt

## Define two functions

def func2(x,param=None):
    return np.sin(np.sum(x,axis=1))

def func3(x, param=None):
    return 0.2*x[:, 0] + 0.9*x[:,1] + 3.0*x[:, 0] * x[:, 1] + x[:,1]**2

def get_ft(func=func2, dim=2):
    lb = -1     # lower bounds of features
    ub = 1      # upper bounds of features
    nparam = 2  # number of parameters per univariate function

    ## Run adaptive sampling scheme
    ft_adapt = c3py.FunctionTrain(dim)
    for ii in range(dim):
        ft_adapt.set_dim_opts(ii,"legendre",lb,ub,nparam) 
    verbose=0
    init_rank=2
    adapt=1
    ft_adapt.build_approximation(func, None, init_rank, verbose, adapt)
    return ft_adapt

if __name__ == "__main__":

    dim=2
    ft = get_ft(func2, dim=dim)
    test_pt = np.random.rand(dim)*2.0-1.0
    hess_eval = ft.hess_eval(test_pt)
    print("hess_eval = ", hess_eval)
    print("each element should be ", -np.sin(test_pt[0] + test_pt[1]))


    dim=2
    ft = get_ft(func3, dim=dim)
    test_pt = np.random.rand(dim)*2.0-1.0
    print("test_pt = ", test_pt)
    hess_eval = ft.hess_eval(test_pt)
    print("evals should match = ", ft.eval(test_pt), func3(test_pt[np.newaxis,:]))
    print("hess_eval = \n", hess_eval)

    hess_should_be = np.array([[0.0, 3.0],
                               [3.0, 2.0]])
    print("hessian should match\n", hess_should_be)

    

