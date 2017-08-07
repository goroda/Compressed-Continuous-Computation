import c3py # import the python interface to the c3 library
import pycback as pcb
import numpy as np

dim = 5  # number of features
def func2(x,param=None):
    # print("x = ",x)
    # print("shape ", x.shape)
    if (param is not None):
        print("param = ", param)
    out = np.sin(np.sum(x,axis=1))
    # print("out = ",out)
    # print("out.shape ", out.shape)
    return out

if __name__ == "__main__":
    

    obj = pcb.alloc_cobj()
    pcb.assign(obj,dim,func2,"hi_from_func")
    pcb.eval_test_5d(obj)

    # fw = c3py.c3.fwrap_create(dim,"python")
    # c3py.fwrap_set_pyfunc(fw,obj);

    lb = -1                                # lower bounds of features
    ub = 1                                 # upper bounds of features
    nparam = 3                             # number of parameters per univariate function


    ## Run a rank-adaptive regression routine to approximate the first function
    ft = c3py.FunctionTrain(dim)
    for ii in range(dim):
        ft.set_dim_opts(ii,"legendre",lb,ub,nparam)

    verbose=1
    init_rank=2
    adapt=1
    ft.build_approximation(func2,None,init_rank,verbose,adapt)

    ## Generate test point
    test_pt = np.random.rand(dim)*2.0-1.0
    ft_adapt_eval = ft.eval(test_pt)
    eval2s = func2(test_pt.reshape((1,dim)))
    print("Second function with CrossApproximation:  Fteval =",ft_adapt_eval, "Should be =",eval2s)
