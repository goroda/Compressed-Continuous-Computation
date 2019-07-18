import c3py # import the python interface to the c3 library
import numpy as np

DIM = 5  # number of features
def func2(x, param=None):
    # print("x = ",x)
    # print("param = ", param)
    # print("shape ", x.shape)
    if param is not None:
        # print("param = ", param)
        # print(x[:, 2])
        for ii in range(DIM):
            x[:, ii] = x[:, ii] * param[ii]
    #     print(x[:, 2])
    # print("\n\n\n\n\n")
    out = np.sin(np.sum(x, axis=1))
    return out

if __name__ == "__main__":
    
    lb = [-1]*DIM    # lower bounds of features
    ub = [1]*DIM     # upper bounds of features

    # number of parameters per univariate function
    nparam = [3]*DIM 

    ## Run a rank-adaptive regression routine to approximate the first function
    ft = c3py.FunctionTrain(DIM)
    for ii in range(DIM):
        ft.set_dim_opts(ii, "legendre", lb[ii], ub[ii], nparam[ii])

    verbose = 1
    init_rank = 2
    adapt = 0
    maxrank = 10
    kickrank = 2
    roundtol = 1e-10
    maxiter = 5
    scales = [2]*DIM; scales[2] = 0.2
    scales = None
    ft.build_approximation(func2,scales,init_rank,verbose,adapt,
                           maxrank=maxrank, round_tol=roundtol,
                           kickrank=kickrank, maxiter=maxiter)

    print("done")
    ## Generate test point
    test_pt = np.random.rand(DIM)*2.0-1.0
    ft_adapt_eval = ft.eval(test_pt)
    eval2s = func2(test_pt.reshape((1,DIM)))
    print("Second function with CrossApproximation:  Fteval =",ft_adapt_eval, "Should be =",eval2s)


    print("SAVE")
    filename = "saving_test.c3"
    ft.save(filename)

    ft_loaded = c3py.FunctionTrain(DIM)
    for ii in range(DIM):
        ft_loaded.set_dim_opts(ii, "legendre", lb[ii], ub[ii], nparam[ii])
    ft_loaded.load(filename)

    ftdiff = ft - ft_loaded
    norm2 = ftdiff.norm2()

    print("Norm = ", norm2)
    
    

    ft2 = ft + ft + ft + ft
    print("ft2 ranks = ", ft2.get_ranks())

    ft3 = ft2 * ft
    print("ft3 ranks = ", ft3.get_ranks())

    ft3.round(eps=1e-3)
    print("ft3 rounded ranks = ", ft3.get_ranks())

    inner_product = ft.inner(ft)
    print("Inner product = ", inner_product)

    integral = ft.integrate()
    print("Integral = ", integral)




    
