
import c3py # import the python interface to the c3 library
import pycback as pcb
import numpy as np
import matplotlib.pyplot as plt

dim = 3  # number of features
def func2(x,param=None):
    if (param is not None):
        print("param = ", param)
    out = np.sin(2*np.pi*np.sum(x,axis=1))
    return out

if __name__ == "__main__":
    
    lb = -1      # lower bounds of features
    ub = 1       # upper bounds of features
    nparam = 3   # number of parameters per univariate function

    ## Run a rank-adaptive regression routine to approximate the first function
    ft = c3py.FunctionTrain(dim)
    for ii in range(dim):
        ft.set_dim_opts(ii,"legendre",lb,ub,nparam)

    verbose=0
    init_rank=2
    adapt=1
    ft.build_approximation(func2,None,init_rank,verbose,adapt)

    # collect functions
    ranks = ft.get_ranks()
    funcs = [[]]*dim
    for ii in range(dim):
        funcs[ii] = [None]*ranks[ii]
        f, axis = plt.subplots(ranks[ii],ranks[ii+1])
        for row in range(ranks[ii]):
            funcs[ii][row] = [None]*ranks[ii+1]
            for col in range(ranks[ii+1]):
                funcs[ii][row][col] = ft.get_uni_func(ii,row,col)
                x = np.linspace(-1,1,100)
                evals = funcs[ii][row][col].eval(x)
                title = "$f_{0}^{{({1},{2})}}$".format(ii,row,col)
                if axis.ndim == 1:
                    if ii == 0:
                        axis[col].plot(x,evals,label=title)
                        axis[col].legend()
                    else:
                        axis[row].plot(x,evals,label=title)
                        axis[row].legend()
                else:
                    axis[row,col].plot(x,evals,label=title)
                    axis[row,col].legend()

    plt.show()
                


    

        
    
