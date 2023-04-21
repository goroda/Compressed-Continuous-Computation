import c3py # import the python interface to the c3 library
import numpy as np
import matplotlib.pyplot as plt

dim = 4  # number of features
def func2(x,param=None):
    assert param is not None
    pnew = np.tile(param, (x.shape[0], 1))
    out = np.sin(np.sum(x*pnew,axis=1))    
    return out

def gen_results(alpha):
    lb = -1      # lower bounds of features
    ub = 1       # upper bounds of features
    nparam = 3    # number of parameters per univariate function


    ## Run a rank-adaptive regression routine to approximate the first function
    ft = c3py.FunctionTrain(dim)
    for ii in range(dim):
        ft.set_dim_opts(ii,"legendre",lb,ub,nparam)

    # print("ftdim = ", ft.dim)
    # exit(1)
    verbose=0
    init_rank=2
    adapt=1
    ft.build_approximation(func2,alpha,init_rank,verbose,adapt)

    print("Computing Sobol Indices")
    SI = c3py.SobolIndices(ft, order=2)
    print("done")
    var = SI.get_variance()
    names = []
    mains = np.zeros((dim,))
    totals = np.zeros((dim,))
    for ii in range(dim):
        mains[ii] = SI.get_main_sensitivity(ii)
        totals[ii] = SI.get_total_sensitivity(ii)
        names.append(str(ii))

    # print("totals = ", totals)
    # print("Sum totals = ", np.sum(totals))

    # print("Mains = ", mains)
    # print("Sum mains = ", np.sum(mains))

    inter = []
    for ii in range(dim):
        for jj in range(ii+1,dim):
            val = SI.get_interaction([ii, jj])
            names.append(str(ii) + str(jj))
            inter.append(val)

    inter = np.array(inter)

    # print("sum = ", np.sum(mains) + np.sum(inter))
    left_over = var - np.sum(mains) - np.sum(inter)
    # print("leftover = ", left_over)
    names.append("other")

    all_sizes = list(mains/var) + list(inter/var) + [left_over/var]
    return names, all_sizes

if __name__ == "__main__":

    fig1, ax1 = plt.subplots(1,3)
    
    coeff = np.array([1.0**(-i) for i in range(dim)])
    names, all_sizes = gen_results(coeff)
    explode = [0] * len(all_sizes)
    explode[1] = 0.1;
    ax1[0].pie(all_sizes, explode=explode, labels=names, autopct='%.1f%%')
    ax1[0].axis('equal')
    ax1[0].set_title("Equal Coefficients")


    coeff = np.array([1.1**(-i) for i in range(dim)])
    names, all_sizes = gen_results(coeff)
    explode = [0] * len(all_sizes)
    explode[1] = 0.1;
    ax1[1].pie(all_sizes, explode=explode, labels=names, autopct='%.1f%%')
    ax1[1].axis('equal')
    ax1[1].set_title("1.1^i")

    coeff = np.array([1.5**(-i) for i in range(dim)])
    names, all_sizes = gen_results(coeff)
    explode = [0] * len(all_sizes)
    explode[1] = 0.1;
    ax1[2].pie(all_sizes, explode=explode, labels=names, autopct='%.1f%%')
    ax1[2].axis('equal')
    ax1[2].set_title("1.5^i")
    
    plt.show()



