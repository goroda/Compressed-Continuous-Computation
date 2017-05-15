import c3py # import the python interface to the c3 library
import numpy as np

## Define two functions
def func1(x):
    return np.sum(x,axis=1)

def func2(x):
    return np.sin(np.sum(x,axis=1))


dim = 5                                # number of features
ndata = 100                            # number of data points
x = np.random.rand(ndata,dim)*2.0-1.0  # training samples
y1 = func1(x)                          # function values 
y2 = func2(x)                          # ditto

lb = -1                                # lower bounds of features
ub = 1                                 # upper bounds of features
nparam = 3                             # number of parameters per univariate function


## Run a rank-adaptive regression routine to approximate the first function
ft = c3py.FunctionTrain(dim)
for ii in range(dim):
    ft.set_dim_opts(ii,"legendre",lb,ub,nparam)
ft.build_data_model(ndata,x,y1,alg="AIO",obj="LS",adaptrank=1,kickrank=1,roundtol=1e-10,verbose=0)


## Run a fixed-rank regression routine to approximate the second function
ft_sgd = c3py.FunctionTrain(dim)
ranks = [2]*(dim+1)
ranks[0] = 1
ranks[dim] = 1
ft_sgd.set_ranks(ranks)
for ii in range(dim):
    ft_sgd.set_dim_opts(ii,"legendre",lb,ub,nparam)
ft_sgd.build_data_model(ndata,x,y2,alg="AIO",obj="LS",opt_type="SGD",verbose=0)


## Run a fixed-rank regression routine to approximate the second function
ft2 = c3py.FunctionTrain(dim)
ranks = [2]*(dim+1)
ranks[0] = 1
ranks[dim] = 1
ft2.set_ranks(ranks)
for ii in range(dim):
    ft2.set_dim_opts(ii,"legendre",lb,ub,nparam)
ft2.build_data_model(ndata,x,y2,alg="AIO",obj="LS",verbose=0)



## Select number of parameters through cross validation
ftcv = c3py.FunctionTrain(dim)
ranks = [2]*(dim+1)
ranks[0] = 1
ranks[dim] = 1
ftcv.set_ranks(ranks)
for ii in range(dim):
    ftcv.set_dim_opts(ii,"legendre",lb,ub,nparam)
ftcv.build_data_model(ndata,x,y2,alg="AIO",obj="LS_SPARSECORE",\
                      cvregweight=[1e-10,1e-8,1e-6,1e-4,1e-3],kfold=3,verbose=0,cvverbose=2)


ft3 = ft + ft2  # add two function-trains
ft4 = ft * ft2  # multiply to function-trains


## Generate test point
test_pt = np.random.rand(dim)*2.0-1.0
ft1eval = ft.eval(test_pt) # evaluate the function train
ft2eval = ft2.eval(test_pt)
ft_sgd_eval = ft_sgd.eval(test_pt)
ftcveval = ftcv.eval(test_pt) 
ft3eval = ft3.eval(test_pt)
ft4eval = ft4.eval(test_pt)
eval1s = func1(test_pt.reshape((1,dim)))
eval2s = func2(test_pt.reshape((1,dim)))
eval3s = eval1s + eval2s
eval4s = eval1s * eval2s

print("Fteval =",ft1eval, "Should be =",eval1s)
print("Second function with BFGS: Fteval =",ft2eval, "Should be =",eval2s)
print("Second function with SGD:  Fteval =",ft_sgd_eval, "Should be =",eval2s)
print("Second function with CV:   Fteval =",ftcveval, "Should be =",eval2s)
print("Fteval =",ft3eval, "Should be =",eval3s)
print("Fteval =",ft4eval, "Should be =",eval4s)


# clean up memory for each function train
ft.close()
ft2.close()
ft_sgd.close()
# ft3.close()
# ft4.close()


