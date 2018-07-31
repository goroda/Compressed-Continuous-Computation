import c3py # import the python interface to the c3 library
import numpy as np
import matplotlib.pyplot as plt

## Define two functions
def func1(x,param=None):
    return np.sum(x,axis=1)

def func1_grad(x,param=None):
    return np.ones(x.shape[1])

def func2(x,param=None):
    return np.sin(np.sum(x,axis=1))

def func2_grad(x):
    return np.cos(np.sum(x,axis=1)) 

dim = 2                                # number of features
ndata = 100                            # number of data points
x = np.random.rand(ndata,dim)*2.0-1.0  # training samples
y1 = func1(x)                          # function values 
y2 = func2(x)                          # ditto

lb = -1                                # lower bounds of features
ub = 1                                 # upper bounds of features
nparam = 2                             # number of parameters per univariate function


## Run a rank-adaptive regression routine to approximate the first function
ft = c3py.FunctionTrain(dim)
for ii in range(dim):
    ft.set_dim_opts(ii,"legendre",lb,ub,nparam)
ft.build_data_model(ndata, x, y1, alg="AIO", obj="LS", adaptrank=1,
                    kickrank=1, roundtol=1e-10, verbose=0, store_opt_info=False)


## Run a fixed-rank regression routine to approximate the second function with stochastic gradient descent
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

# ## Select number of parameters through cross validation
# # ftcv = c3py.FunctionTrain(dim)
# # ranks = [4]*(dim+1)
# # ranks[0] = 1
# # ranks[dim] = 1
# # ftcv.set_ranks(ranks)
# # for ii in range(dim):
# #     ftcv.set_dim_opts(ii,"legendre",lb,ub,nparam)
# # ftcv.build_data_model(ndata,x,y2,alg="AIO",obj="LS_SPARSECORE",\
# #                       cvregweight=[1e-10,1e-8,1e-6,1e-4],kfold=3,verbose=0,cvverbose=2)


ft3 = ft + ft2  # add two function-trains
ft4 = ft * ft2  # multiply to function-trains

## Run adaptive sampling scheme
ft_adapt = c3py.FunctionTrain(dim)
for ii in range(dim):
    ft_adapt.set_dim_opts(ii,"legendre",lb,ub,nparam)
verbose=0
init_rank=2
adapt=1
ft_adapt.build_approximation(func2,None,init_rank,verbose,adapt)

ft_lin_adapt = c3py.FunctionTrain(dim)
for ii in range(dim):
    ft_lin_adapt.set_dim_opts(ii,"linelm",lb,ub,80)
verbose=0
init_rank=2
adapt=1
ft_lin_adapt.build_approximation(func2,None,init_rank,verbose,adapt)



ft.save("saving.c3")
ft_load = c3py.FunctionTrain(0)
ft_load.load("saving.c3")
## Generate test point
test_pt = np.random.rand(dim)*2.0-1.0


print("\n\n\n")

print("test_pt = ", test_pt)
grad = ft.grad_eval(test_pt)
grad_should = func1_grad(test_pt.reshape((1,dim)))
print("Grad = ", grad)
print("should be = ",grad_should)

print("\n\n\n")


ft1eval = ft.eval(test_pt) # evaluate the function train
floadeval = ft_load.eval(test_pt) # evaluate the function train
ft2eval = ft2.eval(test_pt)
ft_sgd_eval = ft_sgd.eval(test_pt)
# ftcveval = ftcv.eval(test_pt) 
ft3eval = ft3.eval(test_pt)
ft4eval = ft4.eval(test_pt)
ft_adapt_eval = ft_adapt.eval(test_pt)
ft_lin_adapt_eval = ft_lin_adapt.eval(test_pt)

eval1s = func1(test_pt.reshape((1,dim)))
eval2s = func2(test_pt.reshape((1,dim)))
eval3s = eval1s + eval2s
eval4s = eval1s * eval2s


print("Fteval =",ft1eval, "Should be =",eval1s)
print("Ft_loadeval =",floadeval, "Should be =",eval1s)
print("Second function with BFGS: Fteval =",ft2eval, "Should be =",eval2s)
print("Second function with SGD:  Fteval =",ft_sgd_eval, "Should be =",eval2s)
print("Second function with CrossApproximation:  Fteval =",ft_adapt_eval, "Should be =",eval2s)
print("Second function with CrossApproximation and linear elements:  Fteval =",ft_lin_adapt_eval, "Should be =",eval2s)
# print("Second function with CV:   Fteval =",ftcveval, "Should be =",eval2s)
print("Fteval =",ft3eval, "Should be =",eval3s)
print("Fteval =",ft4eval, "Should be =",eval4s)


print("\n\n\n")
print("Now getting optimization trajectories, run interactively and then type plt.show()")

dim = 5                                # number of features
ndata = 10000                          # number of data points
x = np.random.rand(ndata,dim)*2.0-1.0  # training samples
y1 = func2(x) + np.random.randn(ndata)*0.01  # function values 

lb = -1                                # lower bounds of features
ub = 1                                 # upper bounds of features
nparam = 10                            # number of parameters per univariate function

ranks = [2]*(dim+1)
ranks[0] = 1
ranks[dim] = 1

ft = c3py.FunctionTrain(dim)
ft.set_ranks(ranks)
for ii in range(dim):
    ft.set_dim_opts(ii,"legendre",lb,ub,nparam)
opt_ft = ft.build_data_model(ndata, x, y1, alg="AIO", obj="LS", adaptrank=0, verbose=0, store_opt_info=True)

ft_sgd = c3py.FunctionTrain(dim)
ft_sgd.set_ranks(ranks)
for ii in range(dim):
    ft_sgd.set_dim_opts(ii,"legendre",lb,ub,nparam)
opt_sgd = ft_sgd.build_data_model(ndata, x, y1, alg="AIO", obj="LS",
                                  opt_type="SGD", opt_sgd_learn_rate=1e-4,
                                  adaptrank=0, verbose=0, opt_maxiter=100, store_opt_info=True)

ft_als = c3py.FunctionTrain(dim)
ft_als.set_ranks(ranks)
for ii in range(dim):
    ft_als.set_dim_opts(ii,"legendre",lb,ub,nparam)
opt_als = ft_als.build_data_model(ndata, x, y1, alg="ALS", obj="LS",
                                  adaptrank=0, verbose=0,  store_opt_info=True)


plt.figure()
plt.plot(np.log10(opt_ft), label='AIO')
plt.plot(np.log10(opt_sgd), label='SGD')
plt.plot(np.log10(opt_als), label='ALS')
# plt.semilogy(opt_als, label='ALS')
plt.legend()


# Test serialization


# clean up memory for each function train
# ft.close()
# ft2.close()
# ft_sgd.close()
# ft3.close()
# ft4.close()


