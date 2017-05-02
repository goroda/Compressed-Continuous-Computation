import c3py # import the python interface to the c3 library
import numpy as np

## Define two functions
def func1(x):
    return np.sum(x,axis=1)

def func2(x):
    return np.sin(np.sum(x,axis=1))


dim = 5                                # number of features
ndata = 1000                           # number of data points
x = np.random.rand(ndata,dim)*2.0-1.0  # training samples
y1 = func1(x)                          # function values 
y2 = func2(x)                          # ditto

lb = -1                                # lower bounds of features
ub = 1                                 # upper bounds of features
nparam = 10                            # number of parameters per univariate function


## Run a rank-adaptive regression routine to approximate the first function
ft = c3py.FunctionTrain(dim)
for ii in range(dim):
    ft.set_dim_opts(ii,"legendre",lb,ub,nparam)
ft.build_data_model(ndata,x,y1,alg="AIO",obj="LS",adaptrank=1,kickrank=1,roundtol=1e-10,verbose=0)


## Run a fixed-rank regression routine to approximate the second function
ft2 = c3py.FunctionTrain(dim)
ranks = [2]*(dim+1)
ranks[0] = 1
ranks[dim] = 1
ft2.set_ranks(ranks)
for ii in range(dim):
    ft2.set_dim_opts(ii,"legendre",lb,ub,nparam)
ft2.build_data_model(ndata,x,y2,alg="AIO",obj="LS",kristoffel=True,verbose=1)


ft3 = ft + ft2  # add two function-trains
ft4 = ft * ft2  # multiply to function-trains


## Generate test point
test_pt = np.random.rand(dim)*2.0-1.0
ft1eval = ft.eval(test_pt) # evaluate the function train
ft2eval = ft2.eval(test_pt) 
ft3eval = ft3.eval(test_pt)
ft4eval = ft4.eval(test_pt)
eval1s = func1(test_pt.reshape((1,dim)))
eval2s = func2(test_pt.reshape((1,dim)))
eval3s = eval1s + eval2s
eval4s = eval1s * eval2s

print("Fteval =",ft1eval, "Should be =",eval1s)
print("Fteval =",ft2eval, "Should be =",eval2s)
print("Fteval =",ft3eval, "Should be =",eval3s)
print("Fteval =",ft4eval, "Should be =",eval4s)


# clean up memory for each function train
ft.close()
ft2.close()
ft3.close()
ft4.close()

