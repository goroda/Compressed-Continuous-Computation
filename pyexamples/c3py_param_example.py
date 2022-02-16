#%%
import numpy as np
import c3py

np.random.seed(10)


def func(x):
    return x[:, 0]**2 + x[:, 1]**2 + x[:, 0] * x[:, 1] + np.sin(np.sum(x, axis=1)) \
        + x[:,0] * x[:,-1] + x[:, -1]**2.0 * np.cos(x[:, 0])

Ntrain = 1000
X = np.random.rand(Ntrain, 2)
Y = func(X)

Ntest = 1000
Xtest = np.random.rand(Ntest, 2)
Ytest = func(Xtest)

LB = -1.0                           # lower bounds of features
UB =  1.0                           # upper bounds of features



def build_ft_regress(xdata, ydata, nparam=2, init_rank=5, adaptrank=1, verbose=0):
    dim = xdata.shape[1]
    ndata = xdata.shape[0]
    
    ft = c3py.FunctionTrain(dim)
    for ii in range(dim):
        ft.set_dim_opts(ii, "kernel", LB, UB, nparam)

    ranks = [init_rank]*(dim+1)
    ranks[0] = 1
    ranks[dim] = 1
    ft.set_ranks(ranks)
    ft.build_data_model(ndata, xdata, ydata,
                        alg="AIO", obj="LS", adaptrank=adaptrank,
                        kickrank=1, roundtol=1e-10, verbose=verbose,
                        seed=10)
    return ft

print("X shape ", X.shape)
print("Y shape ", Y.shape)

#Build FTP with a trained FT
ft_regress = build_ft_regress(X, Y, nparam=2, init_rank=1, adaptrank=0, verbose=0)
ftp_regress = ft_regress.build_param_model(alg="AIO", obj="LS", adaptrank=0)

#Build FTP with untrained FT
dim = X.shape[1]
ft_new = c3py.FunctionTrain(dim)
for ii in range(dim):
         ft_new.set_dim_opts(ii, "kernel", LB, UB, nparam=2)
         
ftp_new = ft_new.build_param_model(alg="AIO", obj="LS", adaptrank=0)

ftp_regress.free()
#%%

nparams = ftp_regress.get_nparams()
print('Number of parameters: ', nparams)

params1 = ftp_regress.get_params()
params2 = [ftp_regress.get_param(i) for i in range(nparams)]

print("First FT Parameters: ", params1[:2], " Should match: ", params2[:2])


ft_evals = ftp_regress.ft_eval(Xtest)
err = np.linalg.norm(ft_evals - Ytest) / np.linalg.norm(Ytest)
mse = np.square(ft_evals - Ytest).mean()
print("Before Relmse = ", err, " MSE = ", mse)


print("--- GRADIENT DESCENT ---")
params = np.random.randn(nparams)
print("First Random Updated Parameters: ", params[:2])
ftp_regress.update_params(params)


epochs = 300
learning_rate = 5e-3

errors = np.zeros(epochs)
for e in range(epochs):
    
    error_diff = (Y - ftp_regress.ft_eval(X))
    errors[e] = (1/len(Y))*(error_diff**2).sum()
    
    param_grads = np.zeros((len(Y), nparams))
    for d in range(len(Y)):
        _, grad_evals = ftp_regress.grad_eval(X[d])
        param_grads[d] = grad_evals
        
    cost_grads = -(2/len(Y))*np.dot(error_diff, param_grads) 
    
    params = params - learning_rate*cost_grads
    ftp_regress.update_params(params)


ft_evals = ftp_regress.ft_eval(Xtest)
err = np.linalg.norm(ft_evals - Ytest) / np.linalg.norm(Ytest)
mse = np.square(ft_evals - Ytest).mean()
print("Gradient Descent Relmse = ", err, " MSE = ", mse)

# %%
