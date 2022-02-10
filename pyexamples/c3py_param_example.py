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


ft = c3py.FunctionTrain(X.shape[1])

ft.build_model(X, Y, LB, UB, nparam=2, init_rank=1, basis="kernel",
                    alg="AIO", obj="LS", adaptrank=0,
                    kickrank=1, roundtol=1e-10, verbose=0)

ft.run_regression(seed=10)

#%%

nparams = ft.get_nparams()
print('Number of parameters: ', nparams)

params1 = ft.get_params()
params2 = [ft.get_param(i) for i in range(nparams)]

print("First FT Parameters: ", params1[:2], " Should match: ", params2[:2])


ft_evals = ft.eval(Xtest)
err = np.linalg.norm(ft_evals - Ytest) / np.linalg.norm(Ytest)
mse = np.square(ft_evals - Ytest).mean()
print("C3 Relmse = ", err, " MSE = ", mse)


print("--- GRADIENT DESCENT ---")
params = np.random.randn(nparams)
print("First Random Updated Parameters: ", params[:2])
ft.update_params(params)


epochs = 300
learning_rate = 5e-3

errors = np.zeros(epochs)
for e in range(epochs):
    
    error_diff = (Y - ft.eval(X))
    errors[e] = (1/len(Y))*(error_diff**2).sum()
    
    param_grads = np.zeros((len(Y), nparams))
    for d in range(len(Y)):
        _, grad_evals = ft.param_grad_eval(X[d])
        param_grads[d] = grad_evals
        
    cost_grads = -(2/len(Y))*np.dot(error_diff, param_grads) 
    
    params = params - learning_rate*cost_grads
    ft.update_params(params)


ft_evals = ft.eval(Xtest)
err = np.linalg.norm(ft_evals - Ytest) / np.linalg.norm(Ytest)
mse = np.square(ft_evals - Ytest).mean()
print("Gradient Descent Relmse = ", err, " MSE = ", mse)


# ftp.free()
# %%
