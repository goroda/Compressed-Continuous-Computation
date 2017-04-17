import c3
import numpy as np


optimizer = c3.c3opt_create(c3.BFGS)

a = np.random.randn(5)
#print(a)

lb = -1
ub = 1
dim = 3

opts = c3.ope_opts_alloc(c3.LEGENDRE)
c3.ope_opts_set_lb(opts,lb)
c3.ope_opts_set_ub(opts,ub)

qmopts = c3.one_approx_opts_alloc(c3.POLYNOMIAL,opts)
multiopts = c3.multi_approx_opts_alloc(dim)

for ii in range(dim):
    c3.multi_approx_opts_set_dim(multiopts,ii,qmopts)

ranks=[1,2,1]
reg = c3.ft_regress_alloc(dim,multiopts,ranks)
c3.ft_regress_set_alg_and_obj(reg,c3.AIO,c3.FTLS);
c3.ft_regress_set_kickrank(reg,2)

ndata = 10
x = np.random.randn(10,dim)
y = np.sum(x,axis=1)

ft = c3.ft_regress_run(reg,optimizer,ndata,x.flatten(),y)    
