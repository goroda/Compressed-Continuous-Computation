from pycback import *

import os
import sys
sys.path.append("/ascldap/users/aagorod/Software/c3/wrappers/python")
import numpy as np
import c3
import c3py

import ctypes

def eval(a,b,params=None):
    if params is not None:
        return a+b+params["c"]
    return a+b

def eval_arra(a,params=None):
    print(a)
    b = a + 1
    print(b.flatten())
    return b.flatten()

def eval_arrb(a,params=None):
    b = a[:,0] + a[:,1]
    return b.flatten()

class obj():
    def __init__(self):
        self.Cobj = alloc_cobj()
    def set_function(self,f,dim,params=None):
        assign(self.Cobj,dim,f,params)
    def eval(self,a,b):
        return call_obj(a,b,self.Cobj)
    def wrap_for_c3(self):
        return pyfunc_to_fwrap(self.Cobj)
    
if __name__ == '__main__':

    params = {"c":3}
    
    d = obj()
    d.set_function(eval_arrb,2,None)
    
    fw = c3.fwrap_create(2,"python")
    c3.fwrap_set_pyfunc(fw,d.Cobj)
    typeis = c3.fwrap_get_type(fw)
    # print(typeis)

    ft = c3py.FunctionTrain(2)
    for ii in range(2):
        ft.set_dim_opts(ii,"legendre",-1,1,10)
        
    verbose = 1
    init_rank = 2
    adapt = 0
    ft.build_approximation(fw,init_rank,verbose,adapt)
    
    pt = np.zeros((2))
    print(ft.eval(pt))

    pt[0] = 0.5
    pt[1] = -0.8
    print(ft.eval(pt))

    # print(wrapped_f)
    # c3.fwrap_set_which_eval(wrapped_f,0)

    
    
