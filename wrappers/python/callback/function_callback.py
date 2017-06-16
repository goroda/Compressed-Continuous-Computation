from pycback import *

def eval(a,b,params=None):
    if params is not None:
        return a+b+params["c"]
    return a+b


class obj():
    def __init__(self):
        self.Cobj = alloc_cobj()
    def set_function(self,f,params=None):
        assign(self.Cobj,f,params)
    def eval(self,a,b):
        return call_obj(a,b,self.Cobj)

if __name__ == '__main__':

    params = {"c":3}
    
    a = obj()
    a.set_function(eval)
    x = a.eval(1,2)
    
 
    b = obj()
    b.set_function(eval,params)
    y = b.eval(1,2)
    print(x)
    print(y)
