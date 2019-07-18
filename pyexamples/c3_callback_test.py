import _c3 as c3
import numpy as np

## Define two functions
def func1(x,param=None):
    print("x = ")
    return np.sum(x,axis=1)

def func2(x, param=None):
    print("x = ", x)
    print("param = ", param)
    return np.sum(x, axis=1)

if __name__ == "__main__":

    x = np.random.randn(2,2)
    roar = "what"
    fw = c3.fwrap_create(2, "python")
    c3.fwrap_set_pyfunc(fw, func2, roar)
    c3.test_py_call()
