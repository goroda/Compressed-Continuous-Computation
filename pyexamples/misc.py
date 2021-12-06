# this is only useful to AG (probably)
import c3py # import the python interface to the c3 library
import numpy as np
import _c3

if __name__ == "__main__":

    dim = 3
    pts = np.linspace(0,1,4)
    op = _c3.build_lp_operator(dim, pts)
    print(op.op)
