"""An example using c3py for tensor train cross approximation"""
import c3py # import the python interface to the c3 library
import numpy as np


def rank_two_fun(indices):
    
    vals = indices[0] + indices[1] + indices[2]
    
    return vals

def rank_four_fun(indices):
    
    vals = indices[0] + indices[1] + indices[2] + np.sin(indices[0] + indices[1] + indices[2])
    
    return vals

if __name__ == "__main__":

    shape = (5, 3, 9)
    dim = len(shape)
    mode_one = np.arange(0, shape[0], 1)
    mode_two = np.arange(0, shape[1], 1)
    mode_three = np.arange(0, shape[2], 1)
    indices = np.meshgrid(mode_one, mode_two, mode_three, indexing='ij')

    # arr = rank_two_fun(indices) # rank 2 function
    arr = rank_four_fun(indices) # rank 3 function
    
    init_rank = np.array([1, 2, 3, 1])
    verbose = 1
    adapt = 1

    tt = c3py.TensorTrain.cross_from_numpy_tensor(arr, init_rank, verbose, adapt,
                                                  maxrank= 12,
                                                  cross_tol=1e-10,
                                                  round_tol=1e-10,
                                                  kickrank=2,
                                                  maxiter=5)


    check_multiple = np.array([[0, 0, 0],
                      [1, 0, 0,],
                      [1, 1, 0],
                      [0, 0, 1]])
    print(arr[0, 0, 0])
    print(arr[1, 0, 0])
    print(arr[1, 1, 0])
    print(arr[0, 0, 1])
    
    print(tt[0, 0, 1]) 
    print(tt.eval(check_multiple))

    cores = tt.cores # <- to get a list of cores that are [rank, n, rank]
    print([c.shape for c in cores])
    
    # to get the full tensor.
    all_elements = np.array(indices).T.reshape(-1, dim, order='C')
    recon = tt.eval(all_elements)# .reshape(arr.shape, order='F')
    recon = recon.reshape(shape, order='F')

    # compute error
    diff = arr - recon
    print("Norm of the difference = ", np.linalg.norm(diff) / np.linalg.norm(arr))
