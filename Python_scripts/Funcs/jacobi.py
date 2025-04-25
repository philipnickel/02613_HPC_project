from os.path import join
import sys

import numpy as np
from numba import jit

#@profile
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u

def jacobi_optimized(u, interior_mask, max_iter, atol=1e-6):
   #u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u

@jit(nopython=True)
def jacobi_numba(u, interior_mask, max_iter, atol=1e-6):
    #u = np.copy(u)
    shape = u.shape
    u_flat = u.ravel()
    n = shape[1]

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = np.empty((shape[0]-2) * (shape[1]-2))
        for j in range(1, shape[0]-1):
            for k in range(1, shape[1]-1):
                idx = j*n + k
                u_new[(j-1)*(shape[1]-2) + (k-1)] = 0.25 * (
                    u_flat[idx-1] +      # left
                    u_flat[idx+1] +      # right 
                    u_flat[idx-n] +      # up
                    u_flat[idx+n]        # down
                )
        
        u_new_interior = u_new[interior_mask]
        interior_indices = np.where(interior_mask)[0]
        flat_indices = np.array([(j+1)*n + (k+1) for j,k in 
                                enumerate(interior_indices//(shape[1]-2))])
        
        delta = np.abs(u_flat[flat_indices] - u_new_interior).max()
        u_flat[flat_indices] = u_new_interior

        if delta < atol:
            break
            
    return u_flat.reshape(shape)

