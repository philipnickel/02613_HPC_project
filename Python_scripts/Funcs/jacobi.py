from os.path import join
import sys

import numpy as np
from numba import jit, njit, prange

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


def jacobi_numba(u, interior_mask, max_iter, atol=1e-6, parallel=False):
    ys, xs = np.where(interior_mask)
    if parallel: 
        return numba_helper_parallel(u, ys, xs, max_iter, atol)
    else: 
        return numba_helper(u, ys, xs, max_iter, atol)

@njit(parallel=True)
def compute_u_new_parallel(u, u_new):
    nx, ny = u.shape
    for i in prange(1, nx - 1):  # prange enables parallel execution
        for j in range(1, ny - 1):
            u_new[i - 1, j - 1] = 0.25 * (
                u[i, j - 1] + u[i, j + 1] + u[i - 1, j] + u[i + 1, j]
            )


@njit
def numba_helper(u, ys, xs, max_iter, atol=1e-6):
    u = u.copy()
    
    for _ in range(max_iter):
        #if parallel:
            #u_new = compute_u_new_parallel(u)

        #else:
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])

        delta = 0.0
        for k in range(len(ys)):
            i = ys[k]
            j = xs[k]
            diff = abs(u[i + 1, j + 1] - u_new[i, j])
            if diff > delta:
                delta = diff
            u[i + 1, j + 1] = u_new[i, j]

        if delta < atol:
            break

    return u

@njit
def numba_helper_parallel(u, ys, xs, max_iter, atol=1e-6):
    u = u.copy()
    nx, ny = u.shape
    u_new = np.empty((nx - 2, ny - 2))
    
    for _ in range(max_iter):

        compute_u_new_parallel(u, u_new)

        delta = 0.0
        for k in range(len(ys)):
            i = ys[k]
            j = xs[k]
            diff = abs(u[i + 1, j + 1] - u_new[i, j])
            if diff > delta:
                delta = diff
            u[i + 1, j + 1] = u_new[i, j]

        if delta < atol:
            break

    return u
