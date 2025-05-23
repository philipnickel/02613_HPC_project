from os.path import join
import sys

import numpy as np
from numba import jit, njit, prange, cuda
import cupy as cp



def jacobi(u, interior_mask, max_iter, atol=1e-6, print_residual=False, save_residuals=False, use_alg_residual=False):
    u = np.copy(u)
    residuals = []
    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if use_alg_residual is True and i % 100 == 0:
            residuals.append(compute_residual(u, interior_mask))

        if use_alg_residual is False:
            if delta < atol:
                if print_residual is True:
                    print(f"Algebraic Residual: {compute_residual(u, interior_mask)})")
                    print(f"Converged in {i} iterations")
                if save_residuals is True:
                    return u, residuals
                else:
                    return u
        elif use_alg_residual is True:
            if compute_residual(u, interior_mask) < atol:
                if print_residual is True:
                    print(f"Algebraic Residual: {compute_residual(u, interior_mask)})")
                    print(f"Converged in {i} iterations")
                if save_residuals is True:
                    return u, residuals
                else:
                    return u
    #print(f"Failed to converge in {max_iter} iterations")
    if save_residuals is True:
        return u, residuals
    else:
        return u



def jacobi_numba(u, interior_mask, max_iter, atol=1e-6, parallel=False, print_residual=False):
    ys, xs = np.where(interior_mask)
    if parallel: 
        u = numba_helper_parallel(u, ys, xs, interior_mask, max_iter, atol)
        if print_residual:
            print(f"Algebraic Residual: {compute_residual(u, interior_mask)})")
        return u 
    else: 
        u = numba_helper(u, ys, xs, interior_mask,  max_iter, atol)
        if print_residual:
            print(f"Algebraic Residual: {compute_residual(u, interior_mask)})")
        return u 

@njit(parallel=True)
def compute_u_new_parallel(u, u_new):
    nx, ny = u.shape
    for i in prange(1, nx - 1):  # prange enables parallel execution
        for j in range(1, ny - 1):
            u_new[i - 1, j - 1] = 0.25 * (
                u[i, j - 1] + u[i, j + 1] + u[i - 1, j] + u[i + 1, j]
            )


@njit
def numba_helper(u, ys, xs,interior_mask,  max_iter, atol=1e-6):
    u = u.copy()
    
    for it in range(max_iter):
        #if parallel:
            #u_new = compute_u_new_parallel(u)

        #else:
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])

        delta = 0.0
        for k in range(len(ys)):
        #for k in range(len(ys) - 1, -1, -1):  # Reverse iteration here

            i = ys[k]
            j = xs[k]
            diff = abs(u[i + 1, j + 1] - u_new[i, j])
            if diff > delta:
                delta = diff
            u[i + 1, j + 1] = u_new[i, j]

        if delta < atol:
          #print(f"Converged in {it} iterations")
          
          break

    return u

@njit
def numba_helper_parallel(u, ys, xs, interior_mask, max_iter, atol=1e-6):
    u = u.copy()
    nx, ny = u.shape
    u_new = np.empty((nx - 2, ny - 2))
    
    for it in range(max_iter):

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
          #print(f"Converged in {it} iterations")
          
          break

    return u



@cuda.jit
def jacobi_kernel(u, u_new, mask):
    i, j = cuda.grid(2)

    if i >= u.shape[0] or j >= u.shape[1]:
        return  # bounds check (required by DTU guide)

    # Update interior only
    if 1 <= i < u.shape[0] - 1 and 1 <= j < u.shape[1] - 1:
        if mask[i - 1, j - 1]:  # mask corresponds to interior domain
            u_new[i, j] = 0.25 * (
                u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]
            )
        else:
            u_new[i, j] = u[i, j]
    else:
        u_new[i, j] = u[i, j]  # ghost cells or outside active region

def get_2d_grid(shape, tpb=(16, 16)):
    return (
        ((shape[0] + tpb[0] - 1) // tpb[0],
         (shape[1] + tpb[1] - 1) // tpb[1]),
        tpb
    )

def jacobi_cuda(u_host, interior_mask, max_iter, atol=1e-6, interval=10):
    assert u_host.shape == (interior_mask.shape[0] + 2, interior_mask.shape[1] + 2)

    u = cuda.to_device(u_host)
    u_new = cuda.device_array_like(u)
    mask = cuda.to_device(interior_mask)

    tpb = (16, 16)
    bpg, tpb = get_2d_grid(u.shape, tpb)

    for it in range(max_iter):
        jacobi_kernel[bpg, tpb](u, u_new, mask)
        u, u_new = u_new, u  # swap

        if (it + 1) % interval == 0:
            cuda.synchronize()
            u_host_current = u.copy_to_host()
            residual = compute_residual(u_host_current, interior_mask)

            if residual < atol:
                #print(f"Converged in {it + 1} iterations (residual={residual:.2e})")
                return u_host_current

    return u.copy_to_host()
  
def compute_residual(u, interior_mask):
    xp = cp.get_array_module(u)
    lap = (
        u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1]
    )
    return xp.linalg.norm((lap[interior_mask]))



def jacobi_cp(u, interior_mask, max_iter, atol=1e-2):
    u = cp.copy(u)
    u = cp.array(u)
    interior_mask = cp.array(interior_mask)
    interval = 500
    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if (i + 1) % interval == 0:
                residual = compute_residual(u, interior_mask)

                if residual < atol:
                    return u
    #print(f"Failed to converge in {max_iter} iterations")
    return u
