# -*- coding: utf-8 -*-
"""
gr_numerics.py — GPU / Numerical Post-Processing for the Colab GR Calculator
=============================================================================

This module converts symbolic SymPy results from run_computations() into fast
numerical functions and provides visualisation + geodesic integration tools.

GPU acceleration is automatic: the module detects CuPy (CUDA) > JAX > NumPy
in that order and uses the fastest available backend.

Public API
----------
detect_backend()
    Return 'cupy', 'jax', or 'numpy' depending on what is installed.

get_backend(name=None)
    Return the array module (numpy / cupy / jax.numpy).

lambdify_scalar(expr, coords, backend='numpy')
    Convert a SymPy scalar expression to a fast callable.

lambdify_matrix(mat, coords, backend='numpy')
    Convert a SymPy Matrix to a 2-D list of callables.

evaluate_scalar_grid(fn, coord_vals)
    Evaluate a lambdified scalar on a meshgrid of 1-D arrays.

evaluate_results_numerical(results, coords, coord_ranges, npts=50)
    Lambdify and grid-evaluate key GR scalars from a results dict.

plot_gr_quantities(num_results, coords, slice_axes=(0, 1))
    Plot 2-D heat maps of GR scalars using matplotlib.

integrate_geodesic_jax(rhs_fn, y0, tau_span=(0, 10), n_steps=1000)
    Integrate a geodesic ODE with diffrax (JAX) or scipy fallback.

plot_geodesic(tau_arr, y_arr, coords)
    Plot geodesic position components vs affine parameter.
"""

import numpy as np
from sympy import lambdify as sp_lambdify


# ==============================================================================
# Backend detection
# ==============================================================================

def detect_backend():
    """
    Return 'cupy', 'jax', or 'numpy' depending on what is available.

    Order of preference: CuPy (CUDA GPU) > JAX (XLA/GPU/CPU) > NumPy (CPU).
    """
    try:
        import cupy as cp
        cp.array([1.0])    # exercises actual GPU allocation
        return 'cupy'
    except Exception:
        pass
    try:
        import jax
        jax.numpy.array([1.0])
        return 'jax'
    except Exception:
        pass
    return 'numpy'


def get_backend(name=None):
    """
    Return the array module for the requested (or auto-detected) backend.

    Parameters
    ----------
    name : str or None
        'cupy', 'jax', 'numpy', or None (auto-detect).

    Returns
    -------
    Module implementing the numpy API (numpy / cupy / jax.numpy).
    """
    if name is None:
        name = detect_backend()
    if name == 'cupy':
        import cupy as cp
        return cp
    if name == 'jax':
        import jax.numpy as jnp
        return jnp
    return np


# ==============================================================================
# Lambdification helpers
# ==============================================================================

def lambdify_scalar(expr, coords, backend='numpy'):
    """
    Convert a SymPy scalar expression to a fast numerical callable.

    Parameters
    ----------
    expr    : SymPy expression
    coords  : list of SymPy symbols (order must match call arguments)
    backend : 'numpy', 'cupy', or 'jax'

    Returns
    -------
    Callable f(*coord_arrays) → array
    """
    mod = {'cupy': 'cupy', 'jax': 'jax.numpy', 'numpy': 'numpy'}.get(backend, 'numpy')
    return sp_lambdify(coords, expr, modules=mod)


def lambdify_matrix(mat, coords, backend='numpy'):
    """
    Convert a SymPy Matrix to a 2-D list of numerical callables.

    Parameters
    ----------
    mat     : sympy.Matrix
    coords  : list of SymPy symbols
    backend : 'numpy', 'cupy', or 'jax'

    Returns
    -------
    fns : list[list[callable]]  — fns[i][j](*coord_arrays) → scalar array
    """
    rows, cols = mat.shape
    return [
        [lambdify_scalar(mat[i, j], coords, backend) for j in range(cols)]
        for i in range(rows)
    ]


# ==============================================================================
# Grid evaluation
# ==============================================================================

def evaluate_scalar_grid(fn, coord_vals):
    """
    Evaluate a lambdified scalar function on a full coordinate meshgrid.

    Parameters
    ----------
    fn         : callable(*grids) — produced by lambdify_scalar
    coord_vals : list of 1-D arrays, one per coordinate

    Returns
    -------
    Array of shape (len(coord_vals[0]), len(coord_vals[1]), ...)
    """
    grids = np.meshgrid(*coord_vals, indexing='ij')
    return fn(*grids)


def evaluate_results_numerical(results, coords, coord_ranges, npts=50):
    """
    Lambdify and evaluate key GR scalars on a coordinate grid.

    Parameters
    ----------
    results      : dict  — returned by run_computations()
    coords       : list of SymPy symbols
    coord_ranges : dict  {symbol: (min_val, max_val)}
                   Coordinates not in the dict default to the range (0, 1).
    npts         : int   — number of grid points per axis

    Returns
    -------
    num_results : dict  {key: array}
        Keys produced: 'R_scalar', 'K' (Kretschmann), plus any other scalar
        entries in *results* that are non-None SymPy expressions.
    """
    backend = detect_backend()
    xp      = get_backend(backend)
    grid_1d = [
        xp.linspace(*coord_ranges.get(c, (0.0, 1.0)), npts)
        for c in coords
    ]

    scalar_keys = ('R_scalar', 'K')
    num_results = {}
    for key in scalar_keys:
        val = results.get(key)
        if val is None:
            continue
        fn = lambdify_scalar(val, coords, backend)
        try:
            num_results[key] = evaluate_scalar_grid(fn, grid_1d)
        except Exception as exc:
            print(f"[gr_numerics] Warning: could not evaluate {key}: {exc}")
    return num_results


# ==============================================================================
# Visualisation
# ==============================================================================

def plot_gr_quantities(num_results, coords, slice_axes=(0, 1)):
    """
    Plot numerical GR scalars as 2-D heat maps using matplotlib.

    For metrics with more than 2 coordinates a central slice is taken along
    the remaining axes.

    Parameters
    ----------
    num_results : dict  {key: array}  — from evaluate_results_numerical()
    coords      : list of SymPy symbols (for axis labels)
    slice_axes  : (int, int)  indices of the two axes to use as x / y
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[gr_numerics] matplotlib not available — skipping plots.")
        return

    ax0, ax1 = slice_axes
    xlabel = str(coords[ax0])
    ylabel = str(coords[ax1])

    for key, arr in num_results.items():
        # Convert CuPy / JAX array to NumPy for matplotlib
        try:
            data_full = arr.get()        # CuPy
        except AttributeError:
            data_full = np.array(arr)    # JAX or already NumPy

        # Take a central slice along all axes that are not ax0 / ax1
        sl = [data_full.shape[i] // 2 for i in range(data_full.ndim)]
        sl[ax0] = slice(None)
        sl[ax1] = slice(None)
        data = data_full[tuple(sl)]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(data.T, origin='lower', aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(key)
        plt.tight_layout()
        plt.show()


# ==============================================================================
# Geodesic integration
# ==============================================================================

def integrate_geodesic_jax(rhs_fn, y0, tau_span=(0.0, 10.0), n_steps=1000):
    """
    Integrate a geodesic ODE using diffrax (JAX) when available, with a
    scipy.integrate.odeint fallback for CPU-only environments.

    Parameters
    ----------
    rhs_fn   : callable(tau, y) → dy/dtau
               Standard first-order ODE interface.  y has length 2*dim
               (first dim entries = positions, next dim = velocities).
    y0       : array-like, length 2*dim  — initial state
    tau_span : (float, float)  — (tau_start, tau_end) affine parameter range
    n_steps  : int             — number of output points

    Returns
    -------
    tau_arr : 1-D numpy array, shape (n_steps,)
    y_arr   : 2-D numpy array, shape (n_steps, 2*dim)
    """
    tau_arr = np.linspace(tau_span[0], tau_span[1], n_steps)

    try:
        import jax
        import jax.numpy as jnp
        import diffrax

        @jax.jit
        def _rhs_jax(t, y, args):
            return jnp.array(rhs_fn(t, y))

        term   = diffrax.ODETerm(_rhs_jax)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=jnp.array(tau_arr))
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=float(tau_span[0]),
            t1=float(tau_span[1]),
            dt0=(tau_span[1] - tau_span[0]) / n_steps,
            y0=jnp.array(y0, dtype=float),
            saveat=saveat,
        )
        return np.array(tau_arr), np.array(sol.ys)

    except ImportError:
        # diffrax or jax not available — fall back to scipy
        from scipy.integrate import odeint
        y_arr = odeint(rhs_fn, y0, tau_arr, tfirst=True)
        return tau_arr, y_arr


def plot_geodesic(tau_arr, y_arr, coords):
    """
    Plot geodesic position components vs affine parameter τ.

    Parameters
    ----------
    tau_arr : 1-D array of affine parameter values
    y_arr   : 2-D array, shape (n_steps, 2*dim)
              Columns 0..dim-1 are positions; dim..2*dim-1 are velocities.
    coords  : list of SymPy symbols (for subplot titles and y-axis labels)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[gr_numerics] matplotlib not available — skipping geodesic plot.")
        return

    dim = len(coords)
    fig, axes = plt.subplots(1, dim, figsize=(4 * dim, 4), squeeze=False)
    axes = axes[0]

    for i, (ax, c) in enumerate(zip(axes, coords)):
        ax.plot(tau_arr, y_arr[:, i])
        ax.set_xlabel(r'$\tau$  (affine parameter)')
        ax.set_ylabel(str(c))
        ax.set_title(f'Geodesic: {c}(τ)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Geodesic Trajectory', fontsize=13)
    plt.tight_layout()
    plt.show()
