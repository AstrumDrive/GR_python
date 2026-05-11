# -*- coding: utf-8 -*-
"""
gr_geodesic_numeric.py — Numerical Geodesic Integration

Integrates the geodesic equation numerically using lambdified Christoffel
symbols and scipy.integrate.solve_ivp.  Conservation quantities are checked
along the computed trajectory.

Provides:
  • lambdify_christoffel()   — convert symbolic Γ to fast numpy callables
  • integrate_geodesic()     — solve the geodesic ODE for given ICs
  • check_conserved()        — verify ε (norm), E (energy), L (ang. mom.) along geodesic
  • plot_geodesic()          — matplotlib 2-D and 3-D trajectory plots

All symbolic → numeric conversions happen through SymPy's lambdify with the
numpy backend, so the integrator is fast enough for visualisation and quick
checks even for complex metrics like Kerr.

References
----------
• Misner, Thorne, Wheeler, "Gravitation" (1973) §25
• Drasco & Hughes, PRD 69 (2004) 044019  (Kerr geodesics)
"""

import numpy as np
import sympy as sp
from sympy import lambdify, cancel, symbols
from gr_tensors import progress


# ==============================================================================
# LAMBDIFY CHRISTOFFEL SYMBOLS
# ==============================================================================

def lambdify_christoffel(Gamma, coords, dim=4, parameter_subs=None):
    """
    Convert symbolic Christoffel symbols to fast numpy callables.

    Parameters
    ----------
    Gamma         : 3D list  Gamma[λ][μ][ν] — SymPy expressions
    coords        : list of coordinate symbols
    dim           : int
    parameter_subs: dict  {symbol: float_value}  — numeric values for M, a, Q, etc.

    Returns
    -------
    Gamma_num : 3D list of callables
        Gamma_num[λ][μ][ν](*coord_values) → float
    free_symbols : set  — symbols that still need numeric values (not in coords)
    """
    progress("Lambdifying Christoffel symbols for numerical integration...")

    # Collect free symbols beyond coordinates
    coord_set = set(coords)
    all_free  = set()
    for lam in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                try:
                    all_free |= Gamma[lam][mu][nu].free_symbols
                except AttributeError:
                    pass
    free_params = all_free - coord_set

    if parameter_subs:
        ps = parameter_subs
    else:
        ps = {}

    Gamma_num = [[[None]*dim for _ in range(dim)] for _ in range(dim)]

    for lam in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                expr = cancel(Gamma[lam][mu][nu])
                # Substitute numeric parameters
                if ps:
                    expr = expr.subs(ps)
                try:
                    f = lambdify(coords, expr, modules='numpy')
                    Gamma_num[lam][mu][nu] = f
                except Exception:
                    # Fallback: return a constant zero callable
                    Gamma_num[lam][mu][nu] = lambda *args: 0.0

    remaining = free_params - set(ps.keys())
    if remaining:
        progress(f"  WARNING: These symbols have no numeric value: {remaining}")
        progress(f"  Pass them in parameter_subs to get correct integration.")

    progress(f"  Lambdification complete. {dim**3} Christoffel components ready.")
    return Gamma_num, remaining


# ==============================================================================
# LAMBDIFY METRIC
# ==============================================================================

def lambdify_metric(g, ginv, coords, dim=4, parameter_subs=None):
    """
    Return numpy callables for g_{μν} and g^{μν}.

    Returns
    -------
    g_num, ginv_num : each a dim×dim list of callables
    """
    ps = parameter_subs or {}

    def make_grid(mat):
        grid = [[None]*dim for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                expr = cancel(mat[i, j]).subs(ps)
                try:
                    grid[i][j] = lambdify(coords, expr, modules='numpy')
                except Exception:
                    grid[i][j] = lambda *args: 0.0
        return grid

    return make_grid(g), make_grid(ginv)


# ==============================================================================
# GEODESIC ODE
# ==============================================================================

def _build_geodesic_rhs(Gamma_num, dim=4):
    """
    Build the RHS function for scipy.integrate.solve_ivp.

    State vector: y = [x^0, x^1, ..., x^{dim-1}, u^0, u^1, ..., u^{dim-1}]
    where u^μ = dx^μ/dλ.

    ODE:
        dx^μ/dλ = u^μ
        du^μ/dλ = −Γ^μ_{αβ} u^α u^β
    """
    def rhs(lam, y):
        x = y[:dim]
        u = y[dim:]
        dxdl = u.copy()
        dudl = np.zeros(dim)
        for mu in range(dim):
            acc = 0.0
            for a in range(dim):
                if u[a] == 0.0: continue
                for b in range(dim):
                    if u[b] == 0.0: continue
                    G = Gamma_num[mu][a][b](*x)
                    if not np.isfinite(G):
                        G = 0.0
                    acc += G * u[a] * u[b]
            dudl[mu] = -acc
        return np.concatenate([dxdl, dudl])

    return rhs


def integrate_geodesic(Gamma_num, x0, u0, lambda_span,
                       dim=4, dense_output=True,
                       rtol=1e-9, atol=1e-11,
                       max_step=np.inf):
    """
    Integrate the geodesic equation numerically.

    Parameters
    ----------
    Gamma_num    : 3D list of callables from lambdify_christoffel()
    x0           : array-like, shape (dim,)  — initial coordinates
    u0           : array-like, shape (dim,)  — initial 4-velocity
    lambda_span  : (λ_start, λ_end)          — affine parameter range
    dim          : int
    dense_output : bool — return continuous interpolant
    rtol, atol   : float — solver tolerances
    max_step     : float — maximum step size

    Returns
    -------
    sol : scipy.integrate.OdeSolution object
        sol.t            — affine parameter values
        sol.y[:dim]      — trajectory x^μ(λ)
        sol.y[dim:]      — 4-velocity u^μ(λ)
    """
    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        raise ImportError("scipy is required for numerical geodesic integration. "
                          "Install with:  pip install scipy")

    progress(f"Integrating geodesic over λ ∈ [{lambda_span[0]}, {lambda_span[1]}]...")
    progress(f"  Initial position:   {np.array(x0)}")
    progress(f"  Initial 4-velocity: {np.array(u0)}")

    y0 = np.concatenate([np.array(x0, dtype=float), np.array(u0, dtype=float)])
    rhs = _build_geodesic_rhs(Gamma_num, dim)

    sol = solve_ivp(
        rhs, lambda_span, y0,
        method='RK45',
        dense_output=dense_output,
        rtol=rtol, atol=atol,
        max_step=max_step,
    )

    if sol.success:
        n_steps = len(sol.t)
        progress(f"  Integration successful: {n_steps} steps, "
                 f"λ_final = {sol.t[-1]:.4f}")
    else:
        progress(f"  WARNING: Integration failed — {sol.message}")

    return sol


# ==============================================================================
# CONSERVATION CHECKS
# ==============================================================================

def check_conserved(sol, g_num, ginv_num, killing_indices=None, dim=4):
    """
    Verify conserved quantities along the integrated geodesic.

    Checks:
      ε = g_{μν} u^μ u^ν  (geodesic norm: = −1 timelike, 0 null, +1 spacelike)
      E = −g_{tμ} u^μ     (energy, if ∂/∂t is Killing: index 0)
      L =  g_{φμ} u^μ     (angular momentum, if ∂/∂φ is Killing: index 3)

    Parameters
    ----------
    sol           : output of integrate_geodesic()
    g_num         : 2D list of callables from lambdify_metric()
    ginv_num      : 2D list of callables (unused here, for future use)
    killing_indices: list of int — coordinate indices with Killing symmetry
                    e.g. [0, 3] for t and φ
    dim           : int

    Returns
    -------
    dict with keys:
        'norm'       : np.array  ε along trajectory
        'norm_drift' : float  (max deviation from initial value)
        'killing_charges': list of np.array  (conserved charge for each Killing index)
    """
    progress("Checking conserved quantities along geodesic...")

    lam_vals = sol.t
    n = len(lam_vals)
    x = sol.y[:dim]   # shape (dim, n)
    u = sol.y[dim:]   # shape (dim, n)

    # Geodesic norm ε = g_{μν} u^μ u^ν
    norm_arr = np.zeros(n)
    for k in range(n):
        xk = x[:, k]
        uk = u[:, k]
        s  = 0.0
        for mu in range(dim):
            for nu in range(dim):
                gval = g_num[mu][nu](*xk)
                s += gval * uk[mu] * uk[nu]
        norm_arr[k] = s

    norm_drift = float(np.max(np.abs(norm_arr - norm_arr[0])))
    progress(f"  Geodesic norm ε₀ = {norm_arr[0]:.6f}  (drift: {norm_drift:.2e})")

    # Killing charges p_α = g_{αμ} u^μ (conserved for cyclic α)
    ki = killing_indices or []
    kc = []
    for a in ki:
        charge = np.zeros(n)
        for k in range(n):
            xk = x[:, k]
            uk = u[:, k]
            s  = 0.0
            for nu in range(dim):
                s += g_num[a][nu](*xk) * uk[nu]
            charge[k] = s
        drift = float(np.max(np.abs(charge - charge[0])))
        progress(f"  Killing charge p_{a}: initial = {charge[0]:.6f}  (drift: {drift:.2e})")
        kc.append(charge)

    return {
        'norm':            norm_arr,
        'norm_drift':      norm_drift,
        'killing_charges': kc,
        'lambda':          lam_vals,
        'x':               x,
        'u':               u,
    }


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_geodesic(conserved, coords, coord_labels=None, output_path=None):
    """
    Plot the geodesic trajectory and conservation diagnostics.

    Produces a 2- or 3-panel figure:
      Panel 1: Orbit in the (r, φ) plane  (Cartesian projection)
      Panel 2: Geodesic norm ε vs λ
      Panel 3: Killing charges vs λ

    Parameters
    ----------
    conserved    : dict from check_conserved()
    coords       : list of coordinate symbols
    coord_labels : list of str — display labels for each coordinate
    output_path  : str or None — if given, save figure to this path (.png/.pdf)

    Returns
    -------
    fig : matplotlib Figure (or None if matplotlib unavailable)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        progress("  matplotlib not available — skipping geodesic plot.")
        return None

    progress("Generating geodesic trajectory plot...")

    lam  = conserved['lambda']
    x    = conserved['x']
    norm = conserved['norm']
    kc   = conserved['killing_charges']
    dim  = x.shape[0]

    labels = coord_labels or [str(c) for c in coords]

    n_panels = 1 + 1 + (1 if kc else 0)
    fig = plt.figure(figsize=(5 * n_panels, 4))
    gs  = gridspec.GridSpec(1, n_panels)

    # ---- Panel 1: orbit ----
    ax0 = fig.add_subplot(gs[0])
    if dim >= 4:
        # Spherical coordinates: convert (r, θ, φ) → (x_cart, y_cart)
        r_idx, theta_idx, phi_idx = 1, 2, 3
        r_arr     = x[r_idx]
        theta_arr = x[theta_idx]
        phi_arr   = x[phi_idx]
        x_cart = r_arr * np.sin(theta_arr) * np.cos(phi_arr)
        y_cart = r_arr * np.sin(theta_arr) * np.sin(phi_arr)
        ax0.plot(x_cart, y_cart, 'b-', lw=1.0, label='Geodesic')
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.set_title('Orbital Projection (x-y plane)')
    else:
        ax0.plot(lam, x[1], 'b-', lw=1.0)
        ax0.set_xlabel('λ')
        ax0.set_ylabel(labels[1])
        ax0.set_title('Radial coordinate vs λ')
    ax0.set_aspect('equal', 'datalim')
    ax0.legend()

    # ---- Panel 2: norm conservation ----
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(lam, norm, 'r-', lw=1.0)
    ax1.axhline(norm[0], color='k', ls='--', lw=0.8, label='Initial value')
    ax1.set_xlabel('λ (affine parameter)')
    ax1.set_ylabel('ε = g_{μν} u^μ u^ν')
    ax1.set_title(f'Geodesic norm (drift = {conserved["norm_drift"]:.2e})')
    ax1.legend()

    # ---- Panel 3: Killing charges ----
    if kc:
        ax2 = fig.add_subplot(gs[2])
        for i, charge in enumerate(kc):
            ax2.plot(lam, charge, lw=1.0, label=f'p_{i}')
        ax2.set_xlabel('λ')
        ax2.set_ylabel('Conserved charge')
        ax2.set_title('Killing charges along geodesic')
        ax2.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        progress(f"  Geodesic plot saved to: {output_path}")

    return fig
