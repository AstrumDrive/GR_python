# -*- coding: utf-8 -*-
"""
gr_tensors.py — GR Computation Functions

All symbolic tensor computation functions for the GR Calculator.
Imported by gr_latex.py and gr_main.py.

Sections contained:
  0 — Imports
  3 — Computation Functions (Christoffel, Riemann, Ricci, Einstein,
      Kretschmann, Weyl, Bianchi, Geodesics, Killing, Tetrad)
"""

# ==============================================================================
# SECTION 0 — IMPORTS
# ==============================================================================

import sympy as sp
from sympy import (
    symbols, Function, Matrix, sqrt, sin, cos, exp,
    diff, simplify, cancel, trigsimp, Rational, pi,
    latex, zeros, eye, S, Abs, Integer
)
import os
import sys
import time
from datetime import datetime


def progress(msg):
    """
    Print a timestamped progress message to stdout.

    Used throughout computation functions to track long-running symbolic
    calculations. Defined here (the lowest-level module) so that gr_latex.py
    and gr_main.py can both import it from a single, cycle-free location.

    On some local Windows consoles the active encoding is cp1252, which cannot
    represent several Unicode symbols used in status messages. In that case we
    degrade gracefully with ASCII replacements instead of crashing the run.
    """
    ts = time.strftime('%H:%M:%S')
    text = f'[{ts}]  {msg}'
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
        replacements = {
            '\u03c1': 'rho',
            '\u03bc': 'mu',
            '\u03bd': 'nu',
            '\u03b7': 'eta',
            '\u2207': 'nabla',
            '\u2192': '->',
            '\u2260': '!=',
            '\u2713': 'OK',
            '\u2717': 'FAIL',
            '\u2026': '...',
            '\u2014': '-',
        }
        safe_text = text
        for old_char, new_char in replacements.items():
            safe_text = safe_text.replace(old_char, new_char)
        safe = safe_text.encode(encoding, errors='replace').decode(encoding, errors='replace')
        print(safe, flush=True)


# Map coordinate symbol names to LaTeX strings for index labels.
# Defined here (gr_tensors, the lowest-level module) so that
# get_geodesic_equations can use coord_latex without importing from gr_latex,
# which would create a circular import (gr_latex already imports from gr_tensors).
_COORD_LATEX_MAP = {
    't':     't',
    'r':     'r',
    'theta': r'\theta',
    'phi':   r'\varphi',
    'x':     'x',
    'y':     'y',
    'z':     'z',
    'chi':   r'\chi',
    'psi':   r'\psi',
}


def coord_latex(coord_sym):
    """
    Return the LaTeX string for a coordinate symbol.
    Falls back to str(coord_sym) if not in the map.

    Examples: theta → \\theta,  phi → \\varphi,  r → r
    """
    name = str(coord_sym)
    return _COORD_LATEX_MAP.get(name, name)


# ==============================================================================
# SECTION 3 — COMPUTATION FUNCTIONS
# ==============================================================================

def compute_christoffel(g, ginv, coords, dim=4):
    """
    Compute Christoffel symbols of the second kind Γ^λ_{μν}.

    Formula
    -------
    Γ^λ_{μν} = ½ g^{λσ} (∂_μ g_{νσ} + ∂_ν g_{μσ} − ∂_σ g_{μν})

    Symmetry Γ^λ_{μν} = Γ^λ_{νμ} is exploited to halve the work.
    cancel() is used in the inner loop for efficiency.

    Parameters
    ----------
    g     : sympy.Matrix  — covariant metric g_{μν}
    ginv  : sympy.Matrix  — contravariant metric g^{μν}
    coords: list          — coordinate symbols [x^0, x^1, ...]
    dim   : int           — spacetime dimension

    Returns
    -------
    Gamma : 3D list  Gamma[λ][μ][ν]  (SymPy expressions)
    """
    progress("Computing Christoffel symbols...")
    Gamma = [[[S.Zero] * dim for _ in range(dim)] for _ in range(dim)]

    for lam in range(dim):
        for mu in range(dim):
            for nu in range(mu, dim):   # exploit μ ↔ ν symmetry
                s = S.Zero
                for sig in range(dim):
                    s += ginv[lam, sig] * (
                        diff(g[nu, sig], coords[mu])
                        + diff(g[mu, sig], coords[nu])
                        - diff(g[mu, nu], coords[sig])
                    )
                val = cancel(Rational(1, 2) * s)
                Gamma[lam][mu][nu] = val
                Gamma[lam][nu][mu] = val   # symmetry

    nz = sum(1 for lam in range(dim)
               for mu in range(dim)
               for nu in range(mu, dim)
               if Gamma[lam][mu][nu] != S.Zero)
    progress(f"  Done. Non-zero independent Christoffel components: {nz}")
    return Gamma


def compute_riemann(Gamma, coords, dim=4):
    """
    Compute the Riemann curvature tensor R^ρ_{σμν}.

    Formula
    -------
    R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} − ∂_ν Γ^ρ_{μσ}
              + Γ^ρ_{μλ} Γ^λ_{νσ} − Γ^ρ_{νλ} Γ^λ_{μσ}

    Antisymmetry in (μ,ν) is enforced after each (ρ,σ,μ,ν) evaluation.
    Progress is reported after each ρ iteration.

    Complexity: O(dim^4) with O(dim) work per entry → O(dim^5) overall.
    For dim=4 this is 256 components, each requiring O(4) operations.

    WARNING: This is the most expensive step. For symbolic Function metrics,
    SymPy must differentiate products of functions — can take 1–10 minutes.
    Use FAST_MODE=True if you only need Christoffel/Ricci/Einstein.

    Parameters
    ----------
    Gamma  : 3D list  — Christoffel symbols from compute_christoffel()
    coords : list     — coordinate symbols
    dim    : int      — spacetime dimension

    Returns
    -------
    R : 4D list  R[ρ][σ][μ][ν]  (SymPy expressions)
    """
    progress("Computing Riemann tensor (may take several minutes for symbolic metrics)...")
    R = [[[[S.Zero] * dim for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]

    for rho in range(dim):
        for sig in range(dim):
            for mu in range(dim):
                for nu in range(mu + 1, dim):   # μ < ν; antisymmetry gives μ > ν
                    term1 = diff(Gamma[rho][nu][sig], coords[mu])
                    term2 = diff(Gamma[rho][mu][sig], coords[nu])
                    term3 = sum(Gamma[rho][mu][lam] * Gamma[lam][nu][sig]
                                for lam in range(dim))
                    term4 = sum(Gamma[rho][nu][lam] * Gamma[lam][mu][sig]
                                for lam in range(dim))
                    val = cancel(term1 - term2 + term3 - term4)
                    R[rho][sig][mu][nu] =  val
                    R[rho][sig][nu][mu] = -val  # antisymmetry
        progress(f"  Riemann: ρ = {rho+1}/{dim} complete")

    nz = sum(1 for rho in range(dim)
               for sig in range(dim)
               for mu in range(dim)
               for nu in range(mu + 1, dim)
               if R[rho][sig][mu][nu] != S.Zero)
    progress(f"  Done. Independent non-zero Riemann components: {nz}")
    return R


def compute_ricci(R_riem, dim=4):
    """
    Compute the Ricci tensor R_{μν} = R^ρ_{μρν}.

    This is a contraction (trace) of the Riemann tensor on indices ρ and the
    third position (σ = μ → first slot, ρ loops over all values).

    Parameters
    ----------
    R_riem : 4D list  — Riemann tensor from compute_riemann()
    dim    : int

    Returns
    -------
    Ric : sympy.Matrix  dim × dim
    """
    progress("Computing Ricci tensor...")
    Ric = zeros(dim, dim)
    for mu in range(dim):
        for nu in range(dim):
            Ric[mu, nu] = cancel(
                sum(R_riem[rho][mu][rho][nu] for rho in range(dim))
            )
    return Ric


def compute_ricci_scalar(Ric, ginv, dim=4):
    """
    Compute the Ricci (curvature) scalar R = g^{μν} R_{μν}.

    For vacuum solutions this will be zero.
    For FRW this is proportional to H² + Ḣ where H = ȧ/a.

    Parameters
    ----------
    Ric  : sympy.Matrix  — Ricci tensor
    ginv : sympy.Matrix  — contravariant metric
    dim  : int

    Returns
    -------
    R_scalar : SymPy expression (scalar)
    """
    progress("Computing Ricci scalar...")
    R_scalar = S.Zero
    for mu in range(dim):
        for nu in range(dim):
            R_scalar += ginv[mu, nu] * Ric[mu, nu]
    return cancel(R_scalar)


def compute_einstein(Ric, g, ginv, R_scalar, dim=4):
    """
    Compute the Einstein tensor G_{μν} = R_{μν} − ½ g_{μν} R.

    The Einstein field equations state:  G_{μν} = 8π T_{μν}
    (with G = c = 1).  So G_{μν} encodes the matter/energy content.

    Parameters
    ----------
    Ric      : sympy.Matrix  — Ricci tensor
    g        : sympy.Matrix  — covariant metric
    ginv     : sympy.Matrix  — contravariant metric (unused directly but
                               kept for signature consistency)
    R_scalar : SymPy expr    — Ricci scalar
    dim      : int

    Returns
    -------
    G : sympy.Matrix  dim × dim
    """
    progress("Computing Einstein tensor...")
    G = zeros(dim, dim)
    for mu in range(dim):
        for nu in range(dim):
            G[mu, nu] = cancel(Ric[mu, nu] - Rational(1, 2) * g[mu, nu] * R_scalar)
    return G


def compute_kretschmann(R_riem, g, ginv, dim=4):
    """
    Compute the Kretschmann scalar K = R^{μνρσ} R_{μνρσ}.

    Physical significance
    ---------------------
    The Kretschmann scalar is a curvature invariant that remains finite at
    coordinate singularities but diverges at genuine (physical) curvature
    singularities. For Schwarzschild: K = 48 M² / r⁶.

    Algorithm
    ---------
    1. Lower first index: R_{αβγδ} = g_{αε} R^ε_{βγδ}
    2. Contract fully:    K = g^{αμ} g^{βν} g^{γρ} g^{δσ} R_{αβγδ} R_{μνρσ}
       Equivalently:      K = Σ_{αβγδ} R^{αβγδ} R_{αβγδ}

    WARNING: This is an O(dim^8) double sum in the general case.
    For diagonal metrics it reduces dramatically (only dim non-zero ginv entries).
    For complex symbolic metrics this may take 10–60 minutes.
    Consider FAST_MODE = True to skip.

    Parameters
    ----------
    R_riem : 4D list         — Riemann tensor (mixed indices R^ρ_{σμν})
    g      : sympy.Matrix    — covariant metric
    ginv   : sympy.Matrix    — contravariant metric
    dim    : int

    Returns
    -------
    K : SymPy expression (scalar)
    """
    progress("Computing Kretschmann scalar (expensive — may take many minutes)...")

    # Step 1: lower first index to get R_{abcd}
    progress("  Step 1: lowering first Riemann index...")
    R_low = [[[[S.Zero] * dim for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                for d in range(dim):
                    R_low[a][b][c][d] = cancel(
                        sum(g[a, e] * R_riem[e][b][c][d] for e in range(dim))
                    )
        progress(f"  Lowering index: a = {a+1}/{dim}")

    # Step 2: form K = R^{abcd} R_{abcd}
    # R^{abcd} = g^{ae} g^{bf} g^{cg} g^{dh} R_{efgh}
    # For sparse ginv (diagonal metrics), most terms are zero.
    progress("  Step 2: contracting to form scalar...")
    K = S.Zero
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                for d in range(dim):
                    if R_low[a][b][c][d] == S.Zero:
                        continue
                    # Raise all four indices
                    R_up = S.Zero
                    for e in range(dim):
                        if ginv[a, e] == S.Zero:
                            continue
                        for ff in range(dim):
                            if ginv[b, ff] == S.Zero:
                                continue
                            for gg in range(dim):
                                if ginv[c, gg] == S.Zero:
                                    continue
                                for h in range(dim):
                                    if ginv[d, h] == S.Zero:
                                        continue
                                    R_up += (ginv[a, e] * ginv[b, ff]
                                             * ginv[c, gg] * ginv[d, h]
                                             * R_low[e][ff][gg][h])
                    K += cancel(R_up) * R_low[a][b][c][d]
        progress(f"  Contraction: a = {a+1}/{dim}")

    K = cancel(K)
    progress(f"  Kretschmann scalar computed.")
    return K


def compute_weyl(R_riem, Ric, g, ginv, R_scalar, dim=4):
    """
    Compute the Weyl conformal tensor C_{ρσμν} (all lower indices).

    Physical significance
    ---------------------
    The Weyl tensor is the trace-free part of the Riemann tensor. It
    represents pure gravitational degrees of freedom — tidal forces and
    gravitational waves — decoupled from the local energy content.
    • Vanishes identically in 3D.
    • Vanishes for conformally flat spacetimes (e.g., FRW).
    • For vacuum (R_{μν} = 0, R = 0): Weyl = Riemann.

    Formula (4D, n = 4)
    -------------------
    C_{ρσμν} = R_{ρσμν}
             − 1/(n−2) (g_{ρμ} R_{σν} − g_{ρν} R_{σμ}
                        + g_{σν} R_{ρμ} − g_{σμ} R_{ρν})
             + R/((n−1)(n−2)) (g_{ρμ} g_{σν} − g_{ρν} g_{σμ})

    Only defined for dim ≥ 3. Returns None for dim < 3.

    Parameters
    ----------
    R_riem   : 4D list       — Riemann tensor R^ρ_{σμν}
    Ric      : sympy.Matrix  — Ricci tensor
    g        : sympy.Matrix  — covariant metric
    ginv     : sympy.Matrix  — contravariant metric (unused here but consistent)
    R_scalar : SymPy expr    — Ricci scalar
    dim      : int

    Returns
    -------
    C : 4D list  C[ρ][σ][μ][ν]  (all indices lowered), or None if dim < 3
    """
    if dim < 3:
        progress("  Weyl tensor undefined for dim < 3. Skipping.")
        return None

    progress("Computing Weyl tensor...")

    # First lower the first index of Riemann: R_{abcd} = g_{ae} R^e_{bcd}
    R_low = [[[[S.Zero] * dim for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                for d in range(dim):
                    R_low[a][b][c][d] = cancel(
                        sum(g[a, e] * R_riem[e][b][c][d] for e in range(dim))
                    )

    n = dim
    C = [[[[S.Zero] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]

    for rho in range(n):
        for sig in range(n):
            for mu in range(n):
                for nu in range(n):
                    riemann_term = R_low[rho][sig][mu][nu]
                    ricci_term = Rational(1, n - 2) * (
                        g[rho, mu] * Ric[sig, nu]
                        - g[rho, nu] * Ric[sig, mu]
                        + g[sig, nu] * Ric[rho, mu]
                        - g[sig, mu] * Ric[rho, nu]
                    )
                    scalar_term = (R_scalar / ((n - 1) * (n - 2))) * (
                        g[rho, mu] * g[sig, nu] - g[rho, nu] * g[sig, mu]
                    )
                    C[rho][sig][mu][nu] = cancel(
                        riemann_term - ricci_term + scalar_term
                    )
        progress(f"  Weyl: ρ = {rho+1}/{n}")

    nz = sum(1 for rho in range(n)
               for sig in range(n)
               for mu in range(n)
               for nu in range(mu + 1, n)
               if C[rho][sig][mu][nu] != S.Zero)
    progress(f"  Done. Independent non-zero Weyl components: {nz}")
    return C


def transform_to_ortho(G_coord, e_matrix, dim=4):
    """
    Project the Einstein tensor into the orthonormal (tetrad) frame.

    Formula
    -------
    Ĝ_{ab} = e^μ_a e^ν_b G_{μν}

    where e^μ_a are the tetrad legs satisfying g_{μν} e^μ_a e^ν_b = η_{ab}.

    This gives the physical stress-energy tensor components directly:
      ρ  = Ĝ_{00} / 8π    (energy density)
      p_i = Ĝ_{ii} / 8π   (principal pressures)

    Parameters
    ----------
    G_coord  : sympy.Matrix  — Einstein tensor in coordinate basis
    e_matrix : sympy.Matrix  — tetrad e_matrix[μ, a] = e^μ_a
    dim      : int

    Returns
    -------
    G_ortho : sympy.Matrix  dim × dim
    """
    progress("Projecting Einstein tensor to orthonormal frame...")
    G_ortho = zeros(dim, dim)
    for a in range(dim):
        for b in range(dim):
            s = S.Zero
            for mu in range(dim):
                for nu in range(dim):
                    s += e_matrix[mu, a] * e_matrix[nu, b] * G_coord[mu, nu]
            G_ortho[a, b] = cancel(s)
    return G_ortho


def check_bianchi(G_coord, ginv, Gamma, coords, dim=4):
    """
    Verify the contracted Bianchi identity: ∇_μ G^{μν} = 0.

    Formula (covariant divergence)
    -------
    ∇_μ G^{μν} = ∂_μ G^{μν}
               + Γ^μ_{μλ} G^{λν}
               + Γ^ν_{μλ} G^{μλ}

    This should vanish identically as a consequence of the second Bianchi
    identity for the Riemann tensor. A non-zero result indicates a bug.

    Parameters
    ----------
    G_coord : sympy.Matrix  — Einstein tensor G_{μν} (covariant)
    ginv    : sympy.Matrix  — contravariant metric
    Gamma   : 3D list       — Christoffel symbols
    coords  : list          — coordinate symbols
    dim     : int

    Returns
    -------
    divergence : list of dim SymPy expressions (each should be 0)
    """
    progress("Verifying Bianchi identity ∇_μ G^{μν} = 0...")
    # Raise both indices: G^{μν} = g^{μρ} g^{νσ} G_{ρσ}
    G_up = zeros(dim, dim)
    for mu in range(dim):
        for nu in range(dim):
            G_up[mu, nu] = cancel(
                sum(
                    ginv[mu, rho] * ginv[nu, sigma] * G_coord[rho, sigma]
                    for rho in range(dim)
                    for sigma in range(dim)
                )
            )

    raw_divergence = []
    for nu in range(dim):
        s = S.Zero
        for mu in range(dim):
            s += diff(G_up[mu, nu], coords[mu])
            for lam in range(dim):
                s += Gamma[mu][mu][lam] * G_up[lam, nu]
                s += Gamma[nu][mu][lam] * G_up[mu, lam]
        raw_divergence.append(cancel(s))

    # Two-step simplification: cancel() first (fast), then simplify() if needed.
    # For metrics with abstract symbolic functions (e.g. B(r), β(r)), cancel()
    # may not reduce long derivative polynomials to zero — simplify() is stronger
    # and handles more derivative identities.
    divergence = []
    for v in raw_divergence:
        if not _is_zero(v):
            progress("  Bianchi: cancel() non-zero, trying simplify() (may be slow)...")
            v = simplify(v)
        divergence.append(v)

    all_zero = all(_is_zero(v) for v in divergence)
    progress(f"  Bianchi check: {'PASSED (all zero)' if all_zero else 'WARNING: non-zero after cancel+simplify'}")
    return divergence


def get_geodesic_equations(Gamma, coords, dim=4):
    """
    Construct the geodesic equations from the Christoffel symbols.

    The geodesic equation for an affinely parameterised curve x^μ(λ):
      d²x^μ/dλ² + Γ^μ_{αβ} (dx^α/dλ)(dx^β/dλ) = 0

    The acceleration term   Γ^μ_{αβ} u^α u^β   is returned as a SymPy
    expression in abstract velocity symbols u^{coord}.

    Parameters
    ----------
    Gamma  : 3D list  — Christoffel symbols
    coords : list     — coordinate symbols
    dim    : int

    Returns
    -------
    List of (μ, accel_expr, u_symbols) tuples.
    Each equation reads:  ü^μ = −accel_expr
    """
    progress("Building geodesic equations...")
    # Define abstract 4-velocity components u^{x^μ}
    u_syms = [symbols(f'dot_{{{coord_latex(c)}}}') for c in coords]

    eqs = []
    for mu in range(dim):
        accel = S.Zero
        for alpha in range(dim):
            for beta in range(dim):
                if Gamma[mu][alpha][beta] != S.Zero:
                    accel += Gamma[mu][alpha][beta] * u_syms[alpha] * u_syms[beta]
        eqs.append((mu, cancel(accel), u_syms))
    return eqs


def find_killing_coordinates(g_metric, coords, dim=4):
    """
    Identify cyclic coordinates — those the metric does not depend on.

    A coordinate x^μ is cyclic (ignorable) if
      ∂g_{αβ}/∂x^μ = 0   for all α, β.

    This corresponds to a Killing vector ξ^μ = δ^μ_μ̂ (coordinate basis
    vector), and the associated conserved quantity along a geodesic is:
      p_μ = g_{μν} dx^ν/dλ = const.

    Parameters
    ----------
    g_metric : sympy.Matrix  — covariant metric
    coords   : list          — coordinate symbols
    dim      : int

    Returns
    -------
    List of (index, coord_symbol) for each cyclic coordinate.
    """
    progress("Detecting cyclic coordinates (Killing vectors)...")
    cyclic = []
    for i, coord in enumerate(coords):
        is_cyclic = all(
            diff(g_metric[a, b], coord) == S.Zero
            for a in range(dim) for b in range(dim)
        )
        if is_cyclic:
            cyclic.append((i, coord))
    progress(f"  Cyclic coordinates: {[str(c) for _, c in cyclic]}")
    return cyclic


def compute_energy_conditions(G_ortho, dim=4):
    """
    Extract stress-energy components and symbolic energy-condition diagnostics
    from the orthonormal-frame Einstein tensor.

    From G_{ab} = 8*pi T_{ab} in the orthonormal frame:
      rho = Ghat_{00} / 8*pi
      p_i = Ghat_{ii} / 8*pi      (principal pressures)
      q_i = Ghat_{0i} / 8*pi      (energy-flux / momentum-density components)

    Important scope note
    --------------------
    The simple textbook expressions NEC = rho + p_i and DEC = rho - |p_i|
    are exact only when the orthonormal stress tensor is diagonal in the
    corresponding (0,i) block. When q_i != 0, the relevant null contractions
    are rho + p_i +/- 2 q_i, and the strongest directional null margin is
    rho + p_i - 2|q_i|.

    Parameters
    ----------
    G_ortho : sympy.Matrix  - orthonormal-frame Einstein tensor
    dim     : int

    Returns
    -------
    dict with keys:
        rho, pressures, fluxes,
        NEC, NEC_plus, NEC_minus, NEC_flux_margin,
        WEC_rho, SEC,
        DEC_pressures, DEC_flux,
        has_flux
    """
    rho = cancel(G_ortho[0, 0] / (8 * pi))
    pressures = [cancel(G_ortho[i, i] / (8 * pi)) for i in range(1, dim)]
    fluxes = [cancel(G_ortho[0, i] / (8 * pi)) for i in range(1, dim)]

    NEC = [cancel(rho + p) for p in pressures]
    NEC_plus = [cancel(rho + p + 2 * q) for p, q in zip(pressures, fluxes)]
    NEC_minus = [cancel(rho + p - 2 * q) for p, q in zip(pressures, fluxes)]
    NEC_flux_margin = [cancel(rho + p - 2 * Abs(q)) for p, q in zip(pressures, fluxes)]

    SEC = cancel(rho + sum(pressures))
    DEC_pressures = [cancel(rho - Abs(p)) for p in pressures]
    DEC_flux = [cancel(rho - Abs(q)) for q in fluxes]
    has_flux = any(not _is_zero(q) for q in fluxes)

    return {
        'rho': rho,
        'pressures': pressures,
        'fluxes': fluxes,
        'NEC': NEC,
        'NEC_plus': NEC_plus,
        'NEC_minus': NEC_minus,
        'NEC_flux_margin': NEC_flux_margin,
        'WEC_rho': rho,
        'SEC': SEC,
        'DEC_pressures': DEC_pressures,
        'DEC_flux': DEC_flux,
        'has_flux': has_flux,
    }

def _is_zero(expr):
    """Return True if cancel(expr) == 0 (robust to unsimplified symbolic zeros)."""
    try:
        return cancel(expr) == S.Zero
    except Exception:
        return False


# Automatic tetrad convention:
#   - diagonal metrics -> canonical coordinate-aligned static tetrad
#   - metrics with g_{0i} -> ADM/Eulerian tetrad adapted to the t = const slicing
#   - non-diagonal spatial blocks -> spatial triad fixed by Cholesky
def compute_tetrad_adm(g, ginv, coords, dim=4):
    """
    Automatically construct an orthonormal tetrad e^μ_a from the metric g.

    Three cases are detected:

    Case A — Fully diagonal metric (Schwarzschild, Minkowski, FRW, …):
        e^μ_a = diag(1/sqrt(-g_00), 1/sqrt(g_11), …, 1/sqrt(g_{n-1,n-1}))

    Case B — ADM shift metric, diagonal spatial block (Painlevé–Gullstrand type):
        Lapse N, shift β^i extracted; spatial triad E = diag(sqrt(γ_ii)).
        Coframe built analytically, then inverted to get e^μ_a.

    Case C — ADM shift metric, non-diagonal spatial block:
        Spatial triad via Cholesky factorisation of γ_ij.

    Parameters
    ----------
    g    : sympy.Matrix  (dim × dim) — covariant metric
    ginv : sympy.Matrix  (dim × dim) — contravariant metric (unused; kept for API)
    coords : list of symbols
    dim  : int (default 4)

    Returns
    -------
    e_contra : Matrix (dim × dim)  — e^μ_a  (what transform_to_ortho expects)
    e_cov    : Matrix (dim × dim)  — e^a_μ  (coframe; rows=frame, cols=coord)
    N        : scalar expression   — lapse function
    beta_up  : Matrix or None      — shift vector β^i as column (None for diagonal)
    E        : Matrix              — spatial triad E^â_i
    tetrad_method : str            — 'diagonal' | 'adm_shift_diagonal_spatial'
                                     | 'adm_shift_cholesky'
    """
    n = dim  # shorthand

    # ------------------------------------------------------------------
    # Detect whether ALL off-diagonal entries vanish
    # ------------------------------------------------------------------
    fully_diagonal = all(
        _is_zero(g[i, j]) for i in range(n) for j in range(n) if i != j
    )

    # ------------------------------------------------------------------
    # Case A: Fully diagonal metric
    # ------------------------------------------------------------------
    if fully_diagonal:
        # Canonical diagonal frame aligned with the coordinate basis.
        # For Schwarzschild coordinates this is the standard static frame.
        entries_contra = []
        entries_cov    = []
        E_diag         = []
        for i in range(n):
            gii = cancel(g[i, i])
            if i == 0:
                # time-like: g_00 < 0
                val_contra = 1 / sqrt(-gii)
                val_cov    = sqrt(-gii)
            else:
                val_contra = 1 / sqrt(gii)
                val_cov    = sqrt(gii)
                E_diag.append(val_cov)
            entries_contra.append(val_contra)
            entries_cov.append(val_cov)

        e_contra = Matrix(n, n, lambda i, j: entries_contra[i] if i == j else S.Zero)
        e_cov    = Matrix(n, n, lambda i, j: entries_cov[i]    if i == j else S.Zero)
        N        = sqrt(-cancel(g[0, 0]))
        beta_up  = None
        E        = Matrix(n - 1, n - 1, lambda i, j: E_diag[i] if i == j else S.Zero)
        return e_contra, e_cov, N, beta_up, E, 'diagonal'

    # ------------------------------------------------------------------
    # Cases B / C: ADM shift metric  g_{0i} ≠ 0
    # ------------------------------------------------------------------
    # Extract spatial metric γ_ij  (lower-right (n-1)×(n-1) block)
    gamma = Matrix(n - 1, n - 1, lambda i, j: cancel(g[i + 1, j + 1]))

    # Shift co-vector β_i = g_{0i}
    beta_down = Matrix(n - 1, 1, lambda i, _: cancel(g[0, i + 1]))

    # γ^{ij}
    gamma_inv = Matrix(n - 1, n - 1, lambda i, j: cancel(gamma.inv()[i, j]))

    # Shift vector β^i = γ^{ij} β_j
    beta_up = gamma_inv * beta_down          # column vector, (n-1)×1
    beta_up = Matrix(n - 1, 1, lambda i, _: cancel(beta_up[i, 0]))

    # β_i β^i
    beta_sq = cancel((beta_down.T * beta_up)[0, 0])

    # Lapse  N = sqrt(β_i β^i − g_{00})
    N = sqrt(cancel(beta_sq - g[0, 0]))

    # ------------------------------------------------------------------
    # Detect whether the spatial block is diagonal
    # ------------------------------------------------------------------
    spatial_diagonal = all(
        _is_zero(gamma[i, j]) for i in range(n - 1) for j in range(n - 1) if i != j
    )

    # ------------------------------------------------------------------
    # Case B: diagonal spatial block
    # ------------------------------------------------------------------
    if spatial_diagonal:
        # ADM/Eulerian frame with lapse N, shift beta^i, and a diagonal
        # spatial triad. This is the natural orthonormal frame associated
        # with the chosen foliation, not an arbitrary boosted tetrad.
        E_diag = [sqrt(cancel(gamma[i, i])) for i in range(n - 1)]
        E = Matrix(n - 1, n - 1, lambda i, j: E_diag[i] if i == j else S.Zero)

        # Coframe  e^a_μ
        #   Row 0 (time frame):  e^0_0 = N,  e^0_i = 0
        #   Row a (spatial, a=1..n-1):
        #       e^a_0 = E^{a-1}_i β^i  (summed over i)
        #       e^a_μ = E^{a-1}_{μ-1}   for μ ≥ 1
        e_cov = zeros(n, n)
        e_cov[0, 0] = N
        for a in range(1, n):
            # e^a_0 = sum_i E[a-1, i] * beta_up[i]  (E diagonal → only i=a-1 term)
            e_cov[a, 0] = cancel(E[a - 1, a - 1] * beta_up[a - 1])
            for mu in range(1, n):
                e_cov[a, mu] = E[a - 1, mu - 1]

        e_contra = Matrix(n, n, lambda i, j: cancel(e_cov.inv()[i, j]))
        return e_contra, e_cov, N, beta_up, E, 'adm_shift_diagonal_spatial'

    # ------------------------------------------------------------------
    # Case C: non-diagonal spatial block — Cholesky factorisation
    # ------------------------------------------------------------------
    try:
        L = gamma.cholesky()
    except Exception:
        # Try after simplification
        try:
            gamma_simp = Matrix(n - 1, n - 1, lambda i, j: simplify(gamma[i, j]))
            L = gamma_simp.cholesky()
        except Exception as exc:
            raise ValueError(
                f"compute_tetrad_adm: Cholesky failed for spatial metric — "
                f"supply e_tetrad manually. Error: {exc}"
            )

    # E^a_i = L[i, a]  (SymPy cholesky gives lower-triangular L s.t. γ = L L^T)
    # We need rows = frame index, cols = coord index
    E = Matrix(n - 1, n - 1, lambda a, i: cancel(L[i, a]))

    # Coframe  e^a_μ (same formula as Case B, but E is full matrix)
    e_cov = zeros(n, n)
    e_cov[0, 0] = N
    for a in range(1, n):
        e_cov[a, 0] = cancel(sum(E[a - 1, i] * beta_up[i] for i in range(n - 1)))
        for mu in range(1, n):
            e_cov[a, mu] = E[a - 1, mu - 1]

    e_contra = Matrix(n, n, lambda i, j: cancel(e_cov.inv()[i, j]))
    return e_contra, e_cov, N, beta_up, E, 'adm_shift_cholesky'


def verify_tetrad(e_contra, g, dim=4):
    """
    Verify that e^μ_a is a valid orthonormal tetrad by computing
        result[a, b] = Σ_{μν} e^μ_a · g_{μν} · e^ν_b
    and checking result == η_{ab} = diag(-1, +1, +1, +1).

    Parameters
    ----------
    e_contra : Matrix (dim × dim)  — entry [μ, a] = e^μ_a
    g        : Matrix (dim × dim)  — covariant metric
    dim      : int

    Returns
    -------
    passed   : bool    — True if result == η up to cancel()
    residual : Matrix  — result − η  (all zeros if passed)
    """
    if isinstance(e_contra, bool):
        raise TypeError(
            "e_contra must be a SymPy Matrix, not a boolean. "
            "Did you mean to set COMPUTE_TETRAD = True instead of e_tetrad = True?"
        )
    if isinstance(g, bool):
        raise TypeError(
            "g must be a metric Matrix, not a boolean. "
            "Check the arguments passed to verify_tetrad() or run_computations()."
        )
    n = dim
    eta = Matrix(n, n, lambda i, j: (-1 if i == 0 else 1) if i == j else 0)
    result = zeros(n, n)
    for a in range(n):
        for b in range(n):
            val = S.Zero
            for mu in range(n):
                for nu in range(n):
                    val += e_contra[mu, a] * g[mu, nu] * e_contra[nu, b]
            result[a, b] = cancel(val)
    residual = Matrix(n, n, lambda i, j: cancel(result[i, j] - eta[i, j]))
    passed = all(_is_zero(residual[i, j]) for i in range(n) for j in range(n))
    return passed, residual


# ==============================================================================
