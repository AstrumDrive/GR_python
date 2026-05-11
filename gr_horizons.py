# -*- coding: utf-8 -*-
"""
gr_horizons.py — Horizon Detection, Surface Gravity, and Causal Structure

Computes:
  • Event horizons   — zeros of g^{rr} = 1/g_{rr} (where g_{rr}^{-1} = 0, i.e., Δ = 0)
  • Static limit / Stationary limit (ergosphere boundary) — zeros of g_{tt} = 0
  • Ergosphere region  — region where g_{tt} > 0 (closed time-like curves possible)
  • Surface gravity κ  — from the Killing horizon definition
  • Null expansion θ   — expansion of outgoing/ingoing null normals (trapped surfaces)
  • Causal structure summary  — overall categorisation of the spacetime

References
----------
• Wald, "General Relativity" (1984) Ch. 12
• Misner, Thorne, Wheeler, "Gravitation" (1973) §32–33
• Bardeen, Carter, Hawking, CMP 31 (1973) 161  [surface gravity]
"""

import sympy as sp
from sympy import (
    solve, symbols, sqrt, cancel, simplify, diff, S, oo,
    Rational, limit, sign, Abs
)
from gr_tensors import progress, _is_zero


# ==============================================================================
# HELPER — null inner product
# ==============================================================================

def _inner(g, u, v, dim):
    """g_{μν} u^μ v^ν"""
    s = S.Zero
    for mu in range(dim):
        for nu in range(dim):
            if g[mu, nu] == S.Zero: continue
            s += g[mu, nu] * u[mu] * v[nu]
    return cancel(s)


# ==============================================================================
# HORIZON DETECTION
# ==============================================================================

def find_horizons(g, ginv, coords, dim=4):
    """
    Locate event horizons and static-limit surfaces symbolically.

    Strategy
    --------
    Event horizons (Killing horizons for static/stationary metrics):
      Solve  g^{rr}(r) = 0,  i.e.  1/g_{rr} = 0  ↔  g_{rr} → ∞.
      Equivalently find zeros of the lapse function Δ(r) = g_{rr}^{-1}.

    Static-limit surface (ergosphere outer boundary):
      Solve  g_{tt}(r, θ) = 0.
      Inside the ergosphere g_{tt} > 0 and ∂/∂t is space-like.

    Parameters
    ----------
    g, ginv : sympy.Matrix
    coords  : list of coordinate symbols  [t, r, θ, φ]
    dim     : int (4 for standard GR)

    Returns
    -------
    dict with keys:
        'r_coord'          : coordinate treated as radial (coords[1])
        'horizon_roots'    : list of SymPy expressions  (g^{rr}=0 solutions)
        'horizon_count'    : int
        'static_limit_roots': list of SymPy expressions (g_{tt}=0 in r)
        'ergosphere_exists': bool — True if static-limit ≠ horizon
        'g_tt'             : SymPy expr
        'g_rr_inv'         : SymPy expr  (Δ-like function)
        'description'      : str
    """
    progress("=" * 60)
    progress("HORIZON DETECTION")
    progress("=" * 60)

    r = coords[1]         # radial coordinate
    theta = coords[2] if dim > 2 else None

    g_tt = cancel(g[0, 0])
    g_rr = cancel(g[1, 1])

    # g^{rr}: from inverse metric
    g_rr_inv = cancel(ginv[1, 1])

    # Horizons: zeros of g^{rr}
    progress(f"  g^{{rr}} = {g_rr_inv}")
    progress("  Solving g^{rr} = 0 for r ...")
    try:
        # Multiply through to get a polynomial form
        h_roots = solve(g_rr_inv, r)
        h_roots = [cancel(rt) for rt in h_roots]
        h_roots = sorted(h_roots, key=lambda x: (x.is_real, x), reverse=True)
    except Exception as exc:
        progress(f"  WARNING: solve() failed for horizons: {exc}")
        h_roots = []

    progress(f"  Horizon roots: {h_roots}")

    # Static limit: zeros of g_{tt}
    progress(f"  g_{{tt}} = {g_tt}")
    progress("  Solving g_{tt} = 0 for r ...")
    try:
        # Substitute equatorial plane θ = π/2 if g_{tt} depends on θ
        g_tt_equatorial = g_tt
        if theta is not None and g_tt.has(theta):
            g_tt_equatorial = g_tt.subs(theta, sp.pi / 2)
        sl_roots = solve(g_tt_equatorial, r)
        sl_roots = [cancel(rt) for rt in sl_roots]
    except Exception as exc:
        progress(f"  WARNING: solve() failed for static limit: {exc}")
        sl_roots = []

    progress(f"  Static-limit roots (equatorial): {sl_roots}")

    # Ergosphere: exists when static-limit ≠ horizon
    ergo = (sl_roots != h_roots) and len(sl_roots) > 0 and len(h_roots) > 0

    # Description
    if len(h_roots) == 0:
        desc = "No horizon found (naked singularity or globally regular spacetime)."
    elif len(h_roots) == 1:
        desc = f"Single horizon at r = {h_roots[0]}."
    elif len(h_roots) == 2:
        desc = (f"Two horizons: outer r+ = {h_roots[0]}, inner r- = {h_roots[1]}. "
                f"Cauchy horizon present.")
    else:
        desc = f"{len(h_roots)} horizon(s) found."

    if ergo:
        desc += " Ergosphere detected: static-limit surface lies outside the event horizon."

    progress(f"  {desc}")

    return {
        'r_coord':             r,
        'horizon_roots':       h_roots,
        'horizon_count':       len(h_roots),
        'static_limit_roots':  sl_roots,
        'ergosphere_exists':   ergo,
        'g_tt':                g_tt,
        'g_rr_inv':            g_rr_inv,
        'description':         desc,
    }


# ==============================================================================
# SURFACE GRAVITY
# ==============================================================================

def compute_surface_gravity(g, ginv, Gamma, coords, horizon_roots, dim=4):
    """
    Compute the surface gravity κ at each horizon.

    Definition (Killing horizon with Killing vector ξ^μ = ∂/∂t)
    ------------------------------------------------------------
        κ² = −(1/2)(∇_μ χ_ν)(∇^μ χ^ν)|_{horizon}

    where χ_μ = g_{μt} is the covariant Killing 1-form. For a static metric
    this simplifies to:

        κ = (1/2) |d(g_{tt})/dr| / √(−g_{tt} g_{rr}^{-1})|_{r→r+}

    which agrees with the Bardeen-Carter-Hawking formula and reduces to κ = 1/(4M)
    for Schwarzschild.

    Parameters
    ----------
    g, ginv  : metric matrices
    Gamma    : Christoffel symbols (3D list)
    coords   : list
    horizon_roots : list of r-values (output of find_horizons)
    dim      : int

    Returns
    -------
    list of dicts, one per horizon:
        'r_horizon' : SymPy expr
        'kappa'     : SymPy expr  (surface gravity)
        'hawking_T' : SymPy expr  (Hawking temperature T_H = κ/(2π))
        'method'    : str
    """
    progress("Computing surface gravity κ at each horizon...")

    r = coords[1]
    results = []

    g_tt = cancel(g[0, 0])
    g_rr_inv = cancel(ginv[1, 1])

    # Derivative of g_{tt} w.r.t. r
    dg_tt_dr = cancel(diff(g_tt, r))

    for r_h in horizon_roots:
        progress(f"  Processing horizon r = {r_h} ...")
        try:
            # κ = (1/2)|∂_r g_{tt}| / √(|g_{tt}||g^{rr}|) evaluated at r=r_h
            # Near a horizon g_{tt} → 0 and g^{rr} → 0; use the ratio
            # κ = (1/2) lim_{r→r_h} [ (∂_r g_{tt}) / √(−g_{tt}/g^{rr}) ]
            # = (1/2) ∂_r(-g_{tt} g^{rr})^{1/2} evaluated via L'Hôpital-like

            # Simplest form for static metrics: κ = -(1/2)(∂_r g_tt) √(g^tt) at horizon
            # g^{tt} = 1/g_{tt} for diagonal, but near horizon we need care.
            # Use κ² = -(1/2) ∇_μ(ξ_ν) ∇^μ(ξ^ν) where ξ^μ = δ^μ_0
            # For static diagonal: κ = (1/2)|∂_r g_{tt}| / √(-g_{tt} · g_{rr})
            # evaluated at r_h via limit

            num = dg_tt_dr.subs(r, r_h)

            # Compute √(-g_{tt}/g_{rr}^{-1}) = √(-g_{tt} * g_{rr})
            g_rr = cancel(g[1, 1])
            denom_expr = cancel(-g_tt * g_rr)
            denom_val = denom_expr.subs(r, r_h)

            if _is_zero(denom_val):
                # Near-horizon limit: both numerator and denominator vanish
                # Use L'Hôpital or polynomial approach
                # For polynomial lapse f(r) with simple zero at r_h:
                # κ = (1/2)|f'(r_h)| · |g_{rr}(r_h)|^{-1/2}
                # Actually for Schwarzschild-like: f = 1-2M/r, g_rr=1/f
                # κ = f'(r_h)/2 = (2M/r_h²)/2 = M/r_h² = 1/(4M) for r_h=2M ✓
                f_prime = cancel(diff(-g_tt, r).subs(r, r_h))
                kappa = cancel(f_prime / 2)
                method = "lapse_derivative"
            else:
                denom_sqrt = sp.sqrt(denom_val)
                kappa = cancel(Abs(num) / (2 * denom_sqrt))
                method = "killing_norm_ratio"

            kappa = cancel(kappa)
            T_H = cancel(kappa / (2 * sp.pi))

            progress(f"    κ = {kappa}  (method: {method})")
            progress(f"    T_Hawking = κ/(2π) = {T_H}")

            results.append({
                'r_horizon':  r_h,
                'kappa':      kappa,
                'hawking_T':  T_H,
                'method':     method,
            })

        except Exception as exc:
            progress(f"  WARNING: surface gravity computation failed at r={r_h}: {exc}")
            results.append({
                'r_horizon': r_h,
                'kappa':     None,
                'hawking_T': None,
                'method':    'failed',
            })

    return results


# ==============================================================================
# NULL EXPANSION (TRAPPED SURFACES)
# ==============================================================================

def compute_null_expansion(g, ginv, Gamma, coords, dim=4):
    """
    Compute the expansion θ of the outgoing and ingoing null normal congruences.

    The expansion of a null geodesic congruence with tangent k^μ is:
        θ = ∇_μ k^μ = ∂_μ k^μ + Γ^μ_{μλ} k^λ

    For a spherically symmetric metric with outgoing null vector
        k^μ_out = (1/√2)(1/√|g_{tt}|, 1/√g_{rr}, 0, 0)
    and ingoing
        k^μ_in  = (1/√2)(1/√|g_{tt}|, −1/√g_{rr}, 0, 0)

    A trapped surface has θ_out < 0 (outgoing light converging).
    The apparent horizon is where θ_out = 0.

    Parameters
    ----------
    g, ginv  : metric matrices
    Gamma    : Christoffel symbols
    coords   : list
    dim      : int

    Returns
    -------
    dict with keys:
        'theta_out'      : SymPy expr  (outgoing null expansion)
        'theta_in'       : SymPy expr  (ingoing null expansion)
        'trapped_condition': str  (symbolic condition for trapped surfaces)
        'apparent_horizon' : list  (roots of theta_out = 0 in r)
    """
    progress("Computing null expansion (trapped surfaces)...")

    r = coords[1]
    g_tt = cancel(g[0, 0])   # negative for exterior
    g_rr = cancel(g[1, 1])

    # Null normals (outgoing/ingoing)
    # Only use t and r components for spherically symmetric metrics
    try:
        norm_t = cancel(1 / sp.sqrt(-g_tt))
        norm_r = cancel(1 / sp.sqrt(g_rr))
    except Exception:
        progress("  Could not construct null normals (metric may not be purely diagonal).")
        return {
            'theta_out': None, 'theta_in': None,
            'trapped_condition': 'undetermined', 'apparent_horizon': [],
        }

    k_out = [S.Zero] * dim
    k_in  = [S.Zero] * dim
    k_out[0] = norm_t / sp.sqrt(2)
    k_out[1] = norm_r / sp.sqrt(2)
    k_in[0]  =  norm_t / sp.sqrt(2)
    k_in[1]  = -norm_r / sp.sqrt(2)

    def divergence(k):
        """∇_μ k^μ = ∂_μ k^μ + Γ^μ_{μλ} k^λ"""
        div = S.Zero
        for mu in range(dim):
            div += diff(k[mu], coords[mu])
            for lam in range(dim):
                if k[lam] == S.Zero: continue
                div += Gamma[mu][mu][lam] * k[lam]
        return cancel(div)

    theta_out = divergence(k_out)
    theta_in  = divergence(k_in)

    progress(f"  θ_out = {theta_out}")
    progress(f"  θ_in  = {theta_in}")

    # Apparent horizon: θ_out = 0
    try:
        ah_roots = solve(theta_out, r)
        ah_roots = [cancel(rt) for rt in ah_roots]
    except Exception:
        ah_roots = []

    progress(f"  Apparent horizon (θ_out=0) roots: {ah_roots}")

    return {
        'theta_out':          theta_out,
        'theta_in':           theta_in,
        'trapped_condition':  f"θ_out < 0 inside r = {ah_roots[0]}" if ah_roots else "θ_out sign unknown",
        'apparent_horizon':   ah_roots,
    }


# ==============================================================================
# CAUSAL STRUCTURE SUMMARY
# ==============================================================================

def causal_structure_summary(g, ginv, coords, dim=4, horizon_info=None):
    """
    Produce a qualitative causal-structure summary of the spacetime.

    This function analyses the metric signature, horizon count, and ergosphere
    to assign the spacetime to a known causal class and describe the Penrose
    diagram type.

    Parameters
    ----------
    g, ginv       : metric matrices
    coords        : list
    dim           : int
    horizon_info  : dict from find_horizons() or None

    Returns
    -------
    dict with keys:
        'type'           : str  ('minkowski', 'schwarzschild', 'reissner_nordstrom',
                                 'kerr', 'de_sitter', 'anti_de_sitter', 'wormhole',
                                 'cosmological', 'other')
        'penrose_type'   : str  (description of Penrose diagram)
        'causal_regions' : list of str  (names of causal regions)
        'bifurcation'    : bool (True if bifurcate Killing horizon present)
        'globally_hyperbolic': bool or None
        'notes'          : str
    """
    progress("Analysing causal structure...")

    h = horizon_info or {}
    h_count = h.get('horizon_count', 0)
    ergo    = h.get('ergosphere_exists', False)
    g_tt    = h.get('g_tt', cancel(g[0, 0]))

    # Detect off-diagonal t-φ term (rotation)
    rotating = (dim == 4 and not _is_zero(cancel(g[0, 3])))

    # Detect cosmological constant sign from g_{tt} at large r
    r = coords[1]
    try:
        g_tt_large_r = cancel(limit(g_tt, r, oo))
    except Exception:
        g_tt_large_r = None

    # Classify
    if h_count == 0 and not rotating:
        if g_tt_large_r is not None and g_tt_large_r == -1:
            cs_type = 'minkowski'
            penrose  = 'Diamond (Minkowski): I^±, i^0, i^±'
            regions  = ['exterior']
            bif      = False
            gh       = True
            notes    = 'Flat spacetime. Globally hyperbolic.'
        elif g_tt_large_r is not None and g_tt_large_r == S.Zero:
            cs_type = 'de_sitter'
            penrose  = 'Square (de Sitter): cosmological horizons at top/bottom'
            regions  = ['static patch', 'cosmological region']
            bif      = True
            gh       = False
            notes    = 'de Sitter space. Not globally hyperbolic (no Cauchy surface).'
        else:
            cs_type = 'other'
            penrose  = 'Unknown Penrose diagram type'
            regions  = ['exterior']
            bif      = False
            gh       = None
            notes    = 'Could not classify causal structure from metric alone.'
    elif h_count == 1 and not rotating and not ergo:
        cs_type = 'schwarzschild'
        penrose  = 'Kruskal–Szekeres extension: four regions (I–IV) with bifurcation sphere'
        regions  = ['exterior (I)', 'future interior (II)',
                    'parallel exterior (III)', 'past interior (IV)']
        bif      = True
        gh       = True
        notes    = ('Schwarzschild-like: single event horizon. '
                    'Maximally extended via Kruskal coordinates.')
    elif h_count == 2 and not rotating:
        cs_type = 'reissner_nordstrom'
        penrose  = 'Infinite tower (RN-type): alternating outer/inner horizons, Cauchy horizon'
        regions  = ['exterior', 'between horizons', 'interior (beyond Cauchy horizon)']
        bif      = True
        gh       = False
        notes    = ('Reissner-Nordström-like: outer and inner (Cauchy) horizons. '
                    'Inner horizon is unstable (mass-inflation instability).')
    elif rotating:
        cs_type = 'kerr'
        penrose  = 'Kerr Carter diagram: outer/inner horizons + ergosphere region'
        regions  = ['exterior (I)', 'ergosphere', 'between horizons (II)', 'interior']
        bif      = True
        gh       = False
        notes    = ('Rotating black hole. Ergosphere exterior to event horizon. '
                    'Penrose process allows energy extraction from ergosphere.')
    else:
        cs_type = 'other'
        penrose  = 'Unknown Penrose diagram type'
        regions  = ['undetermined']
        bif      = False
        gh       = None
        notes    = f'{h_count} horizon(s) detected; further analysis required.'

    progress(f"  Causal type: {cs_type}")
    progress(f"  Penrose diagram: {penrose}")

    return {
        'type':               cs_type,
        'penrose_type':       penrose,
        'causal_regions':     regions,
        'bifurcation':        bif,
        'globally_hyperbolic': gh,
        'notes':              notes,
    }


# ==============================================================================
# MASTER FUNCTION
# ==============================================================================

def analyse_horizons(g, ginv, Gamma, coords, dim=4,
                     compute_expansion=True,
                     compute_surface_kappa=True):
    """
    Run the full horizon analysis pipeline.

    Parameters
    ----------
    g, ginv  : sympy.Matrix
    Gamma    : 3D Christoffel list
    coords   : list of coordinate symbols
    dim      : int
    compute_expansion     : bool — compute null expansion (θ_out, θ_in)
    compute_surface_kappa : bool — compute surface gravity at each horizon

    Returns
    -------
    dict with keys:
        'horizons'       : output of find_horizons()
        'surface_gravity': list from compute_surface_gravity() or []
        'expansion'      : dict from compute_null_expansion() or {}
        'causal'         : dict from causal_structure_summary()
    """
    h_info   = find_horizons(g, ginv, coords, dim)

    if compute_surface_kappa and h_info['horizon_roots']:
        sg = compute_surface_gravity(g, ginv, Gamma, coords,
                                     h_info['horizon_roots'], dim)
    else:
        sg = []

    if compute_expansion:
        try:
            exp_info = compute_null_expansion(g, ginv, Gamma, coords, dim)
        except Exception as exc:
            progress(f"  WARNING: null expansion computation failed: {exc}")
            exp_info = {}
    else:
        exp_info = {}

    causal = causal_structure_summary(g, ginv, coords, dim, h_info)

    return {
        'horizons':        h_info,
        'surface_gravity': sg,
        'expansion':       exp_info,
        'causal':          causal,
    }
