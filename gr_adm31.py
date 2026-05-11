# -*- coding: utf-8 -*-
"""
gr_adm31.py — 3+1 ADM Decomposition, Constraints, and ADM Mass

Implements the Arnowitt-Deser-Misner (ADM) 3+1 split of general relativity:

  • Extrinsic curvature K_{ij}  (from normal deformation of spatial slices)
  • 3-Riemann tensor and 3-Ricci scalar R^{(3)}  (intrinsic curvature of slice)
  • Hamiltonian constraint:  R^{(3)} − K_{ij}K^{ij} + K² = 16πρ
  • Momentum constraints:    ∇_j K^{ij} − ∇^i K = 8π J^i
  • ADM mass  M_ADM from the Regge-Teitelboim surface integral at spatial infinity
  • ADM angular momentum J_ADM  from the surface integral of the extrinsic curvature

The 3+1 splitting assumes a foliation of spacetime by space-like hypersurfaces Σ_t
with unit time-like normal n^μ, lapse N, and shift β^i:

    ds² = −N² dt² + γ_{ij}(dx^i + β^i dt)(dx^j + β^j dt)

References
----------
• Arnowitt, Deser, Misner, in "Gravitation: An Introduction" (2008), arXiv:gr-qc/0405109
• York, in "Sources of Gravitational Radiation", ed. Smarr (1979)
• Baumgarte & Shapiro, "Numerical Relativity" (2010) Ch. 2–3
• Regge & Teitelboim, Ann. Phys. 88 (1974) 286  [ADM mass formula]
"""

import sympy as sp
from sympy import (
    symbols, Function, Matrix, diff, cancel, simplify,
    S, sqrt, Rational, zeros, limit, oo, pi
)
from gr_tensors import progress, _is_zero


# ==============================================================================
# EXTRACT ADM VARIABLES FROM A 4D METRIC
# ==============================================================================

def extract_adm_variables(g, ginv, coords, dim=4):
    """
    Extract the lapse N, shift β^i, and spatial metric γ_{ij} from a 4D metric
    written in ADM form.

    For a metric with the block structure:
        g_{00} = −N² + γ_{ij} β^i β^j
        g_{0i} = γ_{ij} β^j  (= β_i)
        g_{ij} = γ_{ij}

    Parameters
    ----------
    g, ginv : 4D metric matrices
    coords  : list  [t, x¹, x², x³]
    dim     : int

    Returns
    -------
    dict with keys:
        'gamma_cov'   : (dim-1)×(dim-1) Matrix  γ_{ij}
        'gamma_inv'   : (dim-1)×(dim-1) Matrix  γ^{ij}
        'gamma_det'   : SymPy expr
        'N'           : SymPy expr  (lapse)
        'N_sq'        : SymPy expr  (N²)
        'beta_cov'    : list of SymPy expr  β_i = g_{0i}
        'beta_contra' : list of SymPy expr  β^i = γ^{ij} β_j
        'n_cov'       : list  n_μ (covariant unit normal)
        'n_contra'    : list  n^μ (contravariant unit normal)
    """
    progress("Extracting ADM lapse, shift, and spatial metric...")

    d = dim - 1  # spatial dimension

    # Spatial metric γ_{ij} = g_{ij}  (spatial block)
    gamma = zeros(d)
    for i in range(d):
        for j in range(d):
            gamma[i, j] = cancel(g[i + 1, j + 1])

    # Spatial inverse
    try:
        gamma_inv = gamma.inv()
        for i in range(d):
            for j in range(d):
                gamma_inv[i, j] = cancel(gamma_inv[i, j])
    except Exception:
        progress("  WARNING: spatial metric inversion failed.")
        gamma_inv = None

    gamma_det = cancel(gamma.det())

    # Shift (covariant): β_i = g_{0i}
    beta_cov = [cancel(g[0, i + 1]) for i in range(d)]

    # Shift (contravariant): β^i = γ^{ij} β_j
    if gamma_inv is not None:
        beta_contra = [
            cancel(sum(gamma_inv[i, j] * beta_cov[j] for j in range(d)))
            for i in range(d)
        ]
    else:
        beta_contra = [None] * d

    # Lapse: N² = −g^{00}^{-1} = −g_{00} + γ_{ij}β^iβ^j
    beta_sq = S.Zero
    if beta_contra[0] is not None:
        for i in range(d):
            for j in range(d):
                beta_sq += gamma[i, j] * beta_contra[i] * beta_contra[j]

    N_sq = cancel(-g[0, 0] + beta_sq)
    N    = cancel(sqrt(N_sq))

    # Covariant unit normal: n_μ = (−N, 0, 0, 0)
    n_cov = [-N] + [S.Zero] * d

    # Contravariant normal: n^μ = g^{μν} n_ν
    n_contra = [S.Zero] * dim
    for mu in range(dim):
        s = S.Zero
        for nu in range(dim):
            s += ginv[mu, nu] * n_cov[nu]
        n_contra[mu] = cancel(s)

    progress(f"  N = {N}")
    progress(f"  β_i = {beta_cov}")
    progress(f"  det(γ) = {gamma_det}")

    return {
        'gamma_cov':   gamma,
        'gamma_inv':   gamma_inv,
        'gamma_det':   gamma_det,
        'N':           N,
        'N_sq':        N_sq,
        'beta_cov':    beta_cov,
        'beta_contra': beta_contra,
        'n_cov':       n_cov,
        'n_contra':    n_contra,
    }


# ==============================================================================
# EXTRINSIC CURVATURE
# ==============================================================================

def compute_extrinsic_curvature(g, ginv, Gamma, coords, adm_vars, dim=4):
    """
    Compute the extrinsic curvature K_{ij} of the spatial slice.

    Definition via the Lie derivative of the spatial metric:
        K_{ij} = −(1/2) £_n γ_{ij}
               = −(1/(2N)) (∂_t γ_{ij} − ∇_i β_j − ∇_j β_i)

    For a static metric (∂_t γ_{ij} = 0):
        K_{ij} = −(1/(2N)) (−∇_i β_j − ∇_j β_i)
               = (1/(2N)) (∇_i β_j + ∇_j β_i)

    Parameters
    ----------
    g, ginv  : 4D metric
    Gamma    : 4D Christoffel symbols
    coords   : list  (first coord = t)
    adm_vars : dict from extract_adm_variables()
    dim      : int

    Returns
    -------
    dict:
        'K_cov'   : (dim-1)×(dim-1) Matrix  K_{ij}
        'K_trace' : SymPy expr  K = γ^{ij} K_{ij}
        'K_sq'    : SymPy expr  K_{ij} K^{ij}
    """
    progress("Computing extrinsic curvature K_{ij}...")

    d     = dim - 1
    N     = adm_vars['N']
    beta  = adm_vars['beta_cov']   # β_i = g_{0, i+1}
    gamma = adm_vars['gamma_cov']
    ginv3 = adm_vars['gamma_inv']
    t     = coords[0]

    # ∂_t γ_{ij}
    dt_gamma = zeros(d)
    for i in range(d):
        for j in range(d):
            dt_gamma[i, j] = cancel(diff(gamma[i, j], t))

    # ∇_i β_j (3D covariant derivative of β_j using 4D Γ restricted to spatial)
    # ∇_i β_j = ∂_i β_j − Γ^k_{ij} β_k  (3D indices shifted by 1 in 4D)
    def nabla_beta(i, j):
        """∂_i β_j − ³Γ^k_{ij} β_k, with 3D index i → 4D index i+1"""
        s = diff(beta[j], coords[i + 1])
        for k in range(d):
            s -= Gamma[k + 1][i + 1][j + 1] * beta[k]
        return cancel(s)

    # K_{ij} = (1/2N)(−∂_t γ_{ij} + ∇_i β_j + ∇_j β_i)
    K_cov = zeros(d)
    for i in range(d):
        for j in range(d):
            K_cov[i, j] = cancel(
                (-dt_gamma[i, j] + nabla_beta(i, j) + nabla_beta(j, i)) / (2 * N)
            )

    # Trace K = γ^{ij} K_{ij}
    K_trace = S.Zero
    if ginv3 is not None:
        for i in range(d):
            for j in range(d):
                K_trace += ginv3[i, j] * K_cov[i, j]
    K_trace = cancel(K_trace)

    # K_{ij} K^{ij} = γ^{ik} γ^{jl} K_{ij} K_{kl}
    K_sq = S.Zero
    if ginv3 is not None:
        for i in range(d):
            for j in range(d):
                # K^{ij} = γ^{ik} γ^{jl} K_{kl}
                K_up_ij = S.Zero
                for k in range(d):
                    for l in range(d):
                        K_up_ij += ginv3[i, k] * ginv3[j, l] * K_cov[k, l]
                K_sq += K_cov[i, j] * cancel(K_up_ij)
    K_sq = cancel(K_sq)

    progress(f"  K (trace) = {K_trace}")
    progress(f"  K_ij K^ij = {K_sq}")

    return {
        'K_cov':   K_cov,
        'K_trace': K_trace,
        'K_sq':    K_sq,
    }


# ==============================================================================
# 3D RICCI SCALAR
# ==============================================================================

def compute_3ricci_scalar(gamma, gamma_inv, coords, dim=4):
    """
    Compute the 3D Ricci scalar R^{(3)} of the spatial slice.

    Uses the standard Christoffel → Riemann → Ricci → scalar pipeline
    restricted to the (dim-1) spatial coordinates.

    Parameters
    ----------
    gamma     : (dim-1)×(dim-1) Matrix  γ_{ij}
    gamma_inv : (dim-1)×(dim-1) Matrix  γ^{ij}
    coords    : full 4D coordinate list; spatial coords = coords[1:]
    dim       : int  (4D)

    Returns
    -------
    dict:
        'Gamma3'  : 3D Christoffel symbols (3D list)
        'Ric3'    : (dim-1)×(dim-1) Matrix  R^{(3)}_{ij}
        'R3'      : SymPy expr  R^{(3)} (3D Ricci scalar)
    """
    progress("Computing 3D Ricci scalar R^{(3)}...")

    d       = dim - 1
    coords3 = coords[1:]   # spatial coordinates

    # 3D Christoffel symbols
    Gamma3 = [[[S.Zero]*d for _ in range(d)] for _ in range(d)]
    for lam in range(d):
        for mu in range(d):
            for nu in range(d):
                s = S.Zero
                for sig in range(d):
                    s += Rational(1, 2) * gamma_inv[lam, sig] * (
                        diff(gamma[nu, sig], coords3[mu])
                        + diff(gamma[mu, sig], coords3[nu])
                        - diff(gamma[mu, nu], coords3[sig])
                    )
                Gamma3[lam][mu][nu] = cancel(s)

    # 3D Riemann tensor R^λ_{ρμν}
    Riem3 = [[[[S.Zero]*d for _ in range(d)] for _ in range(d)] for _ in range(d)]
    for lam in range(d):
        for rho in range(d):
            for mu in range(d):
                for nu in range(d):
                    r = (diff(Gamma3[lam][nu][rho], coords3[mu])
                         - diff(Gamma3[lam][mu][rho], coords3[nu]))
                    for sig in range(d):
                        r += (Gamma3[lam][mu][sig] * Gamma3[sig][nu][rho]
                              - Gamma3[lam][nu][sig] * Gamma3[sig][mu][rho])
                    Riem3[lam][rho][mu][nu] = cancel(r)

    # 3D Ricci tensor R^{(3)}_{μν} = R^λ_{μλν}
    Ric3 = zeros(d)
    for mu in range(d):
        for nu in range(d):
            s = S.Zero
            for lam in range(d):
                s += Riem3[lam][mu][lam][nu]
            Ric3[mu, nu] = cancel(s)

    # 3D Ricci scalar R^{(3)} = γ^{μν} R^{(3)}_{μν}
    R3 = S.Zero
    for mu in range(d):
        for nu in range(d):
            R3 += gamma_inv[mu, nu] * Ric3[mu, nu]
    R3 = cancel(R3)

    progress(f"  R^{{(3)}} = {R3}")

    return {
        'Gamma3': Gamma3,
        'Ric3':   Ric3,
        'R3':     R3,
    }


# ==============================================================================
# HAMILTONIAN AND MOMENTUM CONSTRAINTS
# ==============================================================================

def compute_adm_constraints(R3, K_sq, K_trace, gamma_inv, Gamma3, K_cov, coords, dim=4,
                             rho_matter=None, J_matter=None):
    """
    Evaluate the ADM Hamiltonian and momentum constraints.

    Hamiltonian constraint:
        H ≡ R^{(3)} + K² − K_{ij} K^{ij} = 16πρ

    Momentum constraints:
        M^i ≡ ∇_j (K^{ij} − γ^{ij} K) = 8π J^i

    Parameters
    ----------
    R3         : 3D Ricci scalar
    K_sq       : K_{ij} K^{ij}
    K_trace    : K = γ^{ij} K_{ij}
    gamma_inv  : (dim-1)×(dim-1) Matrix
    Gamma3     : 3D Christoffel symbols
    K_cov      : (dim-1)×(dim-1) Matrix  K_{ij}
    coords     : full 4D coordinate list
    dim        : int
    rho_matter : SymPy expr or None  (matter energy density; 0 for vacuum)
    J_matter   : list or None  (matter momentum density)

    Returns
    -------
    dict:
        'H'              : SymPy expr  (Hamiltonian constraint LHS)
        'H_residual'     : SymPy expr  (H − 16πρ)
        'H_vacuum_ok'    : bool
        'M'              : list of SymPy expr  (momentum constraint LHS components)
        'M_residual'     : list  (M^i − 8πJ^i)
        'M_vacuum_ok'    : bool
    """
    progress("Evaluating ADM Hamiltonian and momentum constraints...")

    d       = dim - 1
    coords3 = coords[1:]

    rho = rho_matter if rho_matter is not None else S.Zero
    J   = J_matter   if J_matter   is not None else [S.Zero] * d

    # Hamiltonian constraint: H = R^{(3)} + K² − K_{ij}K^{ij}
    H = cancel(R3 + K_trace**2 - K_sq)
    H_res = cancel(H - 16 * pi * rho)
    H_ok  = _is_zero(H_res)
    progress(f"  H = {H}   (16πρ = {16 * pi * rho})")
    progress(f"  Hamiltonian constraint satisfied (vacuum): {H_ok}")

    # K^{ij} = γ^{ik} γ^{jl} K_{kl}
    K_up = zeros(d)
    for i in range(d):
        for j in range(d):
            s = S.Zero
            for k in range(d):
                for l in range(d):
                    s += gamma_inv[i, k] * gamma_inv[j, l] * K_cov[k, l]
            K_up[i, j] = cancel(s)

    # Momentum constraint: M^i = ∇_j(K^{ij} − γ^{ij} K)
    M_list = []
    M_res  = []
    for i in range(d):
        s = S.Zero
        for j in range(d):
            # T^{ij} = K^{ij} − γ^{ij} K
            T_ij = cancel(K_up[i, j] - gamma_inv[i, j] * K_trace)
            # ∂_j T^{ij}
            s += diff(T_ij, coords3[j])
            # Γ^i_{jk} T^{kj}  (first index connection)
            for k in range(d):
                s += Gamma3[i][j][k] * cancel(K_up[k, j] - gamma_inv[k, j] * K_trace)
            # Γ^j_{jk} T^{ik}  (trace connection)
            for k in range(d):
                s += Gamma3[j][j][k] * cancel(K_up[i, k] - gamma_inv[i, k] * K_trace)
        M_i = cancel(s)
        M_list.append(M_i)
        M_res.append(cancel(M_i - 8 * pi * J[i] if i < len(J) else M_i))

    M_ok = all(_is_zero(r) for r in M_res)
    progress(f"  Momentum constraints satisfied (vacuum): {M_ok}")

    return {
        'H':           H,
        'H_residual':  H_res,
        'H_vacuum_ok': H_ok,
        'M':           M_list,
        'M_residual':  M_res,
        'M_vacuum_ok': M_ok,
    }


# ==============================================================================
# ADM MASS
# ==============================================================================

def compute_adm_mass(g, ginv, coords, dim=4, radial_falloff_check=True):
    """
    Compute the ADM mass from the asymptotic falloff of the metric.

    Regge-Teitelboim formula (surface integral at r → ∞):
        M_ADM = (1/16π) ∮_{S²} (∂_j γ_{ij} − ∂_i γ_{jj}) dS^i

    For a metric that asymptotes to Schwarzschild, the expansion of g_{rr}:
        g_{rr} = 1 + 2M/r + O(r⁻²)
    gives M_ADM = M directly.

    We extract M_ADM by matching the leading 1/r coefficient in the deviation
    from flat space:
        h_{rr} = g_{rr} − 1  →  M_ADM = −(r²/2) * (1/r coefficient of h_{rr})
                                        = (1/2) * lim_{r→∞} (r · (g_{rr} − 1))

    For rotating metrics, also extract the angular momentum from g_{tφ}:
        J_ADM = −(1/2) lim_{r→∞} r² · g_{tφ} / sin²θ

    Parameters
    ----------
    g, ginv  : 4D metric
    coords   : [t, r, θ, φ]
    dim      : int
    radial_falloff_check : bool — print the 1/r expansion of each component

    Returns
    -------
    dict:
        'M_ADM'   : SymPy expr  (ADM mass)
        'J_ADM'   : SymPy expr  (ADM angular momentum, or None)
        'method'  : str
        'g_rr_at_inf': SymPy expr  (limit of g_{rr} as r→∞)
        'notes'   : str
    """
    progress("Computing ADM mass from asymptotic metric falloff...")

    r = coords[1]

    g_rr = cancel(g[1, 1])
    g_tt = cancel(g[0, 0])

    # Try to read off M from g_{rr} = 1 + 2M/r + ...
    try:
        g_rr_limit = cancel(limit(g_rr, r, oo))
        progress(f"  lim_{{r→∞}} g_{{rr}} = {g_rr_limit}")

        # Extract 1/r coefficient
        h_rr = cancel(g_rr - 1)
        M_coeff = cancel(limit(r * h_rr, r, oo))
        M_ADM = cancel(M_coeff / 2)
        progress(f"  M_ADM (from g_{{rr}}) = {M_ADM}")
        method = "g_rr_falloff"

        if _is_zero(M_ADM):
            # Fallback: try g_{tt} = −1 + 2M/r + ...
            h_tt = cancel(-g_tt - 1)
            M_coeff2 = cancel(limit(r * h_tt, r, oo))
            M_ADM = cancel(M_coeff2 / 2)
            progress(f"  M_ADM (from g_{{tt}}) = {M_ADM}")
            method = "g_tt_falloff"

    except Exception as exc:
        progress(f"  WARNING: ADM mass limit failed: {exc}")
        M_ADM  = None
        method = "failed"

    # ADM angular momentum from g_{tφ}
    J_ADM = None
    if dim == 4:
        g_tphi = cancel(g[0, 3])
        if not _is_zero(g_tphi):
            try:
                theta = coords[2]
                phi_coeff = cancel(limit(r**2 * g_tphi, r, oo))
                # g_{tφ} = −2J sin²θ / r + ...  → J = −lim(r² g_{tφ}) / (2 sin²θ)
                J_ADM = cancel(-phi_coeff / (2 * sp.sin(theta)**2))
                progress(f"  J_ADM (angular momentum) = {J_ADM}")
            except Exception as exc:
                progress(f"  WARNING: J_ADM extraction failed: {exc}")

    notes = ""
    if M_ADM is not None and not _is_zero(M_ADM):
        notes = (f"Identified M_ADM = {M_ADM} from asymptotic 1/r coefficient. "
                 f"This agrees with the parameter M in the metric if the metric "
                 f"is asymptotically Schwarzschild.")
    elif M_ADM is not None and _is_zero(M_ADM):
        notes = "ADM mass vanishes (flat or de Sitter asymptote)."
    else:
        notes = "Could not extract ADM mass symbolically."

    return {
        'M_ADM':        M_ADM,
        'J_ADM':        J_ADM,
        'method':       method,
        'g_rr_at_inf':  g_rr_limit if method != "failed" else None,
        'notes':        notes,
    }


# ==============================================================================
# MASTER FUNCTION
# ==============================================================================

def run_adm31(g, ginv, Gamma, coords, dim=4,
              rho_matter=None, J_matter=None,
              compute_expansion=True):
    """
    Run the full ADM 3+1 decomposition pipeline.

    Parameters
    ----------
    g, ginv     : 4D metric matrices
    Gamma       : 4D Christoffel symbols
    coords      : list [t, r, θ, φ]
    dim         : int
    rho_matter  : SymPy expr or None  (energy density for constraint check)
    J_matter    : list or None
    compute_expansion : bool — compute R^{(3)} and constraints

    Returns
    -------
    dict with keys:
        'adm_vars'    : extract_adm_variables() output
        'extrinsic'   : compute_extrinsic_curvature() output
        'ricci3'      : compute_3ricci_scalar() output
        'constraints' : compute_adm_constraints() output
        'adm_mass'    : compute_adm_mass() output
    """
    progress("=" * 60)
    progress("3+1 ADM DECOMPOSITION")
    progress("=" * 60)

    adm_vars   = extract_adm_variables(g, ginv, coords, dim)
    extrinsic  = compute_extrinsic_curvature(g, ginv, Gamma, coords, adm_vars, dim)
    ricci3     = compute_3ricci_scalar(adm_vars['gamma_cov'], adm_vars['gamma_inv'],
                                        coords, dim)
    constraints = compute_adm_constraints(
        ricci3['R3'], extrinsic['K_sq'], extrinsic['K_trace'],
        adm_vars['gamma_inv'], ricci3['Gamma3'], extrinsic['K_cov'],
        coords, dim, rho_matter, J_matter
    )
    adm_mass = compute_adm_mass(g, ginv, coords, dim)

    progress("=" * 60)
    progress("ADM 3+1 DECOMPOSITION COMPLETE")
    progress("=" * 60)

    return {
        'adm_vars':    adm_vars,
        'extrinsic':   extrinsic,
        'ricci3':      ricci3,
        'constraints': constraints,
        'adm_mass':    adm_mass,
    }
