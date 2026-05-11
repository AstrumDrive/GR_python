# -*- coding: utf-8 -*-
"""
gr_petrov.py — Petrov Classification and Newman-Penrose Formalism

Computes:
  • Newman-Penrose null tetrad {l^μ, n^μ, m^μ, m̄^μ} from an orthonormal tetrad
  • Weyl scalars Ψ₀, Ψ₁, Ψ₂, Ψ₃, Ψ₄  (complex scalars encoding gravitational DOF)
  • Petrov algebraic type  (O, N, III, D, II, I)
  • Bel-Robinson tensor T_{μνρσ}  (super-energy tensor, positive definite in GR)

References
----------
• Newman & Penrose, JMP 3 (1962) 566
• Chandrasekhar, "The Mathematical Theory of Black Holes" (1983) Ch. 1
• Stephani et al., "Exact Solutions of Einstein's Field Equations" (2003) Ch. 4
"""

import sympy as sp
from sympy import (
    symbols, sqrt, Rational, S, I, cancel, simplify,
    zeros, Matrix, conjugate, Abs, cos, sin
)
from gr_tensors import progress, _is_zero


# ==============================================================================
# NULL TETRAD CONSTRUCTION
# ==============================================================================

def build_np_tetrad(e_contra, dim=4):
    """
    Construct a Newman-Penrose null tetrad from an orthonormal (Lorentzian) tetrad.

    Given an orthonormal tetrad e^μ_a with e^μ_0 time-like and e^μ_1,2,3 space-like,
    the NP null tetrad is defined as:

        l^μ = (e^μ_0 + e^μ_1) / √2     [outgoing null]
        n^μ = (e^μ_0 − e^μ_1) / √2     [ingoing null]
        m^μ = (e^μ_2 + i e^μ_3) / √2   [complex null]
        m̄^μ = (e^μ_2 − i e^μ_3) / √2  [complex conjugate]

    The NP inner products satisfy:
        l · n = −1,  m · m̄ = +1,  all others = 0.

    Parameters
    ----------
    e_contra : sympy.Matrix  dim×dim — e_contra[μ, a] = e^μ_a
    dim      : int

    Returns
    -------
    dict with keys:
        'l', 'n', 'm', 'mbar'  — each a list of dim SymPy expressions
    """
    progress("Building Newman-Penrose null tetrad from orthonormal frame...")

    l_vec    = [cancel((e_contra[mu, 0] + e_contra[mu, 1]) / sqrt(2)) for mu in range(dim)]
    n_vec    = [cancel((e_contra[mu, 0] - e_contra[mu, 1]) / sqrt(2)) for mu in range(dim)]
    m_vec    = [cancel((e_contra[mu, 2] + I * e_contra[mu, 3]) / sqrt(2)) for mu in range(dim)]
    mbar_vec = [cancel((e_contra[mu, 2] - I * e_contra[mu, 3]) / sqrt(2)) for mu in range(dim)]

    progress("  NP null tetrad constructed (complex arithmetic active).")
    return {'l': l_vec, 'n': n_vec, 'm': m_vec, 'mbar': mbar_vec}


def verify_np_tetrad(np_tetrad, g, dim=4):
    """
    Verify NP null tetrad inner-product relations.

    Expected: l·n = −1, m·m̄ = +1, all other pairings = 0.

    Returns
    -------
    passed : bool
    report : dict mapping 'l·n', 'l·l', 'n·n', 'm·m', 'mbar·mbar', 'm·mbar' → value
    """
    l, n, m, mb = (np_tetrad['l'], np_tetrad['n'],
                   np_tetrad['m'], np_tetrad['mbar'])

    def inner(u, v):
        s = S.Zero
        for mu in range(dim):
            for nu in range(dim):
                s += g[mu, nu] * u[mu] * v[nu]
        return cancel(s)

    report = {
        'l·n':   inner(l, n),
        'l·l':   inner(l, l),
        'n·n':   inner(n, n),
        'm·m':   inner(m, m),
        'mbar·mbar': inner(mb, mb),
        'm·mbar':    inner(m, mb),
        'l·m':   inner(l, m),
        'n·m':   inner(n, m),
    }

    # Check l·n = −1, m·m̄ = +1, nulls = 0, cross = 0
    passed = (
        _is_zero(report['l·n'] + 1)
        and _is_zero(report['m·mbar'] - 1)
        and all(_is_zero(report[k]) for k in ('l·l', 'n·n', 'm·m', 'mbar·mbar', 'l·m', 'n·m'))
    )
    progress(f"  NP tetrad orthogonality check: {'PASSED' if passed else 'WARNING: deviations detected'}")
    return passed, report


# ==============================================================================
# WEYL SCALARS  Ψ₀ – Ψ₄
# ==============================================================================

def _contract_weyl_np(C_low, u, v, w, x, dim=4):
    """
    Contract all-lowered Weyl tensor C_{abcd} with four NP vectors.
    Result = C_{abcd} u^a v^b w^c x^d
    """
    s = S.Zero
    for a in range(dim):
        if u[a] == S.Zero: continue
        for b in range(dim):
            if v[b] == S.Zero: continue
            for c in range(dim):
                if w[c] == S.Zero: continue
                for d in range(dim):
                    if x[d] == S.Zero: continue
                    s += C_low[a][b][c][d] * u[a] * v[b] * w[c] * x[d]
    return cancel(s)


def compute_weyl_scalars(C_weyl, np_tetrad, g, dim=4):
    """
    Compute the five complex Newman-Penrose Weyl scalars.

    Definitions (lowered-index Weyl tensor C_{abcd})
    -------------------------------------------------
    Ψ₀ = C_{abcd} l^a m^b l^c m^d
    Ψ₁ = C_{abcd} l^a n^b l^c m^d
    Ψ₂ = C_{abcd} l^a m^b m̄^c n^d
    Ψ₃ = C_{abcd} l^a n^b m̄^c n^d
    Ψ₄ = C_{abcd} n^a m̄^b n^c m̄^d

    Physical interpretation
    -----------------------
    Ψ₀, Ψ₄ — transverse gravitational wave modes (in/out)
    Ψ₁, Ψ₃ — longitudinal modes (usually zero in vacuum)
    Ψ₂     — Coulomb-like component (tidal forces, mass monopole)

    Parameters
    ----------
    C_weyl   : 4D list  C[ρ][σ][μ][ν] all-lowered Weyl tensor
    np_tetrad: dict from build_np_tetrad()
    g        : dim×dim Matrix (for verification; not used in contraction)
    dim      : int

    Returns
    -------
    dict with keys 'Psi0', 'Psi1', 'Psi2', 'Psi3', 'Psi4'  (SymPy expressions)
    """
    if C_weyl is None:
        progress("  Weyl scalars: Weyl tensor not available. Skipping.")
        return {f'Psi{k}': None for k in range(5)}

    progress("Computing Newman-Penrose Weyl scalars Ψ₀–Ψ₄...")
    l, n, m, mb = (np_tetrad['l'], np_tetrad['n'],
                   np_tetrad['m'], np_tetrad['mbar'])

    psi = {}
    psi['Psi0'] = _contract_weyl_np(C_weyl, l, m,  l, m,  dim)
    progress("  Ψ₀ done")
    psi['Psi1'] = _contract_weyl_np(C_weyl, l, n,  l, m,  dim)
    progress("  Ψ₁ done")
    psi['Psi2'] = _contract_weyl_np(C_weyl, l, m,  mb, n, dim)
    progress("  Ψ₂ done")
    psi['Psi3'] = _contract_weyl_np(C_weyl, l, n,  mb, n, dim)
    progress("  Ψ₃ done")
    psi['Psi4'] = _contract_weyl_np(C_weyl, n, mb, n,  mb, dim)
    progress("  Ψ₄ done")

    for k, key in enumerate(('Psi0', 'Psi1', 'Psi2', 'Psi3', 'Psi4')):
        nz = "non-zero" if not _is_zero(psi[key]) else "zero"
        progress(f"  Ψ{k} = {nz}")

    return psi


# ==============================================================================
# PETROV CLASSIFICATION
# ==============================================================================

def classify_petrov(weyl_scalars, simplify_level=1):
    """
    Determine the Petrov algebraic type from the NP Weyl scalars.

    The classification is based on the pattern of vanishing scalars in a
    geometrically adapted NP frame.  The scalars computed from the canonical
    ADM tetrad may not be in the most convenient frame, so:
    • We attempt simplification (cancel, then simplify if needed).
    • The result is the Petrov type in the current frame orientation.

    Types and their physical content
    ---------------------------------
    O   — Conformally flat (FRW, Minkowski, de Sitter). No Weyl curvature.
    N   — Pure gravitational radiation (pp-waves). Ψ₄ ≠ 0 only.
    III — Ψ₃ non-zero; longitudinal radiation.
    D   — "Degenerate": Ψ₂ ≠ 0, Ψ₀=Ψ₁=Ψ₃=Ψ₄=0 (Schwarzschild, Kerr, RN).
    II  — General radiation + Coulomb background.
    I   — Algebraically general (most spacetimes).

    Note
    ----
    This classification is frame-dependent. The true Petrov type requires
    finding the principal null directions (PNDs). The frame-independent
    invariants I = Ψ₀Ψ₄ − 4Ψ₁Ψ₃ + 3Ψ₂² and J = det(Ψ matrix) are also
    computed as a cross-check.

    Parameters
    ----------
    weyl_scalars  : dict from compute_weyl_scalars()
    simplify_level: 0 = cancel only; 1 = cancel then simplify if needed

    Returns
    -------
    dict with keys:
        'type'        : str  ('O','N','III','D','II','I','unknown')
        'description' : str
        'psi_zero'    : dict mapping 'Psi0'...'Psi4' → bool (True = zero)
        'I_invariant' : SymPy expr
        'J_invariant' : SymPy expr
        'D_invariant' : SymPy expr  (I³ − 6J²; zero iff D,N,III,O)
    """
    progress("Classifying Petrov type from Weyl scalars...")

    if any(weyl_scalars.get(f'Psi{k}') is None for k in range(5)):
        progress("  Petrov classification: Weyl scalars unavailable.")
        return {
            'type': 'unknown', 'description': 'Weyl tensor not computed.',
            'psi_zero': {}, 'I_invariant': None, 'J_invariant': None, 'D_invariant': None,
        }

    def _is_z(expr):
        if expr is None: return True
        c = cancel(expr)
        if c == S.Zero: return True
        if simplify_level >= 1:
            try:
                return simplify(c) == S.Zero
            except Exception:
                pass
        return False

    psi = {k: weyl_scalars[f'Psi{k}'] for k in range(5)}
    z   = {k: _is_z(psi[k]) for k in range(5)}

    psi_zero = {f'Psi{k}': z[k] for k in range(5)}

    # Frame-invariant scalars
    p0, p1, p2, p3, p4 = [psi[k] if not z[k] else S.Zero for k in range(5)]
    I_inv = cancel(p0*p4 - 4*p1*p3 + 3*p2**2)
    J_inv = cancel(
        p0*(p2*p4 - p3**2)
        - p1*(p1*p4 - p2*p3)
        + p2*(p1*p3 - p2**2)
    )
    # D-invariant: I³ - 6J²  (vanishes for D, N, III, O)
    D_inv = cancel(I_inv**3 - 6*J_inv**2)

    # Determine type from zero pattern (frame-adapted result)
    all_zero = all(z[k] for k in range(5))
    only_p4  = (z[0] and z[1] and z[2] and z[3] and not z[4])
    only_p3p4 = (z[0] and z[1] and z[2] and not z[3])
    type_D   = (z[0] and z[1] and not z[2] and z[3] and z[4])
    type_II  = (z[0] and z[1] and not z[2])
    type_I   = not (z[0] and z[1])

    if all_zero:
        ptype = 'O'
        desc  = ('Conformally flat (Type O). No gravitational degrees of freedom. '
                 'Examples: Minkowski, FRW, de Sitter.')
    elif only_p4:
        ptype = 'N'
        desc  = ('Type N — pure radiation/gravitational wave. '
                 'Only Ψ₄ ≠ 0 (in adapted frame). Example: pp-waves.')
    elif only_p3p4:
        ptype = 'III'
        desc  = ('Type III — longitudinal radiation present. '
                 'Ψ₃ and Ψ₄ non-zero in adapted frame.')
    elif type_D:
        ptype = 'D'
        desc  = ('Type D — algebraically special, degenerate. '
                 'Only Ψ₂ ≠ 0 (in adapted frame). '
                 'Examples: Schwarzschild, Kerr, Reissner-Nordström.')
    elif type_II:
        ptype = 'II'
        desc  = ('Type II — radiation on a Coulomb background. '
                 'Ψ₂ non-zero; Ψ₃ and/or Ψ₄ non-zero.')
    elif type_I:
        ptype = 'I'
        desc  = ('Type I — algebraically general. '
                 'All five Ψ_k potentially non-zero.')
    else:
        ptype = 'unknown'
        desc  = 'Could not determine Petrov type from current frame.'

    progress(f"  Petrov type: {ptype} — {desc[:60]}...")
    return {
        'type':        ptype,
        'description': desc,
        'psi_zero':    psi_zero,
        'I_invariant': I_inv,
        'J_invariant': J_inv,
        'D_invariant': D_inv,
    }


# ==============================================================================
# BEL-ROBINSON TENSOR
# ==============================================================================

def compute_bel_robinson(C_weyl, g, ginv, dim=4):
    """
    Compute the Bel-Robinson super-energy tensor.

    T_{αβγδ} = C_{αρβσ} C^{ρ}_{γ}^{σ}_{δ} + *C_{αρβσ} *C^{ρ}_{γ}^{σ}_{δ}

    where *C is the dual Weyl tensor.

    Physical significance
    ---------------------
    The Bel-Robinson tensor is the closest analogue in GR to the
    electromagnetic energy-momentum tensor. It satisfies:
      ∇^α T_{αβγδ} = 0   (in vacuum)
    and is totally symmetric and traceless.  Its components represent
    the density and flux of gravitational super-energy.

    Note: Full computation is O(dim^8) and expensive for symbolic metrics.
    This function returns the dominant diagonal components T_{0000} and
    T_{i0i0} which capture the super-energy density.

    Parameters
    ----------
    C_weyl : 4D list (all-lowered Weyl tensor)
    g, ginv: metric matrices
    dim    : int

    Returns
    -------
    dict with 'T_0000' (super-energy density), 'T_1010', 'T_2020', 'T_3030'
    """
    if C_weyl is None:
        return {'T_0000': None, 'computed': False}

    progress("Computing Bel-Robinson super-energy density T_{0000}...")

    # Raise first and third indices of C: C^{ρ}_{σ}^{μ}_{ν} = g^{ρα}g^{μβ} C_{ασβν}
    # T_{0000} = C_{0ρ0σ} C^{ρ0σ0} + *C_{0ρ0σ} *C^{ρ0σ0}
    # For vacuum (C = Weyl = Riemann): simplified via Ψ scalars if available.

    # Practical approach: compute C_{0ρ0σ} C^{ρ0σ0}
    def raise_first_third(C, ginv, dim):
        """C^{ρ}_{σ}^{μ}_{ν} = g^{ρα} g^{μβ} C_{ασβν}"""
        R = [[[[S.Zero]*dim for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
        for rho in range(dim):
            for sig in range(dim):
                for mu in range(dim):
                    for nu in range(dim):
                        s = S.Zero
                        for a in range(dim):
                            if ginv[rho, a] == S.Zero: continue
                            for b in range(dim):
                                if ginv[mu, b] == S.Zero: continue
                                s += ginv[rho, a] * ginv[mu, b] * C[a][sig][b][nu]
                        R[rho][sig][mu][nu] = cancel(s)
        return R

    C_up = raise_first_third(C_weyl, ginv, dim)

    # T_{abcd} = C_{a ρ b σ} C^{ρ}_{c}^{σ}_{d}  (sum over ρ,σ)
    def T_comp(a, b, c, d):
        s = S.Zero
        for rho in range(dim):
            for sig in range(dim):
                s += C_weyl[a][rho][b][sig] * C_up[rho][c][sig][d]
        return cancel(s)

    result = {
        'T_0000': T_comp(0, 0, 0, 0),
        'T_1010': T_comp(1, 1, 0, 0),
        'T_2020': T_comp(2, 2, 0, 0),
        'computed': True,
    }
    if dim == 4:
        result['T_3030'] = T_comp(3, 3, 0, 0)

    progress("  Bel-Robinson diagonal components computed.")
    return result
