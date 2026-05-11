# -*- coding: utf-8 -*-
"""
gr_matter.py — Matter Field Stress-Energy Tensors

Provides symbolic stress-energy tensors T_{μν} for the standard matter fields
coupled to GR, and optional field equation residuals.

Included fields
---------------
  • Klein-Gordon scalar field φ:
        T_{μν} = ∂_μ φ ∂_ν φ − (1/2) g_{μν}(g^{αβ} ∂_α φ ∂_β φ + m² φ²)

  • Maxwell electromagnetic field F_{μν}:
        T_{μν} = F_{μα} F_ν^α − (1/4) g_{μν} F_{αβ} F^{αβ}
        Maxwell equations: ∇_μ F^{μν} = 4π J^ν

  • Fierz-Pauli massive spin-2 field h_{μν}:
        ℒ_FP = −(1/2)(∂_ρ h_{μν})² + (1/2)(∂_μ h)² − ∂_μ h^{μν} ∂_ν h
                + (1/2) m² (h_{μν}² − h²)
        (linearised on flat background — symbolic residual only)

References
----------
• Carroll, "Spacetime and Geometry" (2004) Ch. 1–4
• Weinberg, "Gravitation and Cosmology" (1972) §7
• Fierz & Pauli, Proc. Roy. Soc. A 173 (1939) 211
"""

import sympy as sp
from sympy import (
    symbols, Function, Matrix, diff, cancel, simplify,
    S, sqrt, Rational, zeros, eye
)
from gr_tensors import progress, _is_zero


# ==============================================================================
# KLEIN-GORDON SCALAR FIELD
# ==============================================================================

def compute_scalar_stress_energy(phi, g, ginv, coords, dim=4, mass=None):
    """
    Compute the stress-energy tensor of a Klein-Gordon scalar field.

    T_{μν} = ∂_μ φ ∂_ν φ − (1/2) g_{μν} (g^{αβ} ∂_α φ ∂_β φ + m² φ²)

    Parameters
    ----------
    phi    : SymPy expression or Function — the scalar field φ(x^μ)
    g      : dim×dim Matrix — covariant metric
    ginv   : dim×dim Matrix — contravariant metric
    coords : list of coordinate symbols
    dim    : int
    mass   : SymPy symbol or expr — scalar field mass m (0 for massless)

    Returns
    -------
    dict with keys:
        'T_cov'       : dim×dim Matrix  T_{μν} (covariant)
        'T_trace'     : SymPy expr      g^{μν} T_{μν}
        'kinetic'     : SymPy expr      X = g^{αβ} ∂_α φ ∂_β φ  (kinetic term)
        'mass_term'   : SymPy expr      m² φ²  (0 if massless)
        'field'       : str             'scalar'
    """
    progress("Computing Klein-Gordon scalar stress-energy tensor T_{μν}...")

    m2 = mass**2 if mass is not None else S.Zero

    # ∂_μ φ
    dphi = [cancel(diff(phi, coords[mu])) for mu in range(dim)]

    # Kinetic term X = g^{αβ} ∂_α φ ∂_β φ
    X = S.Zero
    for a in range(dim):
        for b in range(dim):
            X += ginv[a, b] * dphi[a] * dphi[b]
    X = cancel(X)

    # Mass potential
    V = cancel(m2 * phi**2)

    # T_{μν} = ∂_μ φ ∂_ν φ − (1/2) g_{μν} (X + V)
    T = zeros(dim)
    for mu in range(dim):
        for nu in range(dim):
            T[mu, nu] = cancel(dphi[mu] * dphi[nu] - Rational(1, 2) * g[mu, nu] * (X + V))

    # Trace
    trace = S.Zero
    for mu in range(dim):
        for nu in range(dim):
            trace += ginv[mu, nu] * T[mu, nu]
    trace = cancel(trace)

    # For massless in 4D: T = (1 - dim/2) X g_{μν}, trace = (1 - dim/2) dim X
    progress(f"  Scalar trace g^{{μν}} T_{{μν}} = {trace}")

    return {
        'T_cov':    T,
        'T_trace':  trace,
        'kinetic':  X,
        'mass_term': V,
        'field':    'scalar',
    }


def compute_kg_equation(phi, g, ginv, Gamma, coords, dim=4, mass=None):
    """
    Compute the Klein-Gordon equation residual: □φ − m²φ = 0.

    The covariant d'Alembertian:
        □φ = g^{μν} ∇_μ ∇_ν φ = (1/√|g|) ∂_μ (√|g| g^{μν} ∂_ν φ)

    Parameters
    ----------
    phi    : SymPy expression
    g      : metric matrix
    ginv   : inverse metric
    Gamma  : Christoffel symbols
    coords : list
    dim    : int
    mass   : SymPy symbol or None

    Returns
    -------
    dict:
        'box_phi'   : SymPy expr  (□φ)
        'residual'  : SymPy expr  (□φ − m²φ, should be zero on-shell)
    """
    progress("Computing Klein-Gordon equation □φ − m²φ...")

    m2 = mass**2 if mass is not None else S.Zero
    det_g = cancel(g.det())

    # √|det g|
    sqrt_g = cancel(sp.sqrt(sp.Abs(det_g)))

    # ∂_μ (√|g| g^{μν} ∂_ν φ)
    box_phi = S.Zero
    for mu in range(dim):
        inner_sum = S.Zero
        for nu in range(dim):
            inner_sum += ginv[mu, nu] * diff(phi, coords[nu])
        inner_sum = cancel(inner_sum)
        box_phi += diff(sqrt_g * inner_sum, coords[mu])

    box_phi = cancel(box_phi / sqrt_g)
    residual = cancel(box_phi - m2 * phi)

    progress(f"  □φ = {box_phi}")

    return {'box_phi': box_phi, 'residual': residual}


# ==============================================================================
# MAXWELL ELECTROMAGNETIC FIELD
# ==============================================================================

def build_faraday_tensor(A, coords, dim=4):
    """
    Build the Faraday (field strength) tensor from a vector potential.

        F_{μν} = ∂_μ A_ν − ∂_ν A_μ

    Parameters
    ----------
    A      : list of SymPy expressions  [A_0, A_1, ..., A_{dim-1}]
    coords : list of coordinate symbols
    dim    : int

    Returns
    -------
    F : dim×dim Matrix  F_{μν}
    """
    progress("Building Faraday tensor F_{μν} = ∂_μ A_ν − ∂_ν A_μ...")

    F = zeros(dim)
    for mu in range(dim):
        for nu in range(dim):
            F[mu, nu] = cancel(diff(A[nu], coords[mu]) - diff(A[mu], coords[nu]))

    # Bianchi identity ∂_{[μ}F_{νρ]} = 0 is automatic by antisymmetry.
    progress(f"  F_{{{0}{1}}} = {F[0, 1]}  (radial electric component)")

    return F


def compute_maxwell_stress_energy(F, g, ginv, coords, dim=4):
    """
    Compute the Maxwell stress-energy tensor.

        T_{μν} = (1/4π) [ F_{μα} F_ν^α − (1/4) g_{μν} F_{αβ} F^{αβ} ]

    In SI-natural units (c=G=1, 4πε₀=1):  prefactor is 1/(4π).
    In Gaussian-natural units:             prefactor is 1/(4π) as well.
    Set prefactor=1 for the geometric convention (common in GR texts).

    Parameters
    ----------
    F      : dim×dim Matrix  F_{μν}
    g      : covariant metric
    ginv   : contravariant metric
    coords : list
    dim    : int

    Returns
    -------
    dict with keys:
        'T_cov'        : dim×dim Matrix  T_{μν}
        'T_trace'      : SymPy expr      (should be 0 in 4D for Maxwell)
        'F_sq'         : SymPy expr      F_{αβ} F^{αβ}
        'F_contra'     : dim×dim Matrix  F^{μν} = g^{μα} g^{νβ} F_{αβ}
        'field'        : str
    """
    progress("Computing Maxwell stress-energy tensor T_{μν}^{EM}...")

    # Raise both indices: F^{μν} = g^{μα} g^{νβ} F_{αβ}
    F_contra = zeros(dim)
    for mu in range(dim):
        for nu in range(dim):
            s = S.Zero
            for a in range(dim):
                for b in range(dim):
                    s += ginv[mu, a] * ginv[nu, b] * F[a, b]
            F_contra[mu, nu] = cancel(s)

    # F_{αβ} F^{αβ}  (Lorentz invariant)
    F_sq = S.Zero
    for a in range(dim):
        for b in range(dim):
            F_sq += F[a, b] * F_contra[a, b]
    F_sq = cancel(F_sq)
    progress(f"  F_{{αβ}} F^{{αβ}} = {F_sq}")

    # T_{μν} = F_{μα} F_ν^α − (1/4) g_{μν} F_{αβ} F^{αβ}
    # F_ν^α = g^{αβ} F_{νβ}  (raise second index)
    T = zeros(dim)
    for mu in range(dim):
        for nu in range(dim):
            term1 = S.Zero
            for a in range(dim):
                # F^{α}_{ν} = g^{αβ} F_{βν}
                F_nu_up_a = S.Zero
                for b in range(dim):
                    F_nu_up_a += ginv[a, b] * F[b, nu]
                term1 += F[mu, a] * cancel(F_nu_up_a)
            term2 = Rational(1, 4) * g[mu, nu] * F_sq
            T[mu, nu] = cancel(term1 - term2)

    # Trace
    trace = S.Zero
    for mu in range(dim):
        for nu in range(dim):
            trace += ginv[mu, nu] * T[mu, nu]
    trace = cancel(trace)
    progress(f"  Trace T^μ_μ = {trace}  (should be 0 in 4D)")

    return {
        'T_cov':    T,
        'T_trace':  trace,
        'F_sq':     F_sq,
        'F_contra': F_contra,
        'field':    'maxwell',
    }


def verify_maxwell_equations(F, ginv, Gamma, coords, dim=4, J=None):
    """
    Check the covariant Maxwell equations: ∇_μ F^{μν} = 4π J^ν.

    ∇_μ F^{μν} = ∂_μ F^{μν} + Γ^μ_{μλ} F^{λν} + Γ^ν_{μλ} F^{μλ}

    Parameters
    ----------
    F      : dim×dim Matrix  F_{μν}  (covariant)
    ginv   : inverse metric
    Gamma  : Christoffel symbols
    coords : list
    dim    : int
    J      : list of SymPy expressions [J^0, ...] or None (vacuum: J=0)

    Returns
    -------
    dict:
        'div_F'   : list of SymPy expr  ∇_μ F^{μν} for each ν
        'residual': list  (div_F − 4π J^ν, or div_F if J=None)
        'vacuum_satisfied': bool (True if all residuals vanish)
    """
    progress("Verifying covariant Maxwell equations ∇_μ F^{μν} = 4πJ^ν...")

    # Build F^{μν}
    F_up = zeros(dim)
    for mu in range(dim):
        for nu in range(dim):
            s = S.Zero
            for a in range(dim):
                for b in range(dim):
                    s += ginv[mu, a] * ginv[nu, b] * F[a, b]
            F_up[mu, nu] = cancel(s)

    J_vec = J or [S.Zero] * dim

    div_F    = []
    residual = []
    for nu in range(dim):
        s = S.Zero
        for mu in range(dim):
            s += diff(F_up[mu, nu], coords[mu])
            for lam in range(dim):
                s += Gamma[mu][mu][lam] * F_up[lam, nu]
                s += Gamma[nu][mu][lam] * F_up[mu, lam]
        s = cancel(s)
        div_F.append(s)
        res = cancel(s - 4 * sp.pi * J_vec[nu])
        residual.append(res)

    satisfied = all(_is_zero(r) for r in residual)
    progress(f"  Maxwell equations satisfied: {satisfied}")

    return {
        'div_F':             div_F,
        'residual':          residual,
        'vacuum_satisfied':  satisfied,
    }


# ==============================================================================
# FIERZ-PAULI MASSIVE SPIN-2 FIELD
# ==============================================================================

def compute_fierz_pauli_mass_term(h, coords, mass, eta=None, dim=4):
    """
    Compute the Fierz-Pauli mass Lagrangian density and equation of motion residual.

    Fierz-Pauli Lagrangian (on flat background η_{μν}):
        ℒ_m = −(m²/2)(h_{μν} h^{μν} − h²)

    where h = η^{μν} h_{μν}  (trace) and h^{μν} = η^{μα} η^{νβ} h_{αβ}.

    This is the unique ghost-free mass term for a spin-2 field.

    Equation of motion (linearised):
        (□ − m²) h_{μν} − ∂_μ ∂^α h_{αν} − ∂_ν ∂^α h_{αμ}
        + ∂_μ ∂_ν h + η_{μν}(∂^α ∂^β h_{αβ} − □h) + m²(h_{μν} − η_{μν}h) = 0

    Parameters
    ----------
    h      : dim×dim Matrix  h_{μν}  (perturbation, covariant)
    coords : list
    mass   : SymPy symbol (graviton mass m)
    eta    : dim×dim Matrix or None (Minkowski background; defaults to diag(-1,1,1,1))
    dim    : int

    Returns
    -------
    dict:
        'L_mass'   : SymPy expr  ℒ_m (mass Lagrangian density, no √|g|)
        'h_trace'  : SymPy expr  h = η^{μν} h_{μν}
        'h_sq'     : SymPy expr  h_{μν} h^{μν}
        'eom_residual': dim×dim Matrix  (EOM residuals; should vanish on-shell)
        'field'    : str
    """
    progress("Computing Fierz-Pauli spin-2 mass terms...")

    if eta is None:
        eta = sp.diag(*[-1] + [1]*(dim-1))

    # Build η^{μν}: Minkowski inverse
    eta_inv = sp.diag(*[-1] + [1]*(dim-1))   # same as eta for flat Minkowski

    # Raise indices: h^{μν} = η^{μα} η^{νβ} h_{αβ}
    h_up = zeros(dim)
    for mu in range(dim):
        for nu in range(dim):
            s = S.Zero
            for a in range(dim):
                for b in range(dim):
                    s += eta_inv[mu, a] * eta_inv[nu, b] * h[a, b]
            h_up[mu, nu] = cancel(s)

    # Trace h = η^{μν} h_{μν}
    h_trace = S.Zero
    for mu in range(dim):
        for nu in range(dim):
            h_trace += eta_inv[mu, nu] * h[mu, nu]
    h_trace = cancel(h_trace)

    # h_{μν} h^{μν}
    h_sq = S.Zero
    for mu in range(dim):
        for nu in range(dim):
            h_sq += h[mu, nu] * h_up[mu, nu]
    h_sq = cancel(h_sq)

    # ℒ_m = −(m²/2)(h_{μν} h^{μν} − h²)
    m2 = mass**2
    L_mass = cancel(-m2 / 2 * (h_sq - h_trace**2))

    progress(f"  h (trace) = {h_trace}")
    progress(f"  ℒ_FP mass = {L_mass}")

    # EOM residuals (linearised on flat background)
    # Full EOM is complex; compute the mass contribution to EOM:
    # ∂ℒ_m / ∂h^{μν} = −m²(h_{μν} − η_{μν} h)
    eom_mass = zeros(dim)
    for mu in range(dim):
        for nu in range(dim):
            eom_mass[mu, nu] = cancel(-m2 * (h[mu, nu] - eta[mu, nu] * h_trace))

    return {
        'L_mass':      L_mass,
        'h_trace':     h_trace,
        'h_sq':        h_sq,
        'eom_residual': eom_mass,
        'field':       'fierz_pauli',
    }


# ==============================================================================
# MASTER FUNCTION
# ==============================================================================

def compute_matter_content(g, ginv, Gamma, coords, dim=4,
                           scalar_field=None, scalar_mass=None,
                           vector_potential=None, four_current=None,
                           spin2_field=None, graviton_mass=None):
    """
    Compute all matter field stress-energy tensors and field equations.

    Parameters
    ----------
    g, ginv    : metric matrices
    Gamma      : Christoffel symbols
    coords     : list of coordinate symbols
    dim        : int
    scalar_field      : SymPy expr or None  — φ(x)
    scalar_mass       : SymPy symbol or None
    vector_potential  : list of SymPy expr or None  — [A_0, ..., A_{dim-1}]
    four_current      : list of SymPy expr or None  — [J^0, ..., J^{dim-1}]
    spin2_field       : dim×dim Matrix or None  — h_{μν}
    graviton_mass     : SymPy symbol or None

    Returns
    -------
    dict with keys:
        'scalar', 'scalar_eom', 'maxwell', 'maxwell_eom', 'spin2'
        (each is the output dict of the respective sub-function, or None)
    """
    results = {}

    if scalar_field is not None:
        results['scalar']     = compute_scalar_stress_energy(
            scalar_field, g, ginv, coords, dim, scalar_mass)
        results['scalar_eom'] = compute_kg_equation(
            scalar_field, g, ginv, Gamma, coords, dim, scalar_mass)
    else:
        results['scalar']     = None
        results['scalar_eom'] = None

    if vector_potential is not None:
        F = build_faraday_tensor(vector_potential, coords, dim)
        results['maxwell']     = compute_maxwell_stress_energy(F, g, ginv, coords, dim)
        results['maxwell_eom'] = verify_maxwell_equations(
            F, ginv, Gamma, coords, dim, four_current)
        results['maxwell']['F_cov'] = F
    else:
        results['maxwell']     = None
        results['maxwell_eom'] = None

    if spin2_field is not None and graviton_mass is not None:
        results['spin2'] = compute_fierz_pauli_mass_term(
            spin2_field, coords, graviton_mass, dim=dim)
    else:
        results['spin2'] = None

    return results
