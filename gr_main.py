# -*- coding: utf-8 -*-
"""
gr_main.py — User Configuration, Computation Pipeline, and Entry Point

USAGE: Run  python gr_calculator.py  (or python gr_main.py) to generate the PDF.

CUSTOMISE: Edit Section 1 below to change the metric, flags, and output name.

Sections contained:
  0 — Imports
  1 — User Input  (*** EDIT THIS SECTION ***)
  5 — Computation Pipeline (run_computations)
  7 — PDF Compilation (write_and_compile_pdf)
  8 — Main Entry Point
"""

from gr_tensors import (
    compute_christoffel, compute_riemann, compute_ricci, compute_ricci_scalar,
    compute_einstein, compute_kretschmann, compute_weyl, transform_to_ortho,
    check_bianchi, get_geodesic_equations, find_killing_coordinates,
    compute_energy_conditions, compute_tetrad_adm, verify_tetrad,
    compute_covariant_derivative_1form, compute_covariant_derivative_vector,
    compute_lie_derivative_metric, solve_killing_equation,
    compute_conserved_quantities, compute_carter_constant,
    progress,                   # defined in gr_tensors (lowest-level module)
)
from gr_latex import assemble_report
from gr_metric_library import list_builtin_metric_keys, select_metric

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




# ==============================================================================
# SECTION 1 — USER INPUT
#              *** THIS IS THE ONLY SECTION YOU NEED TO MODIFY ***
# ==============================================================================

# ------------------------------------------------------------------------------
# 1.1  COORDINATE SYMBOLS
#      List them in the order you want rows/columns of the metric matrix.
#      Add assumptions (real=True, positive=True) for cleaner simplification.
# ------------------------------------------------------------------------------
t, r, theta, phi = symbols('t r theta phi', real=True)
coords = [t, r, theta, phi]   # <-- edit if using different coordinates
dim    = 4                     # spacetime dimension (usually 4)

# ------------------------------------------------------------------------------
# 1.2  METRIC PARAMETERS AND SYMBOLIC FUNCTIONS
#      Define only the extra symbols/functions you need for a custom metric.
# ------------------------------------------------------------------------------
M = symbols('M', positive=True)   # reusable parameter for built-in examples
Q = symbols('Q', real=True)       # charge-like parameter for built-in/custom metrics
Lambda = symbols('Lambda', real=True)

# Example custom dependencies (uncomment only if your custom metric needs them):
# alpha_r = Function('alpha')(r)
# a_t = Function('a', positive=True)(t)
# H_t = Function('H')(t)
# Phi_r = Function('Phi')(r)
# b_r = Function('b')(r)

# ------------------------------------------------------------------------------
# 1.3  METRIC SELECTION
#      Pick a built-in metric key, or use METRIC_KEY = 'custom'.
# ------------------------------------------------------------------------------
AVAILABLE_METRIC_KEYS = list_builtin_metric_keys()

# Built-in options:
#   schwarzschild
#   reissner_nordstrom
#   de_sitter_static
#   minkowski_spherical
#   frw_flat
#   static_spherical
#   morris_thorne_wormhole
#   pg_areal
#   pg_spatial_conformal
#   warp_doc_baseline
#   warp_doc_variant_a
#   warp_doc_variant_b
# Recommended first run: start with 'schwarzschild' to verify your environment.
METRIC_KEY = 'schwarzschild'

# One-off custom metric:
# 1. Set METRIC_KEY = 'custom'
# 2. Define any extra symbols/functions above
# 3. Fill CUSTOM_METRIC_CONFIG below
CUSTOM_METRIC_CONFIG = None

# CUSTOM_METRIC_CONFIG = {
#     'g_metric': Matrix([
#         [...],  # row for t
#         [...],  # row for r
#         [...],  # row for theta
#         [...],  # row for phi
#     ]),
#     'metric_name': 'My Custom Metric',
#     'metric_description': 'Short description of the line element',
#     'g_inv_metric': None,  # optional manual inverse
#     'e_tetrad': None,      # optional manual orthonormal tetrad
# }
#
# Example A: simple diagonal custom metric
# alpha_r = Function('alpha')(r)
# CUSTOM_METRIC_CONFIG = {
#     'g_metric': Matrix([
#         [-alpha_r, 0, 0, 0],
#         [0, 1 / alpha_r, 0, 0],
#         [0, 0, r**2, 0],
#         [0, 0, 0, r**2 * sin(theta)**2],
#     ]),
#     'metric_name': 'Diagonal toy metric',
#     'metric_description': 'Example custom diagonal ansatz with alpha(r)',
#     'g_inv_metric': None,
#     'e_tetrad': None,
# }
#
# Example B: metric with a dt dr cross-term
# beta_r = Function('beta')(r)
# CUSTOM_METRIC_CONFIG = {
#     'g_metric': Matrix([
#         [-(1 - beta_r**2), beta_r, 0, 0],
#         [beta_r, 1, 0, 0],
#         [0, 0, r**2, 0],
#         [0, 0, 0, r**2 * sin(theta)**2],
#     ]),
#     'metric_name': 'Cross-term toy metric',
#     'metric_description': 'Example ansatz with a radial dt dr mixing term beta(r)',
#     'g_inv_metric': None,
#     'e_tetrad': None,
# }

PARAMETER_CONTEXT = {'M': M, 'Q': Q, 'Lambda': Lambda}
metric_config = select_metric(
    METRIC_KEY,
    coords,
    parameter_context=PARAMETER_CONTEXT,
    custom_metric_config=CUSTOM_METRIC_CONFIG,
)

g_metric = metric_config['g_metric']
METRIC_NAME = metric_config['metric_name']
METRIC_DESCRIPTION = metric_config['metric_description']
g_inv_metric = metric_config['g_inv_metric']
e_tetrad = metric_config['e_tetrad']
SELECTED_METRIC_SUMMARY = f"{METRIC_KEY}: {METRIC_NAME}"

# ------------------------------------------------------------------------------
# 1.4  HOW TO ADD A NEW BUILT-IN METRIC
#      Open gr_metric_library.py and add one more entry inside
#      build_builtin_metric_library(). Each entry uses the same keys as
#      CUSTOM_METRIC_CONFIG above.
# ------------------------------------------------------------------------------
# 1.5  LATEX SUBSTITUTIONS FOR CLEAN OUTPUT
#      Map raw SymPy latex strings to prettier alternatives.
#      Applied as simple string replacements after sp.latex() is called.
# ------------------------------------------------------------------------------
latex_subs = {
    r'\theta': r'\theta',        # already correct — example entry
    # r'2 M': r'2M',            # remove space between coefficient and symbol
}

# ------------------------------------------------------------------------------
# 1.6  COMPUTATION FLAGS
# ------------------------------------------------------------------------------
COMPUTE_WEYL        = True   # Weyl conformal tensor (4D, expensive)
COMPUTE_KRETSCHMANN = True   # Kretschmann scalar (very expensive for complex metrics)
COMPUTE_GEODESICS   = True   # Geodesic equations
COMPUTE_KILLING     = True   # Detect cyclic coordinates / Killing vectors
COMPUTE_TETRAD      = True   # Auto-compute orthonormal tetrad via ADM decomposition.
                              # Set this flag to True if you want the code to build the tetrad automatically.
                              # Do NOT set e_tetrad = True; e_tetrad must be a Matrix or None.
                              # Gauge convention: diagonal metrics use the canonical
                              # coordinate-aligned static tetrad; metrics with shift
                              # terms use the ADM/Eulerian tetrad of the chosen slicing.
                              # Set False to skip Sections 8 & 9 entirely (old behaviour).
                              # A user-supplied e_tetrad above always takes priority.
FAST_MODE           = False  # True = skip Weyl and Kretschmann (recommended for first runs)

# --- NEW MODULES (all default to False to keep runtime fast) ---
COMPUTE_PETROV      = False  # Newman-Penrose Weyl scalars Ψ₀–Ψ₄ + Petrov type (needs COMPUTE_WEYL + COMPUTE_TETRAD)
COMPUTE_HORIZONS    = True   # Horizon detection, surface gravity, causal structure
COMPUTE_KILLING_FULL= False  # Full Killing equation solver (slow; complements COMPUTE_KILLING)
COMPUTE_CARTER      = False  # Carter constant / constants of motion (non-spherical metrics)
COMPUTE_MATTER      = False  # Matter field T_{μν}: set fields below in 1.7
COMPUTE_ADM31       = False  # Full 3+1 ADM decomposition + Hamiltonian/momentum constraints + ADM mass
COMPUTE_PENROSE     = True   # Penrose-Carter conformal diagrams (requires matplotlib)
COMPUTE_GEODESIC_NUM= False  # Numerical geodesic integration (requires scipy)

# ------------------------------------------------------------------------------
# 1.7  MATTER FIELD CONFIGURATION  (only used when COMPUTE_MATTER = True)
#      Set the fields you want to include.
# ------------------------------------------------------------------------------
# Uncomment and define the field(s) you want:

# Scalar field:
# phi_scalar = Function('phi')(r)      # e.g. a static radial scalar
# scalar_mass = symbols('m_s', positive=True)

# Vector potential A_μ (Maxwell):
# Q_charge = symbols('Q', real=True)
# A_maxwell = [Q_charge / r, 0, 0, 0]  # Coulomb potential

# Spin-2 perturbation h_{μν}:
# h_spin2 = zeros(dim)  # define non-zero components here
# m_graviton = symbols('m_g', positive=True)

MATTER_CONFIG = {
    'scalar_field':     None,   # set to a SymPy expression
    'scalar_mass':      None,
    'vector_potential': None,   # set to a list [A_0, A_1, A_2, A_3]
    'four_current':     None,
    'spin2_field':      None,
    'graviton_mass':    None,
}

# ------------------------------------------------------------------------------
# 1.8  NUMERICAL GEODESIC CONFIGURATION  (only used when COMPUTE_GEODESIC_NUM = True)
# ------------------------------------------------------------------------------
# Parameter values to substitute for numerical integration:
NUMERIC_PARAMS  = {'M': 1.0}      # e.g. {'M': 1.0, 'a': 0.5, 'Q': 0.3}

# Initial conditions [t, r, θ, φ] and [u_t, u_r, u_θ, u_φ]:
GEODESIC_X0     = [0.0, 10.0, sp.pi/2, 0.0]
GEODESIC_U0     = [-1.0, 0.0, 0.0, 0.1]   # timelike: norm ≈ −1

# Affine parameter range:
GEODESIC_LAMBDA = (0, 2000)

# Output path for geodesic plot (None = don't save):
GEODESIC_PLOT_PATH = None   # e.g. "geodesic.png"

# ------------------------------------------------------------------------------
# 1.9  OUTPUT / DISPLAY OPTIONS
# ------------------------------------------------------------------------------
OUTPUT_FILENAME = "gr_report"      # Base filename for .tex and .pdf (no extension)
AUTHOR_NAME     = "nebolivar and his AI partners :)"


# ==============================================================================


# ==============================================================================
# SECTION 5 — COMPUTATION PIPELINE
# ==============================================================================

def validate_tetrad_input(e_tetrad):
    """
    Validate the optional user-supplied tetrad input.

    The automatic path is controlled by COMPUTE_TETRAD. Passing a boolean as
    e_tetrad is a common first-time mistake and should fail immediately, before
    the symbolic pipeline spends time computing curvature tensors.
    """
    if isinstance(e_tetrad, bool):
        raise TypeError(
            "e_tetrad must be a SymPy Matrix or None, not a boolean. "
            "Did you mean COMPUTE_TETRAD = True?"
        )


def run_computations(g_metric, coords, dim,
                     g_inv_metric=None,
                     e_tetrad=None,
                     compute_weyl_flag=True,
                     compute_kretschmann_flag=True,
                     compute_geodesics_flag=True,
                     compute_killing_flag=True,
                     compute_tetrad_flag=True,
                     fast_mode=False,
                     compute_petrov_flag=False,
                     compute_horizons_flag=True,
                     compute_killing_full_flag=False,
                     compute_carter_flag=False,
                     compute_matter_flag=False,
                     compute_adm31_flag=False,
                     compute_penrose_flag=True,
                     compute_geodesic_num_flag=False,
                     matter_config=None,
                     numeric_params=None,
                     geodesic_x0=None,
                     geodesic_u0=None,
                     geodesic_lambda=(0, 1000),
                     geodesic_plot_path=None):
    """
    Master computation function — runs all GR calculations in sequence.

    Each phase is clearly announced in the console output. Results are
    collected in a dictionary and returned for use by the report assembler.

    Parameters
    ----------
    g_metric                : sympy.Matrix  — covariant metric
    coords                  : list          — coordinate symbols
    dim                     : int           — spacetime dimension
    g_inv_metric            : sympy.Matrix or None
    e_tetrad                : sympy.Matrix or None
    compute_weyl_flag       : bool
    compute_kretschmann_flag: bool
    compute_geodesics_flag  : bool
    compute_killing_flag    : bool
    fast_mode               : bool          — if True, skip Weyl and Kretschmann
    compute_petrov_flag     : bool
    compute_horizons_flag   : bool
    compute_killing_full_flag: bool
    compute_carter_flag     : bool
    compute_matter_flag     : bool
    compute_adm31_flag      : bool
    compute_penrose_flag    : bool
    compute_geodesic_num_flag: bool
    matter_config           : dict or None
    numeric_params          : dict or None
    geodesic_x0, geodesic_u0: lists for numeric geodesic ICs
    geodesic_lambda         : (start, end) affine parameter range
    geodesic_plot_path      : str or None

    Returns
    -------
    results : dict
    """
    validate_tetrad_input(e_tetrad)

    results = {}

    # ---- Phase 1: Metric ----
    progress("=" * 60)
    progress("PHASE 1: METRIC")
    progress("=" * 60)
    results['g'] = g_metric

    if g_inv_metric is None:
        progress("Computing inverse metric via .inv() ...")
        ginv = g_metric.inv()
        # Cancel each entry for cleanliness
        for i in range(dim):
            for j in range(dim):
                ginv[i, j] = cancel(ginv[i, j])
        results['ginv'] = ginv
    else:
        results['ginv'] = g_inv_metric
        progress("Using user-supplied inverse metric.")

    progress("Computing metric determinant...")
    results['det_g'] = cancel(g_metric.det())
    progress(f"  det(g) = {results['det_g']}")

    # ---- Phase 2: Connection ----
    progress("=" * 60)
    progress("PHASE 2: CHRISTOFFEL SYMBOLS (CONNECTION)")
    progress("=" * 60)
    results['Gamma'] = compute_christoffel(
        results['g'], results['ginv'], coords, dim
    )

    # ---- Phase 3: Curvature ----
    progress("=" * 60)
    progress("PHASE 3: CURVATURE TENSORS")
    progress("=" * 60)
    results['R_riem'] = compute_riemann(results['Gamma'], coords, dim)
    results['Ric']    = compute_ricci(results['R_riem'], dim)
    results['R_scalar'] = compute_ricci_scalar(results['Ric'], results['ginv'], dim)
    progress(f"  Ricci scalar R = {results['R_scalar']}")

    # ---- Phase 4: Einstein ----
    progress("=" * 60)
    progress("PHASE 4: EINSTEIN TENSOR")
    progress("=" * 60)
    results['G'] = compute_einstein(
        results['Ric'], results['g'], results['ginv'],
        results['R_scalar'], dim
    )

    # ---- Phase 5: Invariants ----
    progress("=" * 60)
    progress("PHASE 5: CURVATURE INVARIANTS")
    progress("=" * 60)

    if compute_kretschmann_flag and not fast_mode:
        results['K'] = compute_kretschmann(
            results['R_riem'], results['g'], results['ginv'], dim
        )
    else:
        results['K'] = None
        progress("  Kretschmann: SKIPPED (fast_mode or flag off)")

    if compute_weyl_flag and not fast_mode and dim == 4:
        results['Weyl'] = compute_weyl(
            results['R_riem'], results['Ric'],
            results['g'], results['ginv'],
            results['R_scalar'], dim
        )
    else:
        results['Weyl'] = None
        progress("  Weyl tensor: SKIPPED (fast_mode, flag off, or dim ≠ 4)")

    # ---- Phase 6: Orthonormal Frame ----
    progress("=" * 60)
    progress("PHASE 6: ORTHONORMAL FRAME & ENERGY CONDITIONS")
    progress("=" * 60)

    # Initialise all tetrad result fields
    results['e_tetrad_computed'] = None
    results['e_cov']             = None
    results['tetrad_method']     = None
    results['tetrad_N']          = None
    results['tetrad_beta']       = None
    results['tetrad_E']          = None
    results['tetrad_verified']   = None
    results['tetrad_residual']   = None

    active_tetrad = e_tetrad   # user-supplied takes priority

    if active_tetrad is not None:
        # User-supplied tetrad: store and verify
        progress("  User-supplied tetrad detected — verifying orthonormality …")
        results['e_tetrad_computed'] = active_tetrad
        results['tetrad_method']     = 'user_supplied'
        passed, residual = verify_tetrad(active_tetrad, results['g'], dim)
        results['tetrad_verified'] = passed
        results['tetrad_residual'] = residual
        progress(f"  Tetrad verification: {'PASSED ✓' if passed else 'FAILED ✗'}")

    elif compute_tetrad_flag:
        # Auto-compute tetrad via ADM decomposition
        progress("  Auto-computing orthonormal tetrad (ADM decomposition) …")
        try:
            e_contra, e_cov, N, beta_up, E, method = compute_tetrad_adm(
                results['g'], results['ginv'], coords, dim
            )
            active_tetrad                = e_contra
            results['e_tetrad_computed'] = e_contra
            results['e_cov']             = e_cov
            results['tetrad_method']     = method
            results['tetrad_N']          = N
            results['tetrad_beta']       = beta_up
            results['tetrad_E']          = E
            progress(f"  Tetrad method: {method}")
            progress("  Verifying orthonormality g_{{μν}} e^μ_a e^ν_b = η_ab …")
            passed, residual = verify_tetrad(e_contra, results['g'], dim)
            results['tetrad_verified'] = passed
            results['tetrad_residual'] = residual
            progress(f"  Tetrad verification: {'PASSED ✓' if passed else 'FAILED ✗'}")
        except Exception as exc:
            progress(f"  WARNING: automatic tetrad construction failed: {exc}")
            progress("  Orthonormal-frame projection and energy-condition diagnostics will be skipped.")
            progress("  If you need that section, supply e_tetrad manually in Section 1.")
            active_tetrad = None

    else:
        progress("  COMPUTE_TETRAD = False — skipping orthonormal frame.")

    if active_tetrad is not None:
        results['G_ortho']           = transform_to_ortho(results['G'], active_tetrad, dim)
        results['energy_conditions'] = compute_energy_conditions(results['G_ortho'], dim)
    else:
        results['G_ortho']           = None
        results['energy_conditions'] = None

    # ---- Phase 7: Geodesics & Symmetries ----
    progress("=" * 60)
    progress("PHASE 7: GEODESICS & KILLING VECTORS")
    progress("=" * 60)
    if compute_geodesics_flag:
        results['geodesics'] = get_geodesic_equations(
            results['Gamma'], coords, dim
        )
    else:
        results['geodesics'] = None

    if compute_killing_flag:
        results['killing'] = find_killing_coordinates(
            results['g'], coords, dim
        )
    else:
        results['killing'] = []

    # ---- Phase 8: Conservation Checks ----
    progress("=" * 60)
    progress("PHASE 8: CONSERVATION CHECKS")
    progress("=" * 60)
    results['bianchi'] = check_bianchi(
        results['G'], results['ginv'], results['Gamma'], coords, dim
    )

    # Trace check: g^{μν} R_{μν} − R should be zero
    trace_residual = cancel(
        sum(results['ginv'][mu, nu] * results['Ric'][mu, nu]
            for mu in range(dim) for nu in range(dim))
        - results['R_scalar']
    )
    results['trace_check'] = trace_residual

    # ---- Phase 9: Newman-Penrose / Petrov ----
    if compute_petrov_flag and not fast_mode and dim == 4:
        progress("=" * 60)
        progress("PHASE 9: NEWMAN-PENROSE / PETROV CLASSIFICATION")
        progress("=" * 60)
        if results['Weyl'] is None:
            progress("  Petrov: Weyl tensor not available. Set COMPUTE_WEYL=True.")
            results['petrov'] = None
        elif active_tetrad is None:
            progress("  Petrov: orthonormal tetrad not available. Set COMPUTE_TETRAD=True.")
            results['petrov'] = None
        else:
            try:
                from gr_petrov import build_np_tetrad, verify_np_tetrad, compute_weyl_scalars, classify_petrov, compute_bel_robinson
                np_tetrad = build_np_tetrad(active_tetrad, dim)
                verify_np_tetrad(np_tetrad, results['g'], dim)
                weyl_scalars = compute_weyl_scalars(results['Weyl'], np_tetrad, results['g'], dim)
                petrov_class = classify_petrov(weyl_scalars)
                bel_rob = compute_bel_robinson(results['Weyl'], results['g'], results['ginv'], dim)
                results['petrov'] = {
                    'np_tetrad':    np_tetrad,
                    'weyl_scalars': weyl_scalars,
                    'classification': petrov_class,
                    'bel_robinson': bel_rob,
                }
            except Exception as exc:
                progress(f"  WARNING: Petrov computation failed: {exc}")
                results['petrov'] = None
    else:
        results['petrov'] = None
        if compute_petrov_flag:
            progress("  Petrov: SKIPPED (fast_mode or dim≠4 or Weyl disabled)")

    # ---- Phase 10: Horizons & Causal Structure ----
    if compute_horizons_flag and dim == 4:
        progress("=" * 60)
        progress("PHASE 10: HORIZONS & CAUSAL STRUCTURE")
        progress("=" * 60)
        try:
            from gr_horizons import analyse_horizons
            results['horizons'] = analyse_horizons(
                results['g'], results['ginv'], results['Gamma'], coords, dim
            )
        except Exception as exc:
            progress(f"  WARNING: Horizon analysis failed: {exc}")
            results['horizons'] = None
    else:
        results['horizons'] = None

    # ---- Phase 11: Full Killing Equation Solver ----
    if compute_killing_full_flag:
        progress("=" * 60)
        progress("PHASE 11: FULL KILLING EQUATION SOLVER")
        progress("=" * 60)
        try:
            from gr_tensors import solve_killing_equation, compute_conserved_quantities, compute_carter_constant
            results['killing_full'] = solve_killing_equation(
                results['g'], results['Gamma'], coords, dim
            )
            results['conserved_full'] = compute_conserved_quantities(
                results['g'], results['ginv'], coords, results['killing_full'], dim
            )
        except Exception as exc:
            progress(f"  WARNING: Killing equation solver failed: {exc}")
            results['killing_full'] = None
            results['conserved_full'] = None
    else:
        results['killing_full']   = None
        results['conserved_full'] = None

    # ---- Phase 12: Carter Constant ----
    if compute_carter_flag:
        progress("=" * 60)
        progress("PHASE 12: CARTER CONSTANT")
        progress("=" * 60)
        try:
            from gr_tensors import compute_carter_constant
            results['carter'] = compute_carter_constant(
                results['g'], results['ginv'], coords, dim
            )
        except Exception as exc:
            progress(f"  WARNING: Carter constant computation failed: {exc}")
            results['carter'] = None
    else:
        results['carter'] = None

    # ---- Phase 13: Matter Fields ----
    if compute_matter_flag and matter_config is not None:
        progress("=" * 60)
        progress("PHASE 13: MATTER FIELD STRESS-ENERGY TENSORS")
        progress("=" * 60)
        try:
            from gr_matter import compute_matter_content
            results['matter'] = compute_matter_content(
                results['g'], results['ginv'], results['Gamma'], coords, dim,
                scalar_field      = matter_config.get('scalar_field'),
                scalar_mass       = matter_config.get('scalar_mass'),
                vector_potential  = matter_config.get('vector_potential'),
                four_current      = matter_config.get('four_current'),
                spin2_field       = matter_config.get('spin2_field'),
                graviton_mass     = matter_config.get('graviton_mass'),
            )
        except Exception as exc:
            progress(f"  WARNING: Matter computation failed: {exc}")
            results['matter'] = None
    else:
        results['matter'] = None

    # ---- Phase 14: ADM 3+1 Decomposition ----
    if compute_adm31_flag and dim == 4:
        progress("=" * 60)
        progress("PHASE 14: ADM 3+1 DECOMPOSITION")
        progress("=" * 60)
        try:
            from gr_adm31 import run_adm31
            results['adm31'] = run_adm31(
                results['g'], results['ginv'], results['Gamma'], coords, dim
            )
        except Exception as exc:
            progress(f"  WARNING: ADM 3+1 computation failed: {exc}")
            results['adm31'] = None
    else:
        results['adm31'] = None

    # ---- Phase 15: Penrose Diagrams ----
    if compute_penrose_flag:
        progress("=" * 60)
        progress("PHASE 15: PENROSE-CARTER DIAGRAMS")
        progress("=" * 60)
        try:
            from gr_penrose import draw_penrose_diagram, list_penrose_spacetimes
            # Try to match METRIC_KEY to a known Penrose spacetime
            penrose_key = None
            known = list_penrose_spacetimes()
            if METRIC_KEY in known:
                penrose_key = METRIC_KEY
            elif 'schwarzschild' in METRIC_KEY:
                penrose_key = 'schwarzschild'
            elif 'kerr' in METRIC_KEY:
                penrose_key = 'kerr'
            elif 'reissner' in METRIC_KEY:
                penrose_key = 'reissner_nordstrom'
            elif 'de_sitter' in METRIC_KEY or 'ads' in METRIC_KEY:
                penrose_key = 'de_sitter' if 'anti' not in METRIC_KEY else 'anti_de_sitter'
            elif 'minkowski' in METRIC_KEY:
                penrose_key = 'minkowski'

            if penrose_key:
                results['penrose_fig'] = draw_penrose_diagram(penrose_key)
                results['penrose_key'] = penrose_key
                progress(f"  Penrose diagram generated for: {penrose_key}")
            else:
                progress("  No matching Penrose diagram for this metric.")
                results['penrose_fig'] = None
                results['penrose_key'] = None
        except Exception as exc:
            progress(f"  WARNING: Penrose diagram generation failed: {exc}")
            results['penrose_fig'] = None
            results['penrose_key'] = None
    else:
        results['penrose_fig'] = None
        results['penrose_key'] = None

    # ---- Phase 16: Numerical Geodesic ----
    if compute_geodesic_num_flag and geodesic_x0 is not None and geodesic_u0 is not None:
        progress("=" * 60)
        progress("PHASE 16: NUMERICAL GEODESIC INTEGRATION")
        progress("=" * 60)
        try:
            from gr_geodesic_numeric import (
                lambdify_christoffel, lambdify_metric,
                integrate_geodesic, check_conserved, plot_geodesic
            )
            ps = numeric_params or {}
            Gamma_num, _ = lambdify_christoffel(results['Gamma'], coords, dim, ps)
            g_num, ginv_num = lambdify_metric(results['g'], results['ginv'], coords, dim, ps)

            x0_num = [float(xi) if not hasattr(xi, 'subs') else float(xi.subs(ps))
                      for xi in geodesic_x0]
            u0_num = [float(ui) if not hasattr(ui, 'subs') else float(ui.subs(ps))
                      for ui in geodesic_u0]

            sol = integrate_geodesic(Gamma_num, x0_num, u0_num, geodesic_lambda, dim)
            conserved = check_conserved(sol, g_num, ginv_num,
                                        killing_indices=[0, 3] if dim == 4 else [0], dim=dim)
            if geodesic_plot_path or True:
                fig = plot_geodesic(conserved, coords,
                                    output_path=geodesic_plot_path)
            results['geodesic_num'] = {
                'sol': sol, 'conserved': conserved,
                'norm_drift': conserved['norm_drift'],
            }
        except Exception as exc:
            progress(f"  WARNING: Numerical geodesic failed: {exc}")
            results['geodesic_num'] = None
    else:
        results['geodesic_num'] = None

    progress("=" * 60)
    progress("ALL COMPUTATIONS COMPLETE")
    progress("=" * 60)
    return results


# ==============================================================================


# ==============================================================================
# SECTION 7 — PDF COMPILATION
# ==============================================================================

def write_and_compile_pdf(report_lines, output_base):
    """
    Write the LaTeX source file and attempt to compile it to PDF.

    Strategy
    --------
    1. Write all report_lines to  <output_base>.tex  (UTF-8 encoding).
    2. Detect whether pdflatex is available in the system PATH.
    3. If found: run pdflatex twice (second pass generates correct ToC
       and cross-references) and report the result.
    4. If not found: print manual compilation instructions.

    Parameters
    ----------
    report_lines : list of str — complete LaTeX document lines
    output_base  : str         — base filename without extension
    """
    # Resolve absolute paths so output always lands next to the script,
    # regardless of the working directory from which the script is invoked.
    tex_file   = os.path.abspath(output_base + '.tex')
    pdf_file   = os.path.abspath(output_base + '.pdf')
    script_dir = os.path.dirname(tex_file)   # same folder as the .tex file

    # Write .tex
    progress(f"Writing LaTeX source: {tex_file}")
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    progress(f"  Written ({len(report_lines)} lines).")

    # Check for pdflatex.
    # -output-directory ensures the .pdf, .aux, .log and .toc files are
    # all written to the same folder as the .tex file (i.e. next to the script).
    if os.name == 'nt':
        check_cmd   = 'where pdflatex > nul 2>&1'
        compile_cmd = (f'pdflatex -interaction=nonstopmode '
                       f'-output-directory "{script_dir}" "{tex_file}"')
    else:
        check_cmd   = 'which pdflatex > /dev/null 2>&1'
        compile_cmd = (f'pdflatex -interaction=nonstopmode '
                       f'-output-directory "{script_dir}" "{tex_file}"')

    pdflatex_found = (os.system(check_cmd) == 0)

    if pdflatex_found:
        progress("pdflatex found. Compiling (pass 1 of 2)...")
        os.system(compile_cmd)
        progress("Compiling (pass 2 of 2 — for table of contents)...")
        os.system(compile_cmd)

        if os.path.exists(pdf_file):
            progress(f"SUCCESS:  PDF generated →  {pdf_file}")
        else:
            log_file = os.path.splitext(tex_file)[0] + '.log'
            progress(f"WARNING:  pdflatex ran but {pdf_file} was not found.")
            progress(f"          Check {log_file} for compilation errors.")
            progress(f"          Common issues: missing LaTeX package → install via MiKTeX / TeX Live.")
    else:
        progress("pdflatex not found in PATH.")
        progress("")
        progress("To compile the PDF manually, run these commands in a terminal:")
        progress(f'  pdflatex -output-directory "{script_dir}" "{tex_file}"')
        progress(f'  pdflatex -output-directory "{script_dir}" "{tex_file}"   (second pass for ToC)')
        progress("")
        progress("Install pdflatex via one of:")
        progress("  Windows : MiKTeX   — https://miktex.org/download")
        progress("  Linux   : TeX Live — sudo apt install texlive-full")
        progress("  macOS   : MacTeX   — https://tug.org/mactex/")
        progress("  Online  : Overleaf — https://www.overleaf.com  (upload the .tex file)")


# ==============================================================================


# ==============================================================================
# SECTION 8 — MAIN ENTRY POINT
# ==============================================================================



def main():
    wall_start = time.time()

    validate_tetrad_input(e_tetrad)

    progress("=" * 60)
    progress(" GR CALCULATOR — Symbolic General Relativity Analysis")
    progress(" Powered by SymPy")
    progress("=" * 60)
    progress(f"Metric key: {METRIC_KEY}")
    progress(f"Metric   : {METRIC_NAME}")
    progress(f"Dimension: {dim}")
    progress(f"Coords   : {[str(c) for c in coords]}")
    progress(f"Fast mode: {FAST_MODE}")
    progress(f"Weyl     : {COMPUTE_WEYL and not FAST_MODE}")
    progress(f"Kretschm.: {COMPUTE_KRETSCHMANN and not FAST_MODE}")
    _tet_status = ('user-supplied' if e_tetrad is not None
                   else ('auto' if COMPUTE_TETRAD else 'skip'))
    progress(f"Tetrad   : {_tet_status}")
    progress("=" * 60)

    # Run all symbolic computations
    results = run_computations(
        g_metric                 = g_metric,
        coords                   = coords,
        dim                      = dim,
        g_inv_metric             = g_inv_metric,
        e_tetrad                 = e_tetrad,
        compute_weyl_flag        = COMPUTE_WEYL,
        compute_kretschmann_flag = COMPUTE_KRETSCHMANN,
        compute_geodesics_flag   = COMPUTE_GEODESICS,
        compute_killing_flag     = COMPUTE_KILLING,
        compute_tetrad_flag      = COMPUTE_TETRAD,
        fast_mode                = FAST_MODE,
        compute_petrov_flag      = COMPUTE_PETROV,
        compute_horizons_flag    = COMPUTE_HORIZONS,
        compute_killing_full_flag= COMPUTE_KILLING_FULL,
        compute_carter_flag      = COMPUTE_CARTER,
        compute_matter_flag      = COMPUTE_MATTER,
        compute_adm31_flag       = COMPUTE_ADM31,
        compute_penrose_flag     = COMPUTE_PENROSE,
        compute_geodesic_num_flag= COMPUTE_GEODESIC_NUM,
        matter_config            = MATTER_CONFIG,
        numeric_params           = NUMERIC_PARAMS,
        geodesic_x0              = GEODESIC_X0,
        geodesic_u0              = GEODESIC_U0,
        geodesic_lambda          = GEODESIC_LAMBDA,
        geodesic_plot_path       = GEODESIC_PLOT_PATH,
    )

    # Assemble the LaTeX report
    progress("Assembling LaTeX report...")
    report_lines = assemble_report(
        results         = results,
        coords          = coords,
        dim             = dim,
        metric_name     = METRIC_NAME,
        description     = METRIC_DESCRIPTION,
        author          = AUTHOR_NAME,
        latex_subs_dict = latex_subs,
        g_metric_orig   = g_metric,
        e_tetrad        = e_tetrad,
    )

    # Resolve the output path relative to this script's own directory so that
    # gr_report.tex and gr_report.pdf are always written next to gr_calculator.py,
    # no matter which working directory the user runs the script from.
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    output_path  = os.path.join(SCRIPT_DIR, OUTPUT_FILENAME)

    # Write .tex and compile .pdf
    write_and_compile_pdf(report_lines, output_path)

    elapsed = time.time() - wall_start
    progress("")
    progress(f"Total wall-clock time: {elapsed:.1f} s")
    progress("Done.")



if __name__ == '__main__':
    main()
