# -*- coding: utf-8 -*-
"""
Metric registry helpers for GR_python.

This module centralizes the built-in metric definitions so the main driver can
select one with a single key instead of commenting and uncommenting blocks.

Two common workflows are supported:

1. Pick a built-in metric by key:
       metric_config = select_metric("schwarzschild", coords, {"M": M})

2. Provide a one-off custom metric:
       metric_config = select_metric("custom", coords, custom_metric_config={...})

To permanently add a new built-in metric, append one more entry to
``build_builtin_metric_library()`` with the same dictionary structure used by
the existing entries.
"""

import sympy as sp
from sympy import Function, Matrix, exp, sin, cos, sqrt, symbols, Rational
from gr_warp import build_metric_configuration


def build_builtin_metric_library(coords, parameter_context=None):
    """
    Return the built-in metric registry for the given coordinate list.

    Parameters
    ----------
    coords : list
        Coordinate symbols in the same order used by the metric matrix.
    parameter_context : dict or None
        Optional pre-defined symbols to reuse, e.g. {"M": M}.
    """
    parameter_context = parameter_context or {}
    t, r, theta, phi = coords

    M = parameter_context.get("M", sp.symbols("M", positive=True))
    Q = parameter_context.get("Q", sp.symbols("Q", real=True))
    Lambda = parameter_context.get("Lambda", sp.symbols("Lambda", real=True))
    f = 1 - 2 * M / r
    f_rn = 1 - 2 * M / r + Q**2 / r**2
    f_ds = 1 - Lambda * r**2 / 3

    a_t = Function("a", positive=True)(t)
    A_r = Function("A")(r)
    B_r = Function("B")(r)
    Phi_r = Function("Phi")(r)
    b_r = Function("b")(r)
    B_func = Function("B", positive=True)(r)
    beta_func = Function("beta")(r)
    beta_doc = Function("beta")(r)
    B_doc = Function("B")(r)

    warp_doc_baseline = build_metric_configuration("baseline", coords, beta_doc)
    warp_doc_variant_a = build_metric_configuration("variant_a", coords, beta_doc, B_doc)
    warp_doc_variant_b = build_metric_configuration("variant_b", coords, beta_doc, B_doc)

    # Kerr metric parameters
    a_kerr = parameter_context.get("a", sp.symbols("a", real=True))  # spin parameter
    rho2 = r**2 + a_kerr**2 * cos(theta)**2                         # ρ² = r² + a²cos²θ
    Delta = r**2 - 2*M*r + a_kerr**2                                 # Δ = r² - 2Mr + a²
    Sigma = (r**2 + a_kerr**2)**2 - a_kerr**2 * Delta * sin(theta)**2  # Σ² (for g_φφ)

    # Kerr-Newman parameters
    Delta_kn = r**2 - 2*M*r + a_kerr**2 + Q**2

    # Anti-de Sitter Schwarzschild
    f_ads = 1 - 2*M/r - Lambda*r**2/3

    return {
        "schwarzschild": {
            "g_metric": Matrix([
                [-f, 0, 0, 0],
                [0, 1 / f, 0, 0],
                [0, 0, r**2, 0],
                [0, 0, 0, r**2 * sin(theta)**2],
            ]),
            "metric_name": "Schwarzschild",
            "metric_description": (
                "Exterior vacuum solution for a spherically symmetric, "
                "non-rotating mass M"
            ),
            "g_inv_metric": None,
            "e_tetrad": None,
        },
        "reissner_nordstrom": {
            "g_metric": Matrix([
                [-f_rn, 0, 0, 0],
                [0, 1 / f_rn, 0, 0],
                [0, 0, r**2, 0],
                [0, 0, 0, r**2 * sin(theta)**2],
            ]),
            "metric_name": "Reissner-Nordstrom",
            "metric_description": (
                "Charged, static, spherically symmetric black-hole metric "
                "with mass M and charge Q"
            ),
            "g_inv_metric": None,
            "e_tetrad": None,
        },
        "de_sitter_static": {
            "g_metric": Matrix([
                [-f_ds, 0, 0, 0],
                [0, 1 / f_ds, 0, 0],
                [0, 0, r**2, 0],
                [0, 0, 0, r**2 * sin(theta)**2],
            ]),
            "metric_name": "de Sitter (static patch)",
            "metric_description": (
                "Static-patch de Sitter spacetime with cosmological constant Lambda"
            ),
            "g_inv_metric": None,
            "e_tetrad": None,
        },
        "morris_thorne_wormhole": {
            "g_metric": Matrix([
                [-sp.exp(2 * Phi_r), 0, 0, 0],
                [0, 1 / (1 - b_r / r), 0, 0],
                [0, 0, r**2, 0],
                [0, 0, 0, r**2 * sin(theta)**2],
            ]),
            "metric_name": "Morris-Thorne wormhole",
            "metric_description": (
                "Static traversable wormhole metric with redshift Phi(r) and shape function b(r)"
            ),
            "g_inv_metric": None,
            "e_tetrad": None,
        },
        "minkowski_spherical": {
            "g_metric": Matrix([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, r**2, 0],
                [0, 0, 0, r**2 * sin(theta)**2],
            ]),
            "metric_name": "Minkowski (spherical)",
            "metric_description": "Flat Minkowski spacetime in spherical coordinates",
            "g_inv_metric": None,
            "e_tetrad": None,
        },
        "frw_flat": {
            "g_metric": Matrix([
                [-1, 0, 0, 0],
                [0, a_t**2, 0, 0],
                [0, 0, a_t**2 * r**2, 0],
                [0, 0, 0, a_t**2 * r**2 * sin(theta)**2],
            ]),
            "metric_name": "FRW (flat)",
            "metric_description": (
                "Flat FLRW cosmological metric with scale factor a(t)"
            ),
            "g_inv_metric": None,
            "e_tetrad": None,
        },
        "static_spherical": {
            "g_metric": Matrix([
                [-exp(2 * A_r), 0, 0, 0],
                [0, exp(2 * B_r), 0, 0],
                [0, 0, r**2, 0],
                [0, 0, 0, r**2 * sin(theta)**2],
            ]),
            "metric_name": "General static spherical",
            "metric_description": (
                "Static spherically symmetric metric with arbitrary radial "
                "functions A(r), B(r)"
            ),
            "g_inv_metric": None,
            "e_tetrad": None,
        },
        "pg_areal": {
            "g_metric": Matrix([
                [-(1 - B_func**2 * beta_func**2), B_func**2 * beta_func, 0, 0],
                [B_func**2 * beta_func, B_func**2, 0, 0],
                [0, 0, r**2, 0],
                [0, 0, 0, r**2 * sin(theta)**2],
            ]),
            "metric_name": "PG-type warp metric (areal gauge)",
            "metric_description": (
                "Painleve-Gullstrand-type metric with lapse N=1, radial shift "
                "beta(r), and areal radius"
            ),
            "g_inv_metric": None,
            "e_tetrad": None,
        },
        "warp_doc_baseline": {
            "g_metric": warp_doc_baseline["g_metric"],
            "metric_name": warp_doc_baseline["metric_name"],
            "metric_description": warp_doc_baseline["description"],
            "g_inv_metric": None,
            "e_tetrad": warp_doc_baseline["e_tetrad"],
        },
        "warp_doc_variant_a": {
            "g_metric": warp_doc_variant_a["g_metric"],
            "metric_name": warp_doc_variant_a["metric_name"],
            "metric_description": warp_doc_variant_a["description"],
            "g_inv_metric": None,
            "e_tetrad": warp_doc_variant_a["e_tetrad"],
        },
        "warp_doc_variant_b": {
            "g_metric": warp_doc_variant_b["g_metric"],
            "metric_name": warp_doc_variant_b["metric_name"],
            "metric_description": warp_doc_variant_b["description"],
            "g_inv_metric": None,
            "e_tetrad": warp_doc_variant_b["e_tetrad"],
        },
        "kerr": {
            "g_metric": Matrix([
                [-(1 - 2*M*r/rho2), 0, 0, 2*M*r*a_kerr*sin(theta)**2/rho2],
                [0, rho2/Delta, 0, 0],
                [0, 0, rho2, 0],
                [2*M*r*a_kerr*sin(theta)**2/rho2, 0, 0,
                 (r**2 + a_kerr**2 + 2*M*r*a_kerr**2*sin(theta)**2/rho2)*sin(theta)**2],
            ]),
            "metric_name": "Kerr (Boyer-Lindquist)",
            "metric_description": (
                "Rotating black hole with mass M and specific angular momentum a; "
                "rho^2 = r^2 + a^2 cos^2(theta), Delta = r^2 - 2Mr + a^2"
            ),
            "g_inv_metric": None,
            "e_tetrad": None,
        },
        "kerr_newman": {
            "g_metric": Matrix([
                [-(1 - (2*M*r - Q**2)/rho2), 0, 0,
                 (2*M*r - Q**2)*a_kerr*sin(theta)**2/rho2],
                [0, rho2/Delta_kn, 0, 0],
                [0, 0, rho2, 0],
                [(2*M*r - Q**2)*a_kerr*sin(theta)**2/rho2, 0, 0,
                 (r**2 + a_kerr**2 + (2*M*r - Q**2)*a_kerr**2*sin(theta)**2/rho2)*sin(theta)**2],
            ]),
            "metric_name": "Kerr-Newman (Boyer-Lindquist)",
            "metric_description": (
                "Charged rotating black hole with mass M, charge Q, spin a; "
                "Delta_KN = r^2 - 2Mr + a^2 + Q^2"
            ),
            "g_inv_metric": None,
            "e_tetrad": None,
        },
        "ads_schwarzschild": {
            "g_metric": Matrix([
                [-f_ads, 0, 0, 0],
                [0, 1/f_ads, 0, 0],
                [0, 0, r**2, 0],
                [0, 0, 0, r**2*sin(theta)**2],
            ]),
            "metric_name": "Schwarzschild-AdS",
            "metric_description": (
                "Schwarzschild black hole in Anti-de Sitter background; "
                "f = 1 - 2M/r - Lambda*r^2/3 with Lambda < 0 for AdS"
            ),
            "g_inv_metric": None,
            "e_tetrad": None,
        },
        "anti_de_sitter": {
            "g_metric": Matrix([
                [-(1 + r**2), 0, 0, 0],
                [0, 1/(1 + r**2), 0, 0],
                [0, 0, r**2, 0],
                [0, 0, 0, r**2*sin(theta)**2],
            ]),
            "metric_name": "Anti-de Sitter (global)",
            "metric_description": (
                "Global Anti-de Sitter spacetime in static coordinates with unit AdS radius"
            ),
            "g_inv_metric": None,
            "e_tetrad": None,
        },
        "pg_spatial_conformal": {
            "g_metric": Matrix([
                [-(1 - B_func**2 * beta_func**2), B_func**2 * beta_func, 0, 0],
                [B_func**2 * beta_func, B_func**2, 0, 0],
                [0, 0, B_func**2 * r**2, 0],
                [0, 0, 0, B_func**2 * r**2 * sin(theta)**2],
            ]),
            "metric_name": "PG-type warp metric (Spatial Conformal)",
            "metric_description": (
                "Painleve-Gullstrand-type metric with lapse N=1, radial shift "
                "beta(r), and a conformal factor on the full spatial sector"
            ),
            "g_inv_metric": None,
            "e_tetrad": None,
        },
    }


def list_builtin_metric_keys():
    """
    Return the list of built-in metric keys.
    """
    demo_coords = sp.symbols("t r theta phi", real=True)
    return sorted(build_builtin_metric_library(list(demo_coords)).keys())


def validate_metric_config(metric_config, dim):
    """
    Validate and normalize a metric configuration dictionary.
    """
    required = ("g_metric", "metric_name", "metric_description")
    missing = [key for key in required if key not in metric_config]
    if missing:
        raise KeyError(
            "Metric config is missing required keys: {}".format(", ".join(missing))
        )

    normalized = dict(metric_config)
    normalized.setdefault("g_inv_metric", None)
    normalized.setdefault("e_tetrad", None)

    g_metric = normalized["g_metric"]
    if not isinstance(g_metric, sp.MatrixBase):
        raise TypeError("metric_config['g_metric'] must be a SymPy Matrix.")
    if g_metric.shape != (dim, dim):
        raise ValueError(
            "metric_config['g_metric'] must have shape ({0}, {0}).".format(dim)
        )

    g_inv_metric = normalized["g_inv_metric"]
    if g_inv_metric is not None:
        if not isinstance(g_inv_metric, sp.MatrixBase):
            raise TypeError("metric_config['g_inv_metric'] must be a SymPy Matrix or None.")
        if g_inv_metric.shape != (dim, dim):
            raise ValueError(
                "metric_config['g_inv_metric'] must have shape ({0}, {0}).".format(dim)
            )

    e_tetrad = normalized["e_tetrad"]
    if e_tetrad is not None:
        if not isinstance(e_tetrad, sp.MatrixBase):
            raise TypeError("metric_config['e_tetrad'] must be a SymPy Matrix or None.")
        if e_tetrad.shape != (dim, dim):
            raise ValueError(
                "metric_config['e_tetrad'] must have shape ({0}, {0}).".format(dim)
            )

    return normalized


def select_metric(metric_key, coords, parameter_context=None, custom_metric_config=None):
    """
    Select either a built-in metric or a user-provided custom metric.

    Parameters
    ----------
    metric_key : str
        Built-in metric key, or the literal string "custom".
    coords : list
        Coordinate symbols used by the current run.
    parameter_context : dict or None
        Optional parameters reused by built-in metrics.
    custom_metric_config : dict or None
        User-supplied metric configuration with keys:
        g_metric, metric_name, metric_description, and optional
        g_inv_metric, e_tetrad.
    """
    dim = len(coords)

    if metric_key == "custom":
        if custom_metric_config is None:
            raise ValueError(
                "METRIC_KEY='custom' requires CUSTOM_METRIC_CONFIG to be defined."
            )
        return validate_metric_config(custom_metric_config, dim)

    library = build_builtin_metric_library(coords, parameter_context)
    if metric_key not in library:
        raise KeyError(
            "Unknown METRIC_KEY {!r}. Available keys: {}".format(
                metric_key, ", ".join(sorted(library.keys()))
            )
        )
    return validate_metric_config(library[metric_key], dim)
