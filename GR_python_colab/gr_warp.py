# -*- coding: utf-8 -*-
"""
Helpers for the warp-bubble metrics discussed in the companion notes.

This module provides:
  * exact metric/tetrad builders for the three document variants
  * symbolic stress-energy formulas from the notes
  * comparison helpers against GR_python's computed orthonormal tensor
"""

import sympy as sp
from sympy import Matrix, Rational, cancel, diff, pi, simplify, sin, sqrt


VARIANT_ALIASES = {
    "baseline": "baseline",
    "pg": "baseline",
    "variant_a": "variant_a",
    "radial": "variant_a",
    "radial_only": "variant_a",
    "variant_b": "variant_b",
    "full_spatial": "variant_b",
    "full": "variant_b",
}


def normalize_variant_name(variant):
    key = str(variant).strip().lower()
    if key not in VARIANT_ALIASES:
        raise ValueError(
            "Unknown variant {!r}. Use one of: baseline, variant_a, variant_b.".format(variant)
        )
    return VARIANT_ALIASES[key]


def _coframe_to_tetrad(e_cov):
    """Return e^mu_a from the coframe e^a_mu."""
    return Matrix(e_cov.inv())


def build_metric_configuration(variant, coords, beta_expr, B_expr=None):
    """
    Construct the document metric and a matching orthonormal tetrad.
    """
    variant = normalize_variant_name(variant)
    _t, r, theta, _phi = coords
    angular = r**2 * sin(theta) ** 2

    if variant == "baseline":
        g_metric = Matrix(
            [
                [-(1 - beta_expr**2), -beta_expr, 0, 0],
                [-beta_expr, 1, 0, 0],
                [0, 0, r**2, 0],
                [0, 0, 0, angular],
            ]
        )
        e_cov = Matrix(
            [
                [1, 0, 0, 0],
                [-beta_expr, 1, 0, 0],
                [0, 0, r, 0],
                [0, 0, 0, r * sin(theta)],
            ]
        )
        metric_name = "Baseline PG Warp Metric"
        description = (
            "Painleve-Gullstrand-like warp metric with unit lapse and radial shift beta(r)."
        )
    elif variant == "variant_a":
        if B_expr is None:
            raise ValueError("Variant A requires B_expr.")
        sqrt_B = sqrt(B_expr)
        g_metric = Matrix(
            [
                [-(1 - B_expr * beta_expr**2), -B_expr * beta_expr, 0, 0],
                [-B_expr * beta_expr, B_expr, 0, 0],
                [0, 0, r**2, 0],
                [0, 0, 0, angular],
            ]
        )
        e_cov = Matrix(
            [
                [1, 0, 0, 0],
                [-sqrt_B * beta_expr, sqrt_B, 0, 0],
                [0, 0, r, 0],
                [0, 0, 0, r * sin(theta)],
            ]
        )
        metric_name = "Variant A - Radial-Only Conformal Warp Metric"
        description = (
            "Static warp metric where B(r) multiplies only the radial PG sector."
        )
    else:
        if B_expr is None:
            raise ValueError("Variant B requires B_expr.")
        sqrt_B = sqrt(B_expr)
        g_metric = Matrix(
            [
                [-(1 - B_expr * beta_expr**2), -B_expr * beta_expr, 0, 0],
                [-B_expr * beta_expr, B_expr, 0, 0],
                [0, 0, B_expr * r**2, 0],
                [0, 0, 0, B_expr * angular],
            ]
        )
        e_cov = Matrix(
            [
                [1, 0, 0, 0],
                [-sqrt_B * beta_expr, sqrt_B, 0, 0],
                [0, 0, sqrt_B * r, 0],
                [0, 0, 0, sqrt_B * r * sin(theta)],
            ]
        )
        metric_name = "Variant B - Full Spatial Conformal Warp Metric"
        description = (
            "Static warp metric where B(r) multiplies the entire spatial 3-metric."
        )

    return {
        "variant": variant,
        "g_metric": g_metric,
        "e_cov": e_cov,
        "e_tetrad": _coframe_to_tetrad(e_cov),
        "metric_name": metric_name,
        "description": description,
    }


def document_stress_energy_formulas(variant, r, beta_expr, B_expr=None):
    """
    Return the document formulas for rho, p_r, p_perp and q.

    All quantities are returned in orthonormal-frame stress-energy form,
    i.e. already divided by 8*pi.
    """
    variant = normalize_variant_name(variant)
    beta_p = diff(beta_expr, r)
    beta_pp = diff(beta_p, r)
    zero = sp.S.Zero

    if variant == "baseline":
        rho = beta_expr * (2 * r * beta_p + beta_expr) / (8 * pi * r**2)
        p_r = -rho
        p_perp = (-beta_expr * beta_pp - beta_p**2 - 2 * beta_expr * beta_p / r) / (8 * pi)
        q = zero
    elif variant == "variant_a":
        if B_expr is None:
            raise ValueError("Variant A requires B_expr.")
        S_expr = diff(B_expr, r) / B_expr
        S_p = diff(S_expr, r)
        rho = (
            beta_expr * (2 * r * beta_p + beta_expr) / r**2
            + (1 - B_expr * beta_expr**2) * S_expr / r
        ) / (8 * pi)
        p_r = (
            -beta_expr * (2 * r * beta_p + beta_expr) / r**2
            + (1 - B_expr * beta_expr**2) * S_expr / r
        ) / (8 * pi)
        q = beta_expr * S_expr * (2 * beta_p + beta_expr / r) / (8 * pi)
        p_perp = (
            beta_expr * beta_p / r
            + Rational(1, 2)
            * (1 - B_expr * beta_expr**2)
            * (S_p + S_expr / r - S_expr**2 / 2)
        ) / (8 * pi)
    else:
        if B_expr is None:
            raise ValueError("Variant B requires B_expr.")
        B_p = diff(B_expr, r)
        B_pp = diff(B_p, r)
        G00 = (
            beta_expr**2 * (r * B_p + 2 * B_expr) * (3 * r * B_p + 2 * B_expr)
            / (4 * r**2 * B_expr**2)
            + beta_expr * beta_p * (r * B_p + 2 * B_expr) / (r * B_expr)
            - (4 * r * B_expr * B_pp - 3 * r * B_p**2 + 8 * B_expr * B_p)
            / (4 * r * B_expr**3)
        )
        G11 = (
            -beta_expr**2
            * (4 * r**2 * B_expr * B_pp - r**2 * B_p**2 + 12 * r * B_expr * B_p + 4 * B_expr**2)
            / (4 * r**2 * B_expr**2)
            - beta_expr * beta_p * (r * B_p + 2 * B_expr) / (r * B_expr)
            + (r * B_p + 4 * B_expr) * B_p / (4 * r * B_expr**3)
        )
        G22 = (
            -beta_expr * beta_pp
            - beta_p**2
            - (5 * r * B_p + 4 * B_expr) * beta_expr * beta_p / (2 * r * B_expr)
            - beta_expr**2 * (4 * r * B_expr * B_pp - r * B_p**2 + 6 * B_expr * B_p)
            / (4 * r * B_expr**2)
            + (r * B_expr * B_pp - r * B_p**2 + B_expr * B_p) / (2 * r * B_expr**3)
        )
        G01 = -(
            (r * B_expr * B_pp - r * B_p**2 + B_expr * B_p) * beta_expr
        ) / (r * B_expr ** Rational(5, 2))
        rho = G00 / (8 * pi)
        p_r = G11 / (8 * pi)
        p_perp = G22 / (8 * pi)
        q = G01 / (8 * pi)

    return {
        "rho": cancel(rho),
        "p_r": cancel(p_r),
        "p_perp": cancel(p_perp),
        "q": cancel(q),
    }


def _simplify_residual(expr):
    expr = cancel(expr)
    if expr != 0:
        expr = simplify(expr)
    return expr


def _is_zero(expr):
    try:
        return _simplify_residual(expr) == 0
    except Exception:
        return False


def compare_document_formulas(results, variant, r, beta_expr, B_expr=None):
    """
    Compare the document formulas with the computed orthonormal Einstein tensor.
    """
    if results.get("G_ortho") is None:
        raise ValueError("results['G_ortho'] is missing. Enable tetrad computation first.")

    computed = {
        "rho": cancel(results["G_ortho"][0, 0] / (8 * pi)),
        "p_r": cancel(results["G_ortho"][1, 1] / (8 * pi)),
        "p_perp": cancel(results["G_ortho"][2, 2] / (8 * pi)),
        "q": cancel(results["G_ortho"][0, 1] / (8 * pi)),
        "p_phi": cancel(results["G_ortho"][3, 3] / (8 * pi)),
    }
    expected = document_stress_energy_formulas(variant, r, beta_expr, B_expr)
    residuals = {
        "rho": _simplify_residual(computed["rho"] - expected["rho"]),
        "p_r": _simplify_residual(computed["p_r"] - expected["p_r"]),
        "p_perp": _simplify_residual(computed["p_perp"] - expected["p_perp"]),
        "q": _simplify_residual(computed["q"] - expected["q"]),
        "p_theta_minus_p_phi": _simplify_residual(computed["p_perp"] - computed["p_phi"]),
    }
    checks = {name: _is_zero(resid) for name, resid in residuals.items()}
    return {
        "variant": normalize_variant_name(variant),
        "computed": computed,
        "expected": expected,
        "residuals": residuals,
        "checks": checks,
    }


def print_formula_comparison(comparison):
    """
    Print a compact summary of the comparison results.
    """
    variant = comparison["variant"]
    print("Document comparison for variant:", variant)
    for key in ("rho", "p_r", "p_perp", "q", "p_theta_minus_p_phi"):
        status = "OK" if comparison["checks"][key] else "MISMATCH"
        print(f"  {key}: {status}")
