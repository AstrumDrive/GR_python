# -*- coding: utf-8 -*-
"""
Regression tests for the ADM 3+1 extrinsic curvature on shifted slicings.

Background
----------
compute_extrinsic_curvature used to build the covariant derivative of the
shift from the FOUR-dimensional Christoffel symbols restricted to spatial
indices, while the ADM definition calls for the Levi-Civita connection of
gamma_ij.  The two coincide only when g_{0i} = 0 -- that is, only for slicings
where K_ij vanishes anyway -- so every metric with a shift came out wrong.
For the Painleve-Gullstrand class the old code returned

    K_rr = -(N^2 - beta^2) beta' / N^3     instead of     K_rr = -beta'/N,

a discrepancy by the factor 1 - 2Gm/c^2 r, and the Hamiltonian constraint
failed with a residual proportional to beta^3.

The package's only ADM validation ran on frw_flat (lapse 1, zero shift), which
is why this was invisible.  These tests close that gap.

Run with:  python -m pytest tests/test_adm_pg.py -q     (from the package root)
"""

import os
import sys

import pytest
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gr_adm31 import (  # noqa: E402
    compute_3christoffel,
    compute_3ricci_scalar,
    compute_adm_constraints,
    compute_extrinsic_curvature,
    extract_adm_variables,
)
from gr_tensors import compute_christoffel  # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _zero(expr):
    """True when expr simplifies to zero."""
    return sp.simplify(sp.expand(sp.together(sp.cancel(expr)))) == 0


def _matrix_zero(M):
    return all(_zero(M[i, j]) for i in range(M.rows) for j in range(M.cols))


def _K_from_lie_derivative(adm, coords):
    """
    K_ij built WITHOUT any Christoffel symbols, as an independent route.

    The symmetrised covariant derivative of the lowered shift is the Lie
    derivative of the spatial metric along the shift -- an identity that holds
    for the Levi-Civita connection of gamma and fails for any other:

        D_i beta_j + D_j beta_i = (L_beta gamma)_ij
                                = beta^k d_k gamma_ij
                                  + gamma_kj d_i beta^k + gamma_ik d_j beta^k

    so that   K_ij = (1/2N) ( -d_t gamma_ij + (L_beta gamma)_ij ).

    Only partial derivatives appear, so this cross-check shares no code path
    with compute_extrinsic_curvature.
    """
    d = len(coords) - 1
    t = coords[0]
    sc = coords[1:]
    gamma = adm['gamma_cov']
    bup = adm['beta_contra']
    N = adm['N']

    K = sp.zeros(d)
    for i in range(d):
        for j in range(d):
            lie = sum(bup[k] * sp.diff(gamma[i, j], sc[k]) for k in range(d))
            lie += sum(gamma[k, j] * sp.diff(bup[k], sc[i]) for k in range(d))
            lie += sum(gamma[i, k] * sp.diff(bup[k], sc[j]) for k in range(d))
            K[i, j] = sp.cancel((-sp.diff(gamma[i, j], t) + lie) / (2 * N))
    return K


@pytest.fixture(scope="module")
def pg():
    """
    Painleve-Gullstrand with a general lapse and a general radial shift:

        ds^2 = -(N^2 - beta^2) dt^2 - 2 beta dr dt + dr^2 + r^2 dOmega^2

    The spatial slice is flat, so R^(3) = 0 and the Hamiltonian constraint is
    carried entirely by K_ij.
    """
    t, r, th, ph = sp.symbols('t r theta phi', real=True)
    coords = [t, r, th, ph]
    N = sp.Function('N', positive=True)(r)
    b = sp.Function('beta', positive=True)(r)

    g = sp.Matrix([[-N**2 + b**2, -b, 0, 0],
                   [-b, 1, 0, 0],
                   [0, 0, r**2, 0],
                   [0, 0, 0, r**2 * sp.sin(th)**2]])
    ginv = sp.simplify(g.inv())
    Gamma = compute_christoffel(g, ginv, coords)
    adm = extract_adm_variables(g, ginv, coords)
    ex = compute_extrinsic_curvature(g, ginv, Gamma, coords, adm)
    return dict(coords=coords, N=N, b=b, r=r, th=th,
                g=g, ginv=ginv, Gamma=Gamma, adm=adm, ex=ex)


# ----------------------------------------------------------------------------
# the pinned values
# ----------------------------------------------------------------------------

def test_pg_extrinsic_curvature_matches_textbook(pg):
    """K_rr = -beta'/N, K_thth = -r beta/N, K_phph = -r beta sin^2(theta)/N."""
    r, th, N, b = pg['r'], pg['th'], pg['N'], pg['b']
    K = pg['ex']['K_cov']

    expected = sp.diag(-sp.diff(b, r) / N,
                       -r * b / N,
                       -r * b * sp.sin(th)**2 / N)

    assert _matrix_zero(K - expected), (
        "K_ij does not match the ADM values for Painleve-Gullstrand; got "
        f"{sp.simplify(K)}"
    )


def test_pg_hamiltonian_constraint_returns_the_class_density(pg):
    """rho read off the Hamiltonian constraint is (r beta^2)' / (8 pi r^2 N^2)."""
    r, th, N, b = pg['r'], pg['th'], pg['N'], pg['b']
    adm, ex = pg['adm'], pg['ex']

    ricci3 = compute_3ricci_scalar(adm['gamma_cov'], adm['gamma_inv'], pg['coords'])
    assert _zero(ricci3['R3']), "the PG slice is flat, R^(3) must vanish"

    target = sp.diff(r * b**2, r) / (8 * sp.pi * r**2 * N**2)

    # rho as read off the constraint H = R3 + K^2 - K_ij K^ij = 16 pi rho
    rho = (ricci3['R3'] + ex['K_trace']**2 - ex['K_sq']) / (16 * sp.pi)
    assert _zero(rho - target), (
        f"Hamiltonian constraint gives rho = {sp.simplify(rho)}, "
        f"expected {sp.simplify(target)}"
    )

    # and the same statement through the public constraint API
    con = compute_adm_constraints(
        ricci3['R3'], ex['K_sq'], ex['K_trace'], adm['gamma_inv'],
        ricci3['Gamma3'], ex['K_cov'], pg['coords'], rho_matter=target,
    )
    assert _zero(con['H_residual']), (
        f"H - 16 pi rho = {sp.simplify(con['H_residual'])}, expected 0"
    )


def test_pg_momentum_constraint_gives_the_radial_flux(pg):
    """
    The momentum constraint is not sourceless for a general lapse: the class
    carries a radial energy flux

        M^r = 8 pi J^r = -2 beta N' / (r N^2),      M^theta = M^phi = 0,

    which was checked against the 4D Einstein tensor of the same metric (a
    route sharing no code with gr_adm31).  It vanishes exactly when N is
    constant, i.e. for pure Painleve-Gullstrand.
    """
    r, N, b = pg['r'], pg['N'], pg['b']
    adm, ex = pg['adm'], pg['ex']

    ricci3 = compute_3ricci_scalar(adm['gamma_cov'], adm['gamma_inv'], pg['coords'])
    con = compute_adm_constraints(
        ricci3['R3'], ex['K_sq'], ex['K_trace'], adm['gamma_inv'],
        ricci3['Gamma3'], ex['K_cov'], pg['coords'],
    )

    expected_Mr = -2 * b * sp.diff(N, r) / (r * N**2)
    assert _zero(con['M'][0] - expected_Mr), (
        f"M^r = {sp.simplify(con['M'][0])}, expected {sp.simplify(expected_Mr)}"
    )
    assert _zero(con['M'][1]) and _zero(con['M'][2]), (
        "the angular momentum constraints must vanish for a radial shift"
    )

    # constant lapse -> no flux
    assert _zero(con['M'][0].subs(N, sp.Integer(1)).doit()), (
        "pure Painleve-Gullstrand (N = 1) must have a vanishing momentum "
        "constraint"
    )


def test_pg_constraints_agree_with_the_four_dimensional_einstein_tensor(pg):
    """
    The decisive check, and the one that does not depend on knowing the answer
    in advance: build rho and J^i from the 4D Einstein tensor -- code that
    never touches gr_adm31 -- and require the 3+1 constraints to reproduce
    them.

        rho = n^mu n^nu T_mu_nu          must equal   H / (16 pi)
        J^i = -gamma^ij n^mu T_j_mu      must equal   M^i / (8 pi)

    With the 4D-sliced connection this fails; with the Levi-Civita connection
    of gamma it holds identically.
    """
    from gr_tensors import compute_ricci, compute_ricci_scalar, compute_riemann

    coords, g, ginv, Gamma = pg['coords'], pg['g'], pg['ginv'], pg['Gamma']
    adm, ex = pg['adm'], pg['ex']

    Ric = compute_ricci(compute_riemann(Gamma, coords))
    Rs = compute_ricci_scalar(Ric, ginv)
    T4 = sp.Matrix(4, 4, lambda i, j:
                   (Ric[i, j] - sp.Rational(1, 2) * g[i, j] * Rs) / (8 * sp.pi))

    gamma_inv = adm['gamma_inv']
    bup = adm['beta_contra']
    n_up = [1 / adm['N']] + [-bup[i] / adm['N'] for i in range(3)]

    rho_4d = sum(n_up[m] * n_up[n] * T4[m, n] for m in range(4) for n in range(4))
    J_4d = [
        -sum(gamma_inv[i, j] * sum(T4[j + 1, m] * n_up[m] for m in range(4))
             for j in range(3))
        for i in range(3)
    ]

    ricci3 = compute_3ricci_scalar(adm['gamma_cov'], gamma_inv, coords)
    con = compute_adm_constraints(
        ricci3['R3'], ex['K_sq'], ex['K_trace'], gamma_inv,
        ricci3['Gamma3'], ex['K_cov'], coords,
    )

    assert _zero(con['H'] / (16 * sp.pi) - rho_4d), (
        f"rho: 3+1 gives {sp.simplify(con['H'] / (16 * sp.pi))}, "
        f"4D gives {sp.simplify(rho_4d)}"
    )
    for i, name in enumerate(('r', 'theta', 'phi')):
        assert _zero(con['M'][i] / (8 * sp.pi) - J_4d[i]), (
            f"J^{name}: 3+1 gives {sp.simplify(con['M'][i] / (8 * sp.pi))}, "
            f"4D gives {sp.simplify(J_4d[i])}"
        )


# ----------------------------------------------------------------------------
# the regression itself: the old connection must not come back
# ----------------------------------------------------------------------------

def test_three_and_four_dimensional_connections_differ_under_a_shift(pg):
    """
    Guard on the test's own sensitivity.  If these two ever agreed for this
    metric, the tests above would pass no matter which one the code used.
    """
    adm = pg['adm']
    sc = pg['coords'][1:]
    Gamma3 = compute_3christoffel(adm['gamma_cov'], adm['gamma_inv'], sc)
    Gamma4 = pg['Gamma']

    differing = [
        (a, i, j)
        for a in range(3) for i in range(3) for j in range(3)
        if not _zero(Gamma4[a + 1][i + 1][j + 1] - Gamma3[a][i][j])
    ]
    assert differing, (
        "the 3D and 4D connections coincide for this metric, so it cannot "
        "discriminate between them -- pick a metric with a shift"
    )


def test_old_four_dimensional_slicing_would_fail_these_tests(pg):
    """
    Reproduce the defect explicitly and show it is caught.  This is what
    compute_extrinsic_curvature used to do.
    """
    r, th, N, b = pg['r'], pg['th'], pg['N'], pg['b']
    adm, Gamma4 = pg['adm'], pg['Gamma']
    beta = adm['beta_cov']
    sc = pg['coords'][1:]

    def nabla_beta_4d(i, j):
        s = sp.diff(beta[j], sc[i])
        for k in range(3):
            s -= Gamma4[k + 1][i + 1][j + 1] * beta[k]
        return sp.cancel(s)

    K_bad = sp.zeros(3)
    for i in range(3):
        for j in range(3):
            K_bad[i, j] = sp.cancel(
                (nabla_beta_4d(i, j) + nabla_beta_4d(j, i)) / (2 * N))

    # the historical wrong value, and its ratio to the right one
    assert _zero(K_bad[0, 0] - (-(N**2 - b**2) * sp.diff(b, r) / N**3))
    assert _zero(sp.cancel(K_bad[0, 0] / (-sp.diff(b, r) / N)) - (N**2 - b**2) / N**2)

    # and it breaks the Hamiltonian constraint
    ginv3 = adm['gamma_inv']
    Kmix = sp.simplify(ginv3 * K_bad)
    trK = sp.trace(Kmix)
    KK = sum(Kmix[i, j] * Kmix[j, i] for i in range(3) for j in range(3))
    rho_bad = sp.cancel((trK**2 - KK) / (16 * sp.pi))
    target = sp.diff(r * b**2, r) / (8 * sp.pi * r**2 * N**2)
    assert not _zero(rho_bad - target), (
        "the 4D-sliced connection unexpectedly satisfies the constraint; the "
        "regression guard is no longer meaningful"
    )


# ----------------------------------------------------------------------------
# metric-agnostic invariant, on slices whose 3-connection is non-trivial
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["pg_areal", "pg_spatial_conformal"])
def test_K_agrees_with_the_connection_free_lie_derivative(key):
    """
    For any metric, K_ij must equal (1/2N)(-d_t gamma + L_beta gamma), which
    uses no connection at all.  The identity D_(i beta_j) = (1/2) L_beta gamma
    holds for the Levi-Civita connection of gamma and for no other, so this
    pins the fix on library metrics whose spatial slice is curved -- where
    3Gamma^r_rr is itself non-zero and a "flat slice" shortcut would fail.
    """
    from gr_metric_library import select_metric

    t, r, th, ph = sp.symbols('t r theta phi', real=True)
    coords = [t, r, th, ph]
    cfg = select_metric(key, coords)
    g = cfg['g_metric']
    ginv = cfg['g_inv_metric'] or sp.simplify(g.inv())
    Gamma = compute_christoffel(g, ginv, coords)

    adm = extract_adm_variables(g, ginv, coords)
    assert any(not _zero(bc) for bc in adm['beta_cov']), (
        f"{key} has no shift; it cannot exercise the regression"
    )

    K = compute_extrinsic_curvature(g, ginv, Gamma, coords, adm)['K_cov']
    K_ref = _K_from_lie_derivative(adm, coords)

    assert _matrix_zero(K - K_ref), (
        f"{key}: K_ij disagrees with the Lie-derivative form; "
        f"difference {sp.simplify(K - K_ref)}"
    )


def test_zero_shift_slicing_is_unchanged():
    """
    Schwarzschild in Schwarzschild coordinates: no shift, static slice, so
    K_ij = 0.  This is the case the old code got right, and it must stay right.
    """
    t, r, th, ph = sp.symbols('t r theta phi', positive=True)
    coords = [t, r, th, ph]
    M = sp.Symbol('M', positive=True)
    f = 1 - 2 * M / r

    g = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(th)**2)
    ginv = sp.diag(-1 / f, f, 1 / r**2, 1 / (r**2 * sp.sin(th)**2))
    Gamma = compute_christoffel(g, ginv, coords)

    adm = extract_adm_variables(g, ginv, coords)
    ex = compute_extrinsic_curvature(g, ginv, Gamma, coords, adm)

    assert _matrix_zero(ex['K_cov'])
    assert _zero(ex['K_trace'])
    assert _zero(ex['K_sq'])
