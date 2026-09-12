#!/usr/bin/env python
"""Focused checks for the calculable quantities in Baez's GR tutorial."""

import sympy as sp

from gr_geodesic_numeric import (
    integrate_geodesic,
    integrate_parallel_transport,
    integrate_parallel_transport_on_geodesic,
    lambdify_christoffel,
)
from gr_tensors import (
    classify_causal_vector,
    compute_christoffel,
    compute_einstein,
    compute_geodesic_deviation_acceleration,
    compute_ricci,
    compute_ricci_scalar,
    compute_riemann,
    compute_torsion,
    einstein_equation_residual,
    get_geodesic_equations,
    get_parallel_transport_equations,
    infer_stress_energy_from_einstein,
    is_torsion_free,
    lower_vector,
    metric_inner_product,
    proper_time_rate,
    raise_covector,
)


def check(label, condition):
    print(f"[{'OK' if condition else 'FAIL'}] {label}")
    return bool(condition)


def main():
    t, x, y, z = sp.symbols('t x y z', real=True)
    coords = [t, x, y, z]
    eta = sp.diag(-1, 1, 1, 1)
    u = sp.Matrix([1, 0, 0, 0])
    spatial = sp.Matrix([0, 1, 0, 0])
    null = sp.Matrix([1, 1, 0, 0])
    checks = []

    checks.append(check('metric product gives timelike norm', metric_inner_product(eta, u, u) == -1))
    checks.append(check('causal types are timelike/spacelike/null',
                        classify_causal_vector(eta, u)['causal_type'] == 'timelike'
                        and classify_causal_vector(eta, spatial)['causal_type'] == 'spacelike'
                        and classify_causal_vector(eta, null)['causal_type'] == 'null'))
    u_low = lower_vector(u, eta)
    checks.append(check('raising and lowering recover the vector', raise_covector(u_low, eta) == u))
    checks.append(check('proper-time rate of unit timelike vector is one', proper_time_rate(eta, u) == 1))

    Gamma = compute_christoffel(eta, eta, coords)
    torsion = compute_torsion(Gamma)
    checks.append(check('Levi-Civita connection is torsion free', is_torsion_free(torsion)))
    checks.append(check('parallel-transport equations are zero in Cartesian Minkowski',
                        all(term == 0 for _, term, _, _ in get_parallel_transport_equations(Gamma, coords))))
    checks.append(check('geodesic equations are zero in Cartesian Minkowski',
                        all(accel == 0 for _, accel, _ in get_geodesic_equations(Gamma, coords))))

    Riem = compute_riemann(Gamma, coords)
    Ric = compute_ricci(Riem)
    R_scalar = compute_ricci_scalar(Ric, eta)
    G = compute_einstein(Ric, eta, eta, R_scalar)
    T = infer_stress_energy_from_einstein(G)
    checks.append(check('flat curvature tensors vanish', R_scalar == 0 and G == sp.zeros(4)))
    checks.append(check('Einstein equation residual vanishes for inferred T',
                        einstein_equation_residual(G, T) == sp.zeros(4)))

    toy_riemann = [[[[sp.S.Zero for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(2)]
    toy_riemann[0][1][0][1] = sp.Integer(7)
    deviation = compute_geodesic_deviation_acceleration(
        toy_riemann, [0, 1], [1, 0], dim=2
    )
    checks.append(check('geodesic-deviation contraction has the expected index order',
                        deviation == sp.Matrix([7, 0])))

    Gamma_num, remaining = lambdify_christoffel(Gamma, coords)
    geodesic = integrate_geodesic(Gamma_num, [0, 0, 0, 0], [1, 0, 0, 0], (0, 1))
    transported = integrate_parallel_transport_on_geodesic(
        Gamma_num, geodesic, [2, 3, 5, 7]
    )
    checks.append(check('numeric parallel transport preserves a vector in flat spacetime',
                        geodesic.success and transported.success
                        and remaining == set()
                        and max(abs(value - initial) for value, initial in zip(transported.y[:, -1], [2, 3, 5, 7])) < 1e-10))

    radius, angle = sp.symbols('radius angle', positive=True)
    polar_coords = [radius, angle]
    polar_metric = sp.diag(1, radius**2)
    Gamma_polar = compute_christoffel(polar_metric, polar_metric.inv(), polar_coords, dim=2)
    Gamma_polar_num, polar_remaining = lambdify_christoffel(
        Gamma_polar, polar_coords, dim=2
    )
    polar_transport = integrate_parallel_transport(
        Gamma_polar_num,
        position=lambda lam: [1.0, lam],
        tangent=lambda lam: [0.0, 1.0],
        V0=[1.0, 0.0],
        lambda_span=(0.0, float(sp.pi / 2)),
        dim=2,
    )
    polar_final = polar_transport.y[:, -1]
    checks.append(check('numeric transport follows the nonzero polar connection',
                        polar_transport.success and polar_remaining == set()
                        and abs(polar_final[0]) < 1e-8
                        and abs(polar_final[1] + 1.0) < 1e-8))

    passed = sum(checks)
    print(f"\nBaez tutorial checks passed: {passed}/{len(checks)}")
    return passed == len(checks)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
