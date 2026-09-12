"""Pytest coverage for the Baez-tutorial helpers in gr_tensors.

validate_baez_tutorial.py is a standalone smoke check and is kept as such.
These tests exist so the same surface runs under pytest, and so that
contract_rank_02, the one helper the standalone validator never calls, is
exercised.
"""
import sympy as sp

from gr_tensors import (
    classify_causal_vector,
    compute_torsion,
    contract_rank_02,
    infer_stress_energy_from_einstein,
    is_torsion_free,
    lower_vector,
    metric_inner_product,
    proper_time_rate,
    raise_covector,
)


def _minkowski():
    return sp.diag(-1, 1, 1, 1)


def test_metric_inner_product_signs_match_the_mostly_plus_convention():
    eta = _minkowski()
    assert metric_inner_product(eta, [1, 0, 0, 0], [1, 0, 0, 0]) == -1
    assert metric_inner_product(eta, [0, 1, 0, 0], [0, 1, 0, 0]) == 1
    assert metric_inner_product(eta, [1, 0, 0, 0], [0, 1, 0, 0]) == 0


def test_classify_causal_vector_on_the_three_cases():
    eta = _minkowski()
    assert classify_causal_vector(eta, [1, 0, 0, 0])["causal_type"] == "timelike"
    assert classify_causal_vector(eta, [0, 1, 0, 0])["causal_type"] == "spacelike"
    assert classify_causal_vector(eta, [1, 1, 0, 0])["causal_type"] == "null"
    assert classify_causal_vector(eta, [1, 0, 0, 0])["norm_sq"] == -1


def test_classify_causal_vector_refuses_to_guess_an_undetermined_sign():
    """A symbolic norm of unknown sign must not be reported as a causal type."""
    a = sp.Symbol("a")           # no positivity assumption
    result = classify_causal_vector(_minkowski(), [a, 1, 0, 0])
    assert result["causal_type"] == "undetermined"


def test_proper_time_rate_of_a_unit_timelike_vector_is_one():
    assert proper_time_rate(_minkowski(), [1, 0, 0, 0]) == 1


def test_lowering_then_raising_recovers_the_vector():
    eta = _minkowski()
    v = [2, 3, 5, 7]
    assert raise_covector(lower_vector(v, eta), eta) == sp.Matrix(v)


def test_lower_vector_flips_only_the_timelike_component():
    assert lower_vector([2, 3, 5, 7], _minkowski()) == sp.Matrix([-2, 3, 5, 7])


def test_contract_rank_02_gives_the_trace_with_the_inverse_metric():
    """The helper the standalone validator does not reach."""
    eta = _minkowski()
    # trace of the metric with its own inverse is the dimension
    assert contract_rank_02(eta, eta) == 4
    # a diagonal tensor contracts to the signed sum of its entries
    T = sp.diag(2, 3, 5, 7)
    assert contract_rank_02(T, eta) == -2 + 3 + 5 + 7
    # symmetric off-diagonal entries are counted twice, once per index pair
    S = sp.zeros(4)
    S[0, 1] = S[1, 0] = 1
    assert contract_rank_02(S, eta) == 0


def test_contract_rank_02_is_linear():
    eta = _minkowski()
    A = sp.diag(1, 2, 3, 4)
    B = sp.diag(5, 6, 7, 8)
    a, b = sp.symbols("a b")
    assert sp.simplify(
        contract_rank_02(a * A + b * B, eta)
        - (a * contract_rank_02(A, eta) + b * contract_rank_02(B, eta))
    ) == 0


def test_levi_civita_connection_is_torsion_free():
    r, th = sp.symbols("r theta", positive=True)
    # polar connection, whose non-zero symbols are symmetric in the lower pair
    Gamma = [[[sp.Integer(0)] * 2 for _ in range(2)] for _ in range(2)]
    Gamma[0][1][1] = -r
    Gamma[1][0][1] = Gamma[1][1][0] = 1 / r
    torsion = compute_torsion(Gamma, dim=2)
    assert is_torsion_free(torsion, dim=2)


def test_compute_torsion_detects_an_antisymmetric_part():
    Gamma = [[[sp.Integer(0)] * 2 for _ in range(2)] for _ in range(2)]
    Gamma[0][0][1] = 1          # no matching Gamma[0][1][0], so torsion is non-zero
    torsion = compute_torsion(Gamma, dim=2)
    assert not is_torsion_free(torsion, dim=2)
    assert sp.simplify(torsion[0][0][1] - 1) == 0
    assert sp.simplify(torsion[0][1][0] + 1) == 0


def test_inferred_stress_energy_reproduces_the_einstein_equation():
    G = sp.diag(1, 2, 3, 4)
    T = infer_stress_energy_from_einstein(G)
    residual = sp.simplify(sp.Matrix(G) - 8 * sp.pi * sp.Matrix(T))
    assert residual == sp.zeros(4)


def test_inferred_stress_energy_vanishes_in_vacuum():
    T = infer_stress_energy_from_einstein(sp.zeros(4))
    assert sp.Matrix(T) == sp.zeros(4)
