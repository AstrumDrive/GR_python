import pytest
import sympy as sp

# grsuite_contracts is an optional dependency of grsuite_adapter.  Skip rather
# than fail collection where it is absent, otherwise the whole suite aborts on
# any environment that does not install it (the ASTRUM pkgs env, for one).
pytest.importorskip(
    "grsuite_contracts",
    reason="optional dependency grsuite-contracts is not installed",
)

from grsuite_adapter import (  # noqa: E402  (import after the skip guard)
    bundles_from_run,
    charge_bundles_from_results,
    geometry_bundle_from_results,
    metric_bundle_from_config,
    source_bundles_from_results,
)
from grsuite_contracts import to_dict  # noqa: E402


def _metric():
    t, x = sp.symbols("t x")
    return [t, x], {
        "metric_name": "flat-2d",
        "metric_description": "test metric",
        "g_metric": sp.diag(-1, 1),
        "g_inv_metric": sp.diag(-1, 1),
    }


def test_metric_adapter_preserves_symbolic_payload_and_conventions():
    coords, config = _metric()
    bundle = metric_bundle_from_config(config, coords)

    assert bundle.name == "flat-2d"
    assert bundle.coordinates == ("t", "x")
    assert bundle.metric_covariant == sp.diag(-1, 1)
    assert bundle.provenance["producer"] == "GR_python"


def test_geometry_adapter_maps_only_existing_results():
    coords, config = _metric()
    metric = metric_bundle_from_config(config, coords)
    geometry = geometry_bundle_from_results(
        metric,
        {"Gamma": [[[0]]], "R_riem": [[[[0]]]], "Ric": [[0]], "R_scalar": 0, "G": [[0]], "petrov": "O"},
    )

    assert geometry.status == "computed"
    assert set(geometry.curvature) == {"christoffel", "riemann", "ricci", "ricci_scalar", "einstein"}
    assert geometry.symmetries == {"petrov": "O"}


def test_source_and_charge_adapters_preserve_conditionality():
    tensor = sp.diag(1, 2)
    sources = source_bundles_from_results(
        {"matter": {"scalar": {"T_cov": tensor}, "scalar_eom": {"residual": 0}}}
    )
    charges = charge_bundles_from_results(
        {"adm31": {"adm_mass": {"M_ADM": sp.Symbol("M"), "J_ADM": None}}}
    )

    assert sources["scalar"].tensor_covariant == tensor
    assert sources["scalar"].conservation["field_equations"] == {"residual": 0}
    assert charges["M_ADM"].status == "computed"
    assert charges["J_ADM"].status == "conditional"


def test_bundles_from_run_has_no_duplicate_computation():
    coords, config = _metric()
    bundles = bundles_from_run(
        config,
        coords,
        {"Gamma": {}, "R_riem": {}, "Ric": {}, "R_scalar": 0, "G": {}},
    )

    assert set(bundles) == {"metric", "geometry", "sources", "charges"}
    assert to_dict(bundles["metric"])["provenance"]["producer"] == "GR_python"
