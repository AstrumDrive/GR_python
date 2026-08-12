"""Small deterministic gate for the GR_python -> grsuite-contracts adapter."""

import sympy as sp

from grsuite_adapter import (
    bundles_from_run,
    charge_bundles_from_results,
    geometry_bundle_from_results,
    metric_bundle_from_config,
    source_bundles_from_results,
)
from grsuite_contracts import to_dict


def check(name, condition):
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")
    return bool(condition)


t, x = sp.symbols("t x")
coords = [t, x]
config = {
    "metric_name": "flat-2d",
    "metric_description": "ASTRA adapter fixture",
    "g_metric": sp.diag(-1, 1),
    "g_inv_metric": sp.diag(-1, 1),
}
metric = metric_bundle_from_config(config, coords)
geometry = geometry_bundle_from_results(
    metric,
    {"Gamma": {}, "R_riem": {}, "Ric": {}, "R_scalar": 0, "G": {}, "petrov": "O"},
)
sources = source_bundles_from_results({"matter": {"scalar": {"T_cov": sp.diag(1, 2)}}})
charges = charge_bundles_from_results(
    {"adm31": {"adm_mass": {"M_ADM": sp.Symbol("M"), "J_ADM": None}}}
)
bundles = bundles_from_run(config, coords, {"Gamma": {}, "R_riem": {}, "Ric": {}, "R_scalar": 0, "G": {}})

checks = [
    check("metric_coordinates", metric.coordinates == ("t", "x")),
    check("metric_provenance", metric.provenance["producer"] == "GR_python"),
    check("geometry_core", geometry.status == "computed"),
    check("geometry_symmetry", geometry.symmetries["petrov"] == "O"),
    check("source_tensor", sources["scalar"].tensor_covariant.shape == (2, 2)),
    check("charge_statuses", charges["M_ADM"].status == "computed" and charges["J_ADM"].status == "conditional"),
    check("run_bundle_keys", set(bundles) == {"metric", "geometry", "sources", "charges"}),
    check("json_safe", to_dict(metric)["metric_covariant"] == [["-1", "0"], ["0", "1"]]),
]

print(f"CHECKS={len(checks)} FAILURES={checks.count(False)}")
print(f"VERDICT: {'PASS' if all(checks) else 'FAIL'}")
