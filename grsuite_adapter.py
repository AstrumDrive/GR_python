"""Adapters from GR_python result dictionaries to the neutral GR-suite contracts.

This module only translates already-computed payloads.  It deliberately does
not calculate an inverse metric, curvature, ADM charges or matter tensors;
those remain the responsibility of the existing GR_python pipeline.

The optional dependency is :mod:`grsuite_contracts`.  Install that package
before importing this adapter when using GR_python as a producer in the suite.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

try:
    from grsuite_contracts import (
        ChargeBundle,
        ConventionBundle,
        FoliationBundle,
        GeometryBundle,
        MetricBundle,
        SourceBundle,
    )
except ImportError as exc:  # pragma: no cover - exercised by import users
    raise ImportError(
        "grsuite_adapter requires the optional 'grsuite-contracts' package"
    ) from exc


_CURVATURE_KEYS = {
    "Gamma": "christoffel",
    "R_riem": "riemann",
    "Ric": "ricci",
    "R_scalar": "ricci_scalar",
    "G": "einstein",
    "K": "kretschmann",
    "Weyl": "weyl",
}
_CAUSAL_KEYS = {"horizons", "bianchi", "trace_check", "geodesics", "geodesic_num"}
_SYMMETRY_KEYS = {"killing", "killing_full", "conserved_full", "carter", "petrov"}


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _coordinate_names(coords: Sequence[Any]) -> tuple[str, ...]:
    names = tuple(str(item) for item in coords)
    if not names or any(not item.strip() for item in names):
        raise ValueError("coords must contain at least one non-empty coordinate")
    return names


def _conventions(
    coords: tuple[str, ...],
    conventions: ConventionBundle | Mapping[str, Any] | None,
) -> ConventionBundle:
    if conventions is None:
        return ConventionBundle(coordinates=coords, index_order=f"({','.join(coords)})")
    if isinstance(conventions, ConventionBundle):
        if tuple(conventions.coordinates) != coords:
            raise ValueError("convention coordinates do not match coords")
        return conventions
    values = dict(_require_mapping(conventions, "conventions"))
    values["coordinates"] = coords
    values.setdefault("index_order", f"({','.join(coords)})")
    return ConventionBundle(**values)


def _provenance(base: Mapping[str, Any] | None, metric_config: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(base or {})
    values.setdefault("producer", "GR_python")
    values.setdefault("metric_description", metric_config.get("metric_description", ""))
    values.setdefault("source_api", "gr_metric_library.select_metric/run_computations")
    if metric_config.get("e_tetrad") is not None:
        values.setdefault("e_tetrad_present", True)
    return values


def _foliation_from_config(metric_config: Mapping[str, Any]) -> FoliationBundle | None:
    supplied = metric_config.get("foliation")
    if supplied is not None:
        values = dict(_require_mapping(supplied, "metric_config['foliation']"))
    else:
        aliases = {
            "lapse": ("lapse", "N"),
            "shift": ("shift", "beta_contra"),
            "spatial_metric": ("spatial_metric", "gamma_cov"),
            "extrinsic_curvature": ("extrinsic_curvature", "K_cov"),
        }
        values = {
            target: metric_config[source]
            for target, sources in aliases.items()
            for source in sources
            if source in metric_config
        }
    if not values:
        return None
    values.setdefault("time_coordinate", "t")
    values.setdefault(
        "status",
        "computed"
        if all(values.get(name) is not None for name in ("lapse", "shift", "spatial_metric", "extrinsic_curvature"))
        else "conditional",
    )
    return FoliationBundle(**values)


def metric_bundle_from_config(
    metric_config: Mapping[str, Any],
    coords: Sequence[Any],
    *,
    conventions: ConventionBundle | Mapping[str, Any] | None = None,
    representation: str = "symbolic",
    regularity: str = "unspecified",
    status: str = "computed",
    provenance: Mapping[str, Any] | None = None,
) -> MetricBundle:
    """Wrap a validated ``GR_python`` metric configuration in ``MetricBundle``."""

    config = _require_mapping(metric_config, "metric_config")
    names = _coordinate_names(coords)
    metric = config.get("g_metric", config.get("g"))
    if metric is None:
        raise KeyError("metric_config must provide 'g_metric' (or 'g')")
    name = config.get("metric_name", config.get("name"))
    if not isinstance(name, str) or not name.strip():
        raise KeyError("metric_config must provide a non-empty 'metric_name'")
    return MetricBundle(
        name=name,
        dimension=len(names),
        conventions=_conventions(names, conventions),
        representation=representation,
        metric_covariant=metric,
        metric_contravariant=config.get("g_inv_metric", config.get("ginv")),
        coordinates=names,
        foliation=_foliation_from_config(config),
        regularity=regularity,
        status=status,
        provenance=_provenance(provenance, config),
    )


def geometry_bundle_from_results(
    metric: MetricBundle,
    results: Mapping[str, Any],
    *,
    status: str | None = None,
) -> GeometryBundle:
    """Translate the existing ``run_computations`` dictionary without recomputing it."""

    data = _require_mapping(results, "results")
    curvature = {
        target: data[source]
        for source, target in _CURVATURE_KEYS.items()
        if source in data and data[source] is not None
    }
    causal = {key: data[key] for key in _CAUSAL_KEYS if key in data and data[key] is not None}
    symmetries = {key: data[key] for key in _SYMMETRY_KEYS if key in data and data[key] is not None}
    required = {"christoffel", "riemann", "ricci", "ricci_scalar", "einstein"}
    inferred_status = "computed" if required.issubset(curvature) else "conditional"
    return GeometryBundle(
        metric=metric,
        curvature=curvature,
        causal=causal,
        symmetries=symmetries,
        status=status or inferred_status,
    )


def source_bundles_from_results(
    results: Mapping[str, Any],
    *,
    frame: str = "coordinate",
    source_prefix: str = "GR_python",
) -> dict[str, SourceBundle]:
    """Wrap matter tensors already produced by ``compute_matter_content``."""

    data = _require_mapping(results, "results")
    matter = data.get("matter")
    if matter is None:
        return {}
    matter = _require_mapping(matter, "results['matter']")
    bundles: dict[str, SourceBundle] = {}
    for field_name, payload in matter.items():
        if not isinstance(payload, Mapping) or payload.get("T_cov") is None:
            continue
        bundles[str(field_name)] = SourceBundle(
            source_name=f"{source_prefix}:{field_name}",
            tensor_covariant=payload["T_cov"],
            dimension=payload["T_cov"].shape[0] if hasattr(payload["T_cov"], "shape") else 4,
            frame=frame,
            conservation={"field_equations": matter.get(f"{field_name}_eom")},
            provenance={"producer": "GR_python", "matter_key": str(field_name)},
        )
    return bundles


def charge_bundles_from_results(
    results: Mapping[str, Any],
    *,
    units: str = "G=c=1",
) -> dict[str, ChargeBundle]:
    """Translate ADM outputs when the ADM phase was explicitly requested."""

    data = _require_mapping(results, "results")
    adm = data.get("adm31")
    if not isinstance(adm, Mapping) or not isinstance(adm.get("adm_mass"), Mapping):
        return {}
    values = adm["adm_mass"]
    bundles: dict[str, ChargeBundle] = {}
    specs = {
        "M_ADM": ("ADM_mass", "ADM mass"),
        "J_ADM": ("ADM_angular_momentum", "ADM angular momentum"),
    }
    for key, (kind, label) in specs.items():
        value = values.get(key)
        bundles[key] = ChargeBundle(
            name=label,
            charge_kind=kind,
            value=value,
            units=units,
            method="GR_python.gr_adm31.compute_adm_mass",
            status="computed" if value is not None else "conditional",
            boundary_data={"source_key": key, "asymptotic": True},
            provenance={"producer": "GR_python", "adm_phase": True},
        )
    return bundles


def bundles_from_run(
    metric_config: Mapping[str, Any],
    coords: Sequence[Any],
    results: Mapping[str, Any],
    **metric_kwargs: Any,
) -> dict[str, Any]:
    """Build the suite payloads from one completed GR_python run."""

    metric = metric_bundle_from_config(metric_config, coords, **metric_kwargs)
    return {
        "metric": metric,
        "geometry": geometry_bundle_from_results(metric, results),
        "sources": source_bundles_from_results(results),
        "charges": charge_bundles_from_results(results),
    }


__all__ = [
    "bundles_from_run",
    "charge_bundles_from_results",
    "geometry_bundle_from_results",
    "metric_bundle_from_config",
    "source_bundles_from_results",
]
