# Interoperabilidad con la suite GR

`grsuite_adapter.py` convierte los resultados ya calculados por `GR_python`
en contratos de `grsuite-contracts`. Es una capa de traducción, no un segundo
motor de geometría.

```python
from gr_metric_library import select_metric
from gr_main import run_computations
from grsuite_adapter import bundles_from_run

coords = [t, r, theta, phi]
config = select_metric("schwarzschild", coords, parameter_context={"M": M})
results = run_computations(config["g_metric"], coords, 4,
                           g_inv_metric=config["g_inv_metric"])
payloads = bundles_from_run(config, coords, results)
```

El adaptador expone cuatro salidas: `metric`, `geometry`, `sources` y
`charges`. Las dos últimas solo contienen datos cuando las fases de materia o
ADM fueron solicitadas explícitamente. Un resultado ausente se conserva como
`conditional` o se omite; no se presenta como un cálculo realizado.

La dependencia es opcional para mantener el uso independiente de
`GR_python`:

```text
pip install -e ..\grsuite_contracts --no-deps
```

La validación de esta integración está en
`astra_jobs/verify_grsuite_adapter.py` y en `tests/test_grsuite_adapter.py`.
