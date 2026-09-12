# Cobertura del tutorial de Relatividad General de John Baez

Fuente: [The General Relativity Tutorial](https://math.ucr.edu/home/baez/gr/gr.html),
en especial el [Long Course Outline](https://math.ucr.edu/home/baez/gr/outline2.html)
y la pagina sobre [desviacion geodesica](https://math.ucr.edu/home/baez/gr/geodesic.deviation.html).

Este documento limita el alcance a las cantidades matematicas que el tutorial
presenta y que se pueden calcular desde una metrica o una conexion. No pretende
ser una implementacion de toda la pedagogia o de los ejemplos narrativos.

| Cantidad del tutorial | API de GR_python | Estado |
| --- | --- | --- |
| Vectores tangentes y covectores | Listas o `sympy.Matrix` de componentes; `lower_vector`, `raise_covector` | Implementado |
| Producto metrico, norma y tipo causal | `metric_inner_product`, `classify_causal_vector`, `proper_time_rate` | Implementado |
| Tensores y contracciones | `contract_rank_02`, derivadas covariantes para vector, 1-forma y tensor (0,2) | Implementado |
| Conexion de Levi-Civita | `compute_christoffel` | Implementado |
| Torsion | `compute_torsion`, `is_torsion_free` | Implementado |
| Transporte paralelo | `get_parallel_transport_equations`; `integrate_parallel_transport` | Implementado |
| Geodesicas | `get_geodesic_equations`, `integrate_geodesic` | Implementado |
| Riemann | `compute_riemann` | Implementado |
| Ricci, escalar y Einstein | `compute_ricci`, `compute_ricci_scalar`, `compute_einstein` | Implementado |
| Weyl | `compute_weyl` | Implementado |
| Desviacion geodesica | `compute_geodesic_deviation_acceleration` | Implementado |
| Tensor energia-momento | `gr_matter` para campos escalar y Maxwell; `infer_stress_energy_from_einstein` para una metrica | Implementado |
| Ecuacion de Einstein | `einstein_equation_residual(G, T, kappa)` | Implementado |

## Ejemplo de desviacion geodesica

```python
from gr_tensors import compute_geodesic_deviation_acceleration

# Primero construye Riem con compute_riemann(Gamma, coords).
# u es la tangente de la geodesica y xi la separacion infinitesimal.
acceleration = compute_geodesic_deviation_acceleration(Riem, u, xi)
```

El resultado sigue la convencion interna
`A^a = R^a_{bcd} u^b xi^c u^d`, la misma forma usada en el tutorial de Baez.

## Validacion

```bash
python validate_baez_tutorial.py
```

La prueba comprueba en Minkowski los productos y tipos causales, subir/bajar
indices, conexion/torsion, transporte paralelo, geodesicas, curvatura plana y
ecuacion de Einstein. Tambien comprueba la contraccion de desviacion geodesica
con un tensor de Riemann de prueba.
