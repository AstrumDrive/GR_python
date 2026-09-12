# GR_python - Guia de comandos y ejemplos

Esta guia es la puerta de entrada para usuarios nuevos. La idea es que cada
"comando" tenga tres cosas: que hace, como se usa y que resultado esperar.

Desde terminal, Spyder o Google Cloud:

```bash
python gr_help.py
python gr_help.py topics
python gr_help.py examples
python gr_help.py example fast_minkowski_run
python gr_help.py validate
```

Desde Colab o Spyder:

```python
from gr_help import gr_help, validate_examples

gr_help()
gr_help("metrics")
gr_help("example:fast_minkowski_run")
validate_examples()
```

## Secciones desplegables

<details>
<summary>Inicio rapido</summary>

Usa este flujo cuando solo quieres comprobar que el entorno funciona:

```bash
python gr_calculator.py
```

Por defecto corre Schwarzschild desde `gr_main.py`. Para una primera prueba
rapida, pon:

```python
METRIC_KEY = "schwarzschild"
FAST_MODE = True
COMPUTE_TETRAD = True
```

Resultado esperado:

- `R_scalar = 0` para Schwarzschild.
- La tetrada se verifica si `COMPUTE_TETRAD = True`.
- Se escriben `gr_report.tex` y, si existe `pdflatex`, `gr_report.pdf`.

</details>

<details>
<summary>Listar y elegir metricas</summary>

```python
from gr_metric_library import list_builtin_metric_keys, select_metric
from sympy import symbols

t, r, theta, phi = symbols("t r theta phi", real=True)
coords = [t, r, theta, phi]

print(list_builtin_metric_keys())
cfg = select_metric("warp_doc_variant_b_alpha", coords)
print(cfg["metric_name"])
```

Resultado esperado:

- Una lista de claves como `schwarzschild`, `kerr`, `warp_doc_variant_a`,
  `warp_doc_variant_b` y `warp_doc_variant_b_alpha`.
- Un diccionario con `g_metric`, `g_inv_metric`, `e_tetrad` y metadatos.

</details>

<details>
<summary>Agregar una metrica custom</summary>

En `gr_main.py`:

```python
METRIC_KEY = "custom"

CUSTOM_METRIC_CONFIG = {
    "g_metric": Matrix([
        [-f, 0, 0, 0],
        [0, 1/f, 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2*sin(theta)**2],
    ]),
    "metric_name": "Mi metrica",
    "metric_description": "Descripcion corta del elemento de linea",
    "g_inv_metric": None,
    "e_tetrad": None,
}
```

Resultado esperado:

- Si `g_inv_metric = None`, GR_python calcula la inversa.
- Si `e_tetrad = None` y `COMPUTE_TETRAD = True`, GR_python construye una
  tetrada automatica.

</details>

<details>
<summary>Tetrada y carta local</summary>

```python
COMPUTE_TETRAD = True
```

Convencion usada:

- Metricas diagonales: tetrada estatica alineada con coordenadas.
- Metricas con shift `g_{0i}`: tetrada ADM/Euleriana adaptada a las hojas
  `t = const`.
- Tetrada manual: `e_tetrad` debe ser una `Matrix`, no `True`.

Resultado esperado:

- `tetrad_method` indica el metodo usado.
- `tetrad_verified = True` confirma
  `g_{mu nu} e^mu_a e^nu_b = eta_ab`.

</details>

<details>
<summary>Pipeline simbolico rapido</summary>

```python
import sympy as sp
import gr_main as gm
from gr_metric_library import select_metric

t, r, theta, phi = sp.symbols("t r theta phi", real=True)
coords = [t, r, theta, phi]
cfg = select_metric("minkowski_spherical", coords)

results = gm.run_computations(
    cfg["g_metric"], coords, 4,
    compute_weyl_flag=False,
    compute_kretschmann_flag=False,
    compute_geodesics_flag=False,
    compute_killing_flag=False,
    compute_tetrad_flag=True,
    fast_mode=True,
    compute_horizons_flag=False,
    compute_penrose_flag=False,
)

print(results["R_scalar"])
print(results["tetrad_method"], results["tetrad_verified"])
```

Resultado esperado:

- `R_scalar = 0`.
- `tetrad_method = diagonal`.
- `tetrad_verified = True`.

</details>

<details>
<summary>Horizontes</summary>

```python
import sympy as sp
from gr_metric_library import select_metric
from gr_horizons import find_horizons

t, r, theta, phi = sp.symbols("t r theta phi", real=True)
M = sp.symbols("M", positive=True)
coords = [t, r, theta, phi]

cfg = select_metric("schwarzschild", coords, {"M": M})
h = find_horizons(cfg["g_metric"], cfg["g_metric"].inv(), coords)
print(h["horizon_roots"])
```

Resultado esperado:

- Para Schwarzschild: `[2*M]`.

</details>

<details>
<summary>Campos de materia</summary>

```python
import sympy as sp
from gr_metric_library import select_metric
from gr_matter import compute_scalar_stress_energy

t, r, theta, phi = sp.symbols("t r theta phi", real=True)
coords = [t, r, theta, phi]
cfg = select_metric("minkowski_spherical", coords)

matter = compute_scalar_stress_energy(
    sp.Integer(0), cfg["g_metric"], cfg["g_metric"].inv(), coords
)
print(matter["T_cov"])
```

Resultado esperado:

- El campo escalar nulo da tensor energia-impulso nulo.

</details>

<details>
<summary>Warp / VdB con lapse generico</summary>

```python
import sympy as sp
from gr_warp import document_vdb_alpha_formulas

r = sp.symbols("r", positive=True)
alpha = sp.Function("alpha", positive=True)(r)
B = sp.Function("B", positive=True)(r)
beta = sp.Function("beta")(r)

formulas = document_vdb_alpha_formulas(r, alpha, B, beta)
print(formulas["alpha_magic"])
print(formulas["j_r"])
```

Resultado esperado:

- `alpha_magic = 1 + r B'/B`.
- `j_r` usa el signo de GR_python:
  `8*pi*j_hat{r} = -(2 beta/(alpha B)) V_tilde`.

</details>

<details>
<summary>Petrov / Newman-Penrose</summary>

```python
import sympy as sp
from gr_metric_library import select_metric
from gr_tensors import (
    compute_christoffel, compute_riemann, compute_ricci,
    compute_ricci_scalar, compute_weyl, compute_tetrad_adm,
)
from gr_petrov import build_np_tetrad, compute_weyl_scalars, classify_petrov

t, r, theta, phi = sp.symbols("t r theta phi", real=True)
M = sp.symbols("M", positive=True)
coords = [t, r, theta, phi]
cfg = select_metric("schwarzschild", coords, {"M": M})
g = cfg["g_metric"]
ginv = cfg["g_inv_metric"] or g.inv()

Gamma = compute_christoffel(g, ginv, coords)
Riem = compute_riemann(Gamma, coords)
Ric = compute_ricci(Riem)
R = compute_ricci_scalar(Ric, ginv)
C = compute_weyl(Riem, Ric, g, ginv, R)
e_contra, *_ = compute_tetrad_adm(g, ginv, coords)
np_tetrad = build_np_tetrad(e_contra)
psi = compute_weyl_scalars(C, np_tetrad, g)
print(psi["Psi2"])
print(classify_petrov(psi)["type"])
```

Resultado esperado:

- `Psi2 = -M/r**3`.
- Tipo de Petrov `D`.

</details>

<details>
<summary>Horizontes de Kerr</summary>

```python
import sympy as sp
from gr_metric_library import select_metric
from gr_horizons import find_horizons

t, r, theta, phi = sp.symbols("t r theta phi", real=True)
M, a = sp.symbols("M a", positive=True)
coords = [t, r, theta, phi]
cfg = select_metric("kerr", coords, {"M": M, "a": a})
g = cfg["g_metric"]
ginv = cfg["g_inv_metric"] or g.inv()
h = find_horizons(g, ginv, coords)
print(h["horizon_roots"])
print(h["static_limit_roots"])
```

Resultado esperado:

- `r_+ = M + sqrt(M**2 - a**2)`.
- `r_- = M - sqrt(M**2 - a**2)`.
- En el plano ecuatorial, limite estatico `[2*M]`.

</details>

<details>
<summary>Geodesica numerica corta</summary>

```python
import math
import sympy as sp
from gr_metric_library import select_metric
from gr_tensors import compute_christoffel
from gr_geodesic_numeric import (
    lambdify_christoffel, lambdify_metric,
    integrate_geodesic, check_conserved,
)

t, r, theta, phi = sp.symbols("t r theta phi", real=True)
M = sp.symbols("M", positive=True)
coords = [t, r, theta, phi]
cfg = select_metric("schwarzschild", coords, {"M": M})
g = cfg["g_metric"]
ginv = cfg["g_inv_metric"] or g.inv()
Gamma = compute_christoffel(g, ginv, coords)

Gamma_num, remaining = lambdify_christoffel(Gamma, coords, parameter_subs={M: 1.0})
g_num, ginv_num = lambdify_metric(g, ginv, coords, parameter_subs={M: 1.0})

r0 = 10.0
u_t = 1.0 / math.sqrt(1.0 - 3.0 / r0)
u_phi = math.sqrt(1.0 / r0**3) * u_t
sol = integrate_geodesic(
    Gamma_num, [0.0, r0, math.pi/2, 0.0], [u_t, 0.0, 0.0, u_phi],
    (0.0, 50.0), rtol=1e-8, atol=1e-10, max_step=1.0,
)
diag = check_conserved(sol, g_num, ginv_num, killing_indices=[0, 3])
print(sol.success)
print(diag["norm_drift"])
```

Resultado esperado:

- Integracion exitosa.
- Deriva de la norma muy pequena.

</details>

<details>
<summary>Maxwell / Reissner-Nordstrom</summary>

```python
import sympy as sp
from gr_metric_library import select_metric
from gr_matter import build_faraday_tensor, compute_maxwell_stress_energy

t, r, theta, phi = sp.symbols("t r theta phi", real=True)
Q = sp.symbols("Q", real=True)
coords = [t, r, theta, phi]
cfg = select_metric("reissner_nordstrom", coords, {"Q": Q})
g = cfg["g_metric"]
ginv = cfg["g_inv_metric"] or g.inv()
F = build_faraday_tensor([Q/r, 0, 0, 0], coords)
T = compute_maxwell_stress_energy(F, g, ginv, coords)
print(F[0, 1], F[1, 0])
print(T["T_trace"])
```

Resultado esperado:

- `F_01 = Q/r**2`.
- `F_10 = -Q/r**2`.
- La traza de Maxwell es cero en 4D.

</details>

<details>
<summary>ADM 3+1 de FLRW plano</summary>

```python
import sympy as sp
from gr_metric_library import select_metric
from gr_tensors import compute_christoffel
from gr_adm31 import extract_adm_variables, compute_extrinsic_curvature

t, r, theta, phi = sp.symbols("t r theta phi", real=True)
coords = [t, r, theta, phi]
cfg = select_metric("frw_flat", coords)
g = cfg["g_metric"]
ginv = cfg["g_inv_metric"] or g.inv()
Gamma = compute_christoffel(g, ginv, coords)
adm = extract_adm_variables(g, ginv, coords)
K = compute_extrinsic_curvature(g, ginv, Gamma, coords, adm)
print(adm["N"])
print(adm["beta_contra"])
print(sp.simplify(K["K_trace"]))
```

Resultado esperado:

- `N = 1`.
- `beta^i = [0, 0, 0]`.
- Con la convencion ADM de GR_python, `K = -3*a'(t)/a(t)`.

</details>

<details>
<summary>Diagramas de Penrose</summary>

```python
from gr_penrose import (
    list_penrose_spacetimes,
    draw_penrose_diagram,
    draw_all_penrose_diagrams,
)

print(list_penrose_spacetimes())
fig = draw_penrose_diagram("schwarzschild", output_path="penrose_schwarzschild.pdf")
panel = draw_all_penrose_diagrams(output_path="all_penrose.pdf", show=False)
```

Resultado esperado:

- Lista de plantillas disponibles.
- Un PDF/figura cualitativa del diagrama elegido.
- Un panel 2x3 con las seis plantillas principales.

</details>

<details>
<summary>Validar ejemplos de esta guia</summary>

```bash
python validate_help_examples.py
```

o:

```bash
python gr_help.py validate
```

Resultado esperado:

```text
Help example checks passed: N/N
```

Si algun ejemplo falla, eso indica que la documentacion y el codigo dejaron de
estar sincronizados.

</details>

<details>
<summary>Cantidades centrales del tutorial de Baez</summary>

Este bloque cubre producto metrico, tipo causal, subir/bajar indices, torsion,
transporte paralelo, desviacion geodesica y la ecuacion de Einstein. Para el
mapa completo, abre `BAEZ_TUTORIAL_COVERAGE.md`.

```bash
python validate_baez_tutorial.py
```

Ejemplo simbolico minimo:

```python
import sympy as sp
from gr_tensors import (
    metric_inner_product, classify_causal_vector, lower_vector,
    raise_covector,
)

eta = sp.diag(-1, 1, 1, 1)
u = sp.Matrix([1, 0, 0, 0])
u_low = lower_vector(u, eta)

print(metric_inner_product(eta, u, u))
print(classify_causal_vector(eta, u))
print(u_low)
print(raise_covector(u_low, eta))
```

Resultado esperado:

- Norma `-1` y tipo `timelike` con la convencion `(-,+,+,+)`.
- `u_low = [-1, 0, 0, 0]`.
- Al volver a subir el indice se recupera `u`.

</details>
