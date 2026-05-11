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
<summary>Diagramas de Penrose</summary>

```python
from gr_penrose import list_penrose_spacetimes, draw_penrose_diagram

print(list_penrose_spacetimes())
fig = draw_penrose_diagram("schwarzschild", output_path="penrose_schwarzschild.pdf")
```

Resultado esperado:

- Lista de plantillas disponibles.
- Un PDF/figura cualitativa del diagrama elegido.

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
Help example checks passed: 11/11
```

Si algun ejemplo falla, eso indica que la documentacion y el codigo dejaron de
estar sincronizados.

</details>

