# GR_python — Guía de Uso en Español

Calculadora simbólica de Relatividad General potenciada por **SymPy** y **LaTeX**.  
A partir de cualquier tensor métrico computa todas las cantidades estándar de RG y genera un informe PDF de calidad de publicación.

---

## Capacidades

| Categoría | Cantidades calculadas |
|---|---|
| **Conexión** | Símbolos de Christoffel Γ^λ_{μν} |
| **Curvatura** | Riemann R^λ_{ρμν}, Ricci R_{μν}, escalar R, Einstein G_{μν} |
| **Invariantes** | Kretschmann K, tensor de Weyl C_{μνρσ} |
| **Marco ortonormal** | Tétrada ADM, Einstein de marco Ĝ_{âb̂}, condiciones de energía (NEC/WEC/SEC/DEC) |
| **Geodésicas** | Ecuaciones geodésicas simbólicas, coordenadas cíclicas, momentos conservados |
| **Killing / simetrías** | Vectores de Killing, derivada covariante ∇_μ, derivada de Lie £_ξg_{μν} |
| **Constante de Carter** | Constantes de movimiento para métricas esféricas y tipo Kerr |
| **Newman-Penrose** | Tétrada nula {l,n,m,m̄}, escalares de Weyl Ψ₀–Ψ₄, invariantes I, J, D |
| **Clasificación de Petrov** | Tipo O/N/III/D/II/I, tensor de Bel-Robinson |
| **Horizontes** | Horizontes de eventos, ergosfera, gravedad superficial κ, temperatura de Hawking T_H, expansión nula θ |
| **Estructura causal** | Clasificación automática + diagramas de Penrose-Carter |
| **Campos de materia** | Klein-Gordon T_{μν}, Maxwell F_{μν} + T_{μν}, campo espín-2 de Fierz-Pauli |
| **Descomposición 3+1 ADM** | Curvatura extrínseca K_{ij}, Ricci 3D R^{(3)}, vínculos hamiltoniano y de momento |
| **Masa ADM** | M_ADM y J_ADM por caída asintótica |
| **Geodésicas numéricas** | Integración RK45 (scipy), diagnósticos de conservación, gráficos matplotlib |
| **Informe PDF** | LaTeX con 16 secciones; compilación automática con pdflatex |

---

## Dónde ejecutarlo

| Entorno | Archivo | Qué hacer |
|---|---|---|
| Google Colab | `GR_python_colab/GR_Colab.ipynb` | Ejecutar las celdas en orden |
| Spyder / Python local | `gr_main.py` | Elegir `METRIC_KEY` y correr `gr_calculator.py` |
| Google Cloud / Linux | `gr_main.py` | Instalar dependencias y correr `gr_calculator.py` |

---

## Instalación

```bash
pip install sympy scipy matplotlib
```

Para generar el PDF instalar MiKTeX (Windows) o TeX Live (Linux/macOS).

---

## Uso básico paso a paso

### Sistema de ayuda interactiva

Antes de editar codigo, puedes explorar los comandos y ejemplos:

```bash
python gr_help.py
python gr_help.py topics
python gr_help.py examples
python gr_help.py example fast_minkowski_run
python gr_help.py validate
```

La guia desplegable esta en [GUIA_COMANDOS_ES.md](GUIA_COMANDOS_ES.md).

### Ejecutar una métrica incorporada

1. Abrir `gr_main.py`.
2. En la Sección 1.3 elegir la métrica:
   ```python
   METRIC_KEY = 'schwarzschild'
   ```
3. Correr `gr_calculator.py` o `gr_main.py` directamente.
4. Los archivos generados aparecen en la carpeta del proyecto:
   - `gr_report.tex`
   - `gr_report.pdf` (si pdflatex está disponible)

### Ejecutar una métrica personalizada

1. Abrir `gr_main.py` y poner:
   ```python
   METRIC_KEY = 'custom'
   ```
2. Definir los símbolos o funciones necesarias en la Sección 1.2.
3. Completar `CUSTOM_METRIC_CONFIG`:
   ```python
   CUSTOM_METRIC_CONFIG = {
       'g_metric': Matrix([...]),
       'metric_name': 'Mi Métrica',
       'metric_description': 'Descripción breve',
       'g_inv_metric': None,
       'e_tetrad': None,
   }
   ```
4. Correr el script.

---

## Métricas incorporadas

| Clave | Espacio-tiempo |
|---|---|
| `schwarzschild` | Solución exterior de Schwarzschild |
| `reissner_nordstrom` | Agujero negro estático cargado (M, Q) |
| `kerr` | Agujero negro rotante, Boyer-Lindquist (M, a) |
| `kerr_newman` | Agujero negro rotante cargado (M, Q, a) |
| `de_sitter_static` | Parche estático de de Sitter (Λ) |
| `ads_schwarzschild` | Schwarzschild-AdS (M, Λ < 0) |
| `anti_de_sitter` | AdS global con radio unitario |
| `minkowski_spherical` | Minkowski en coordenadas esféricas |
| `frw_flat` | FLRW plano con factor de escala a(t) |
| `static_spherical` | Ansatz esférico estático genérico A(r), B(r) |
| `morris_thorne_wormhole` | Agujero de gusano traversable Φ(r), b(r) |
| `pg_areal` | Métrica warp tipo PG (gauge areal) |
| `pg_spatial_conformal` | Métrica warp PG (gauge conforme espacial) |
| `warp_doc_baseline` | Métrica warp de referencia |
| `warp_doc_variant_a` | Variante A de warp |
| `warp_doc_variant_b` | Variante B de warp |
| `warp_doc_variant_b_alpha` | Variante VdB/PG espacial completa con lapse genérico α(r) |

---

## Flags de cómputo (Sección 1.6 de gr_main.py)

```python
# Módulos base
COMPUTE_WEYL        = True    # Tensor de Weyl
COMPUTE_KRETSCHMANN = True    # Escalar de Kretschmann
COMPUTE_GEODESICS   = True    # Ecuaciones geodésicas simbólicas
COMPUTE_KILLING     = True    # Coordenadas cíclicas / vectores de Killing
COMPUTE_TETRAD      = True    # Tétrada ortonormal (ADM)
FAST_MODE           = False   # Omite Weyl y Kretschmann para correr rápido

# Módulos extendidos (apagados por defecto)
COMPUTE_PETROV      = False   # Newman-Penrose + tipo Petrov (requiere WEYL + TETRAD)
COMPUTE_HORIZONS    = True    # Horizontes, κ, T_Hawking, estructura causal
COMPUTE_KILLING_FULL= False   # Ecuación de Killing completa (lenta)
COMPUTE_CARTER      = False   # Constante de Carter
COMPUTE_MATTER      = False   # Campos de materia (configurar MATTER_CONFIG)
COMPUTE_ADM31       = False   # Descomposición 3+1 + masa ADM (lenta)
COMPUTE_PENROSE     = True    # Diagrama de Penrose-Carter (instantáneo)
COMPUTE_GEODESIC_NUM= False   # Geodésica numérica (configurar Sección 1.8)
```

---

## Diagramas de Penrose

Desde Python directamente:

```python
from gr_penrose import draw_penrose_diagram, draw_all_penrose_diagrams

# Un diagrama
fig = draw_penrose_diagram('kerr', output_path='penrose_kerr.pdf', show=True)

# Los seis en un panel
fig = draw_all_penrose_diagrams(output_path='todos_penrose.pdf', show=True)
```

Claves disponibles: `minkowski`, `schwarzschild`, `reissner_nordstrom`, `kerr`, `de_sitter`, `anti_de_sitter`.

---

## Geodésica numérica

Configurar en la Sección 1.8 de `gr_main.py`:

```python
COMPUTE_GEODESIC_NUM = True
NUMERIC_PARAMS  = {'M': 1.0}                       # valores numéricos de parámetros
GEODESIC_X0     = [0.0, 10.0, sp.pi/2, 0.0]        # posición inicial [t, r, θ, φ]
GEODESIC_U0     = [-1.0, 0.0, 0.0, 0.1]            # 4-velocidad inicial
GEODESIC_LAMBDA = (0, 3000)                         # rango del parámetro afín
GEODESIC_PLOT_PATH = 'geodesica.png'                # None = no guardar
```

---

## Campos de materia

Configurar en la Sección 1.7 de `gr_main.py`:

```python
COMPUTE_MATTER = True

from sympy import Function, symbols
phi = Function('phi')(r)          # campo escalar Klein-Gordon
m_s = symbols('m_s', positive=True)
Q   = symbols('Q', real=True)
A   = [Q/r, 0, 0, 0]             # potencial de Maxwell (Coulomb)

MATTER_CONFIG = {
    'scalar_field':     phi,
    'scalar_mass':      m_s,
    'vector_potential': A,
    'four_current':     None,
    'spin2_field':      None,
    'graviton_mass':    None,
}
```

---

## Agregar una métrica incorporada nueva

1. Abrir `gr_metric_library.py`.
2. Agregar una entrada dentro de `build_builtin_metric_library()`:

```python
"mi_metrica": {
    "g_metric": Matrix([...]),
    "metric_name": "Mi Métrica",
    "metric_description": "Descripción de una línea",
    "g_inv_metric": None,
    "e_tetrad": None,
},
```

3. En `gr_main.py` usar:
   ```python
   METRIC_KEY = 'mi_metrica'
   ```

---

## Estructura del proyecto

```
gr_calculator.py          Punto de entrada
gr_main.py                Configuración del usuario, flags, pipeline
gr_metric_library.py      Registro de métricas incorporadas
gr_tensors.py             Motor de cómputo tensorial simbólico
gr_latex.py               Ensamblador del informe LaTeX/PDF (16 secciones)
gr_warp.py                Configuraciones de métricas warp
gr_petrov.py              Formalismo Newman-Penrose, escalares de Weyl, Petrov
gr_horizons.py            Horizontes, gravedad superficial, estructura causal
gr_geodesic_numeric.py    Integración geodésica numérica (scipy + matplotlib)
gr_matter.py              Tensores de energía-impulso de campos de materia
gr_adm31.py               Descomposición ADM 3+1, vínculos, masa ADM
gr_penrose.py             Diagramas de Penrose-Carter (matplotlib)
GR_python_colab/          Notebook de Google Colab y scripts de acompañamiento
```

---

## Licencia y cita

Este proyecto está publicado bajo la [Licencia BSD 3-Clause](LICENSE).

Si usas este software en investigación o publicaciones, por favor citarlo usando [CITATION.cff](CITATION.cff) o el DOI de Zenodo asociado al release de GitHub.

---

## Referencias

- Misner, Thorne, Wheeler — *Gravitation* (1973)
- Wald — *General Relativity* (1984)
- Chandrasekhar — *The Mathematical Theory of Black Holes* (1983)
- Newman & Penrose, JMP 3 (1962) 566
- Arnowitt, Deser, Misner, arXiv:gr-qc/0405109
- Regge & Teitelboim, Ann. Phys. 88 (1974) 286
