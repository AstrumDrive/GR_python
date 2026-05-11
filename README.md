# GR_python — Symbolic General Relativity Calculator

A full symbolic General Relativity pipeline powered by **SymPy** and **LaTeX**.  
Given any metric tensor, the engine computes every standard GR quantity and assembles a publication-quality PDF report.

Spanish guide: [GUIA_USO_ES.md](GUIA_USO_ES.md)

---

## Capabilities

| Category | Quantities computed |
|---|---|
| **Connection** | Christoffel symbols Γ^λ_{μν} |
| **Curvature** | Riemann R^λ_{ρμν}, Ricci R_{μν}, Ricci scalar R, Einstein G_{μν} |
| **Invariants** | Kretschmann scalar K = R_{μνρσ}R^{μνρσ}, Weyl tensor C_{μνρσ} |
| **Orthonormal frame** | ADM tetrad e^μ_â, frame Einstein tensor Ĝ_{âb̂}, energy conditions (NEC/WEC/SEC/DEC) |
| **Geodesics** | Symbolic geodesic equations, cyclic coordinates, conserved momenta |
| **Killing / symmetries** | Cyclic-coordinate Killing vectors, full Killing equation solver, covariant derivative ∇_μ, Lie derivative £_ξg_{μν} |
| **Carter constant** | Constants of motion for spherical and Kerr-type metrics |
| **Newman-Penrose** | NP null tetrad {l,n,m,m̄}, Weyl scalars Ψ₀–Ψ₄, frame-invariants I, J, D |
| **Petrov classification** | Algebraic type O/N/III/D/II/I, Bel-Robinson super-energy tensor |
| **Horizons** | Event horizons (zeros of g^{rr}), static-limit/ergosphere, surface gravity κ, Hawking temperature T_H, null expansion θ, apparent horizon |
| **Causal structure** | Automatic classification (Schwarzschild/Kerr/RN/de Sitter/...) + Penrose-Carter conformal diagrams |
| **Matter fields** | Klein-Gordon T_{μν} + □φ equation, Maxwell F_{μν} + T_{μν} + ∇_μF^{μν} verification, Fierz-Pauli spin-2 mass term |
| **3+1 ADM** | Lapse N, shift β^i, induced metric γ_{ij}, extrinsic curvature K_{ij}, 3D Ricci scalar R^{(3)}, Hamiltonian constraint H, momentum constraints M^i |
| **ADM mass** | M_ADM and J_ADM from asymptotic falloff |
| **Numeric geodesics** | scipy RK45 integration, lambdified Christoffel symbols, conservation diagnostics, matplotlib orbit plots |
| **PDF report** | Full LaTeX report with all quantities; auto-compiled with pdflatex when available |

---

## Built-in Metrics

| Key | Spacetime |
|---|---|
| `schwarzschild` | Exterior Schwarzschild vacuum |
| `reissner_nordstrom` | Charged static black hole (mass M, charge Q) |
| `kerr` | Rotating black hole, Boyer-Lindquist coordinates (mass M, spin a) |
| `kerr_newman` | Charged rotating black hole (M, Q, a) |
| `de_sitter_static` | Static-patch de Sitter (cosmological constant Λ) |
| `ads_schwarzschild` | Schwarzschild-AdS (M, Λ < 0) |
| `anti_de_sitter` | Global AdS with unit radius |
| `minkowski_spherical` | Flat Minkowski in spherical coordinates |
| `frw_flat` | Flat FLRW cosmology with scale factor a(t) |
| `static_spherical` | General static spherical ansatz A(r), B(r) |
| `morris_thorne_wormhole` | Traversable wormhole with Φ(r), b(r) |
| `pg_areal` | Painlevé-Gullstrand-type warp metric (areal gauge) |
| `pg_spatial_conformal` | PG-type warp metric (spatial conformal gauge) |
| `warp_doc_baseline` | Warp drive baseline metric |
| `warp_doc_variant_a` | Warp drive Variant A |
| `warp_doc_variant_b` | Warp drive Variant B |

---

## Module Structure

```
gr_calculator.py          Entry point (thin wrapper around gr_main.py)
gr_main.py                User configuration, computation flags, pipeline
gr_metric_library.py      Built-in metric registry
gr_tensors.py             Core symbolic tensor engine
gr_latex.py               LaTeX/PDF report assembler (16 sections)
gr_warp.py                Warp drive metric configurations
gr_petrov.py              Newman-Penrose formalism, Weyl scalars, Petrov type
gr_horizons.py            Horizon detection, surface gravity, causal structure
gr_geodesic_numeric.py    Numerical geodesic integration (scipy + matplotlib)
gr_matter.py              Matter field stress-energy tensors
gr_adm31.py               ADM 3+1 decomposition, constraints, ADM mass
gr_penrose.py             Penrose-Carter conformal diagrams (matplotlib)
GR_python_colab/          Google Colab notebook and companion scripts
```

---

## Requirements

```
sympy >= 1.12
scipy          (for COMPUTE_GEODESIC_NUM)
matplotlib     (for COMPUTE_PENROSE and geodesic plots)
pdflatex       (optional, for PDF compilation — MiKTeX / TeX Live)
```

Install Python dependencies:

```bash
pip install sympy scipy matplotlib
```

---

## Quick Start

### 1 — Run a built-in metric

```bash
python gr_calculator.py
```

The default metric is `schwarzschild`. Output:
- `gr_report.tex` — full LaTeX source
- `gr_report.pdf` — compiled PDF (if pdflatex is available)

### 2 — Change the metric

Open `gr_main.py`, Section 1.3:

```python
METRIC_KEY = 'kerr'          # rotating black hole
# METRIC_KEY = 'frw_flat'   # cosmology
# METRIC_KEY = 'custom'     # fill CUSTOM_METRIC_CONFIG below
```

### 3 — Use a custom metric

```python
METRIC_KEY = 'custom'

CUSTOM_METRIC_CONFIG = {
    'g_metric': Matrix([
        [-f,      0,    0,                0],
        [0,    1/f,    0,                0],
        [0,      0,  r**2,               0],
        [0,      0,    0,  r**2*sin(theta)**2],
    ]),
    'metric_name': 'My Custom Metric',
    'metric_description': 'Short description',
    'g_inv_metric': None,   # computed automatically if None
    'e_tetrad': None,       # computed automatically if None
}
```

### 4 — Enable / disable computations

Section 1.6 of `gr_main.py` controls which phases run:

```python
# Core (always runs)
COMPUTE_WEYL        = True
COMPUTE_KRETSCHMANN = True
COMPUTE_GEODESICS   = True
COMPUTE_KILLING     = True
COMPUTE_TETRAD      = True
FAST_MODE           = False   # skip heavy invariants for quick checks

# Extended modules (default off — enable as needed)
COMPUTE_PETROV      = False   # Newman-Penrose + Petrov type (needs WEYL + TETRAD)
COMPUTE_HORIZONS    = True    # horizon detection + surface gravity + causal structure
COMPUTE_KILLING_FULL= False   # full Killing equation solver (slow)
COMPUTE_CARTER      = False   # Carter constant / constants of motion
COMPUTE_MATTER      = False   # matter field T_{μν} (configure MATTER_CONFIG)
COMPUTE_ADM31       = False   # 3+1 decomposition + ADM mass (slow for complex metrics)
COMPUTE_PENROSE     = True    # Penrose-Carter diagram (instant, matplotlib)
COMPUTE_GEODESIC_NUM= False   # numerical geodesic (configure Section 1.8)
```

---

## Penrose Diagrams

Draw a standalone diagram directly from Python:

```python
from gr_penrose import draw_penrose_diagram, draw_all_penrose_diagrams

# Single diagram
fig = draw_penrose_diagram('schwarzschild', output_path='penrose_sch.pdf', show=True)

# All six spacetimes in one figure
fig = draw_all_penrose_diagrams(output_path='all_penrose.pdf', show=True)
```

Available keys: `minkowski`, `schwarzschild`, `reissner_nordstrom`, `kerr`, `de_sitter`, `anti_de_sitter`.

---

## Numerical Geodesic

Configure in `gr_main.py` Section 1.8:

```python
COMPUTE_GEODESIC_NUM = True
NUMERIC_PARAMS  = {'M': 1.0}          # numeric values for metric parameters
GEODESIC_X0     = [0.0, 10.0, sp.pi/2, 0.0]   # initial coordinates
GEODESIC_U0     = [-1.0, 0.0, 0.0, 0.1]        # initial 4-velocity
GEODESIC_LAMBDA = (0, 3000)                     # affine parameter range
GEODESIC_PLOT_PATH = 'geodesic.png'             # None = don't save
```

The integrator uses RK45 and checks conservation of the geodesic norm and Killing charges along the trajectory.

---

## Matter Fields

Configure in `gr_main.py` Section 1.7:

```python
COMPUTE_MATTER = True

# Scalar field φ(r)
from sympy import Function, symbols
phi = Function('phi')(r)
m_s = symbols('m_s', positive=True)

# Maxwell potential A_μ (Coulomb)
Q = symbols('Q', real=True)
A = [Q/r, 0, 0, 0]

MATTER_CONFIG = {
    'scalar_field':     phi,
    'scalar_mass':      m_s,
    'vector_potential': A,
    'four_current':     None,   # vacuum
    'spin2_field':      None,
    'graviton_mass':    None,
}
```

---

## Add a New Built-in Metric

1. Open `gr_metric_library.py`.
2. Add an entry inside `build_builtin_metric_library()`:

```python
"my_metric": {
    "g_metric": Matrix([...]),
    "metric_name": "My Metric",
    "metric_description": "One-sentence description",
    "g_inv_metric": None,
    "e_tetrad": None,
},
```

3. Set `METRIC_KEY = 'my_metric'` in `gr_main.py`.

---

## Google Colab

Open `GR_python_colab/GR_Colab.ipynb` in Google Colab and run the cells in order. The notebook installs all dependencies automatically.

---

## License and Citation

This project is released under the [BSD 3-Clause License](LICENSE).

If you use this software in research or publications, please cite it using [CITATION.cff](CITATION.cff) or the Zenodo DOI associated with the GitHub release.

---

## References

- Misner, Thorne, Wheeler — *Gravitation* (1973)
- Wald — *General Relativity* (1984)
- Chandrasekhar — *The Mathematical Theory of Black Holes* (1983)
- Newman & Penrose, JMP 3 (1962) 566
- Arnowitt, Deser, Misner, arXiv:gr-qc/0405109
- Regge & Teitelboim, Ann. Phys. 88 (1974) 286
