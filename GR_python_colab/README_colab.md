# GR Calculator — Google Colab Edition

Symbolic General Relativity analysis powered by **SymPy**, with optional GPU
numerical post-processing via CuPy / JAX and parallel symbolic computation
on multi-core Linux machines.

---

## Files

| File | Purpose |
|---|---|
| `GR_Colab.ipynb` | Main notebook — open this in Google Colab |
| `gr_tensors.py` | All symbolic GR computations + parallel helpers |
| `gr_latex.py` | LaTeX report assembly |
| `gr_main.py` | User configuration, pipeline, PDF compiler |
| `gr_numerics.py` | Numerical evaluation, GPU support, geodesic integration |

---

## Quick Start

### 1. Upload to Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Open `GR_Colab.ipynb` (File → Upload notebook)
3. Upload all five `.py` files to `/content/` via the Files panel or Cell 2

### 2. Run cells in order

| Cell | Action |
|---|---|
| **Cell 1** | Install LaTeX + Python packages (once per session) |
| **Cell 2** | Upload modules and detect GPU backend |
| **Cell 3** | **Edit this cell** — set your metric, coordinates, and flags |
| **Cell 4** | Run all symbolic GR computations |
| **Cell 5** | Generate PDF report (downloads to `/content/gr_report.pdf`) |
| **Cell 6** | Numerical evaluation of GR scalars on a coordinate grid |
| **Cell 7** | 2-D heat maps of GR scalars |
| **Cell 8** | Geodesic integration and trajectory plot |

---

## Changing the Metric

Edit **Cell 3** in the notebook. The pattern is:

```python
from sympy import symbols, Matrix, sin

t, r, theta, phi = symbols('t r theta phi', real=True)
coords = [t, r, theta, phi]
dim    = 4

M = symbols('M', positive=True)

grm.g_metric = Matrix([
    [...],  # row for t
    [...],  # row for r
    [...],  # row for theta
    [...],  # row for phi
])
grm.METRIC_NAME        = 'My Metric'
grm.METRIC_DESCRIPTION = 'Short description'
```

Example metrics are commented out in `gr_main.py` (Minkowski, FRW, Kerr-type, etc.).

---

## GPU Acceleration

GPU accelerates **numerical post-processing only** (cells 6–8).
Symbolic SymPy computation always runs on CPU.

To enable GPU:

1. In Colab: *Runtime → Change runtime type → T4 GPU*
2. In Cell 1, uncomment the GPU package install:
   ```
   !pip install -q cupy-cuda12x        # for CuPy (recommended)
   # OR
   !pip install -q "jax[cuda12]" diffrax  # for JAX + diffrax
   ```
3. In Cell 3, set `grm.USE_GPU = True`

The `grn.detect_backend()` call in Cell 2 confirms which backend is active.

---

## Parallel Symbolic Computation

On Colab (Linux) you can parallelise the Christoffel and Riemann symbol
computation across multiple CPU cores:

```python
grm.USE_PARALLEL    = True
grm.N_PARALLEL_JOBS = 2   # free Colab tier typically has 2 cores
```

**Note:** This uses Python's `ProcessPoolExecutor` with `fork` and only works
on Linux. Do **not** enable on Windows (use the original `GR_python` folder there).

---

## Local Run (without Colab)

The Colab fork also runs locally on Linux/macOS:

```bash
cd GR_python_colab
python gr_main.py
```

It auto-detects that `__file__` is defined and writes output next to `gr_main.py`.

---

## Dependency Chain (no circular imports)

```
gr_tensors.py   (sympy + stdlib — no project imports)
      ↓
gr_latex.py     (imports from gr_tensors)
      ↓
gr_main.py      (imports from gr_tensors and gr_latex)

gr_numerics.py  (imports numpy/cupy/jax + sympy.lambdify — standalone)
```

---

## Requirements

- Python 3.8+
- sympy
- matplotlib
- scipy (for geodesic fallback)
- Optional: cupy-cuda12x, jax, diffrax (for GPU acceleration)
- Optional: texlive-latex-extra (for PDF generation)
