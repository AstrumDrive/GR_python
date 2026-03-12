# General Relativity Symbolic Calculator

A symbolic General Relativity calculator and report generator powered by SymPy and LaTeX.

Spanish quick guide: [GUIA_USO_ES.md](C:/Users/Nelson/Downloads/GR_python/GUIA_USO_ES.md)

## What It Does

Starting from a metric, the project computes:

- Christoffel symbols
- Riemann curvature tensor
- Ricci tensor and Ricci scalar
- Einstein tensor
- Curvature invariants such as Kretschmann and Weyl
- Orthonormal-frame stress-energy quantities
- Energy conditions
- Geodesic equations
- Killing-coordinate checks
- Bianchi and trace consistency checks

It also generates a LaTeX report and, when `pdflatex` is available, a PDF report.

## Requirements

- Python 3.x
- `sympy`
- a LaTeX distribution with `pdflatex` in PATH if you want PDF generation

## Local Use Step by Step (Spyder or similar)

### First-time setup

1. Open the folder in Spyder or your preferred Python IDE.
2. Make sure `sympy` is installed in the same Python environment used by Spyder.
3. If you want the PDF directly, install a LaTeX distribution such as MiKTeX and ensure `pdflatex` is in PATH.

### Run a built-in metric

1. Open [gr_main.py](C:/Users/Nelson/Downloads/GR_python/gr_main.py).
2. In Section 1, leave the coordinates as they are unless you really want a different chart.
3. Choose a built-in metric with one line, for example:
   ```python
   METRIC_KEY = 'schwarzschild'
   ```
4. Run [gr_calculator.py](C:/Users/Nelson/Downloads/GR_python/gr_calculator.py) or run [gr_main.py](C:/Users/Nelson/Downloads/GR_python/gr_main.py) directly.
5. Check the generated files in the project folder:
   - `gr_report.tex`
   - `gr_report.pdf` if `pdflatex` is available

### Run a custom metric

1. Open [gr_main.py](C:/Users/Nelson/Downloads/GR_python/gr_main.py).
2. Set:
   ```python
   METRIC_KEY = 'custom'
   ```
3. In Section 1.2, define any extra parameters or symbolic functions your metric needs.
4. Fill `CUSTOM_METRIC_CONFIG`.
5. Run the script.

Two ready-to-adapt examples are already included inside [gr_main.py](C:/Users/Nelson/Downloads/GR_python/gr_main.py):

- a diagonal custom metric with a radial function `alpha(r)`
- a metric with a `dt dr` cross-term controlled by `beta(r)`

## Built-in Metrics

The built-in metric registry currently includes:

- `schwarzschild`
- `reissner_nordstrom`
- `de_sitter_static`
- `minkowski_spherical`
- `frw_flat`
- `static_spherical`
- `morris_thorne_wormhole`
- `pg_areal`
- `pg_spatial_conformal`

These are defined in [gr_metric_library.py](C:/Users/Nelson/Downloads/GR_python/gr_metric_library.py).

## Add a New Built-in Metric

If you want a metric to become a reusable named option:

1. Open [gr_metric_library.py](C:/Users/Nelson/Downloads/GR_python/gr_metric_library.py).
2. Add one more entry inside `build_builtin_metric_library()`.
3. Reuse the same keys as `CUSTOM_METRIC_CONFIG`:
   - `g_metric`
   - `metric_name`
   - `metric_description`
   - optional `g_inv_metric`
   - optional `e_tetrad`
4. Go back to [gr_main.py](C:/Users/Nelson/Downloads/GR_python/gr_main.py) and set:
   ```python
   METRIC_KEY = 'your_new_key'
   ```

## Useful Flags

Inside [gr_main.py](C:/Users/Nelson/Downloads/GR_python/gr_main.py), the most useful switches are:

- `FAST_MODE = True` to skip heavy invariant computations
- `COMPUTE_TETRAD = True` to keep orthonormal-frame analysis enabled
- `COMPUTE_WEYL = True`
- `COMPUTE_KRETSCHMANN = True`
- `OUTPUT_FILENAME = 'gr_report'`

## Project Structure

- [gr_calculator.py](C:/Users/Nelson/Downloads/GR_python/gr_calculator.py): entry point
- [gr_main.py](C:/Users/Nelson/Downloads/GR_python/gr_main.py): local user configuration and pipeline
- [gr_metric_library.py](C:/Users/Nelson/Downloads/GR_python/gr_metric_library.py): built-in metric registry
- [gr_tensors.py](C:/Users/Nelson/Downloads/GR_python/gr_tensors.py): symbolic tensor engine
- [gr_latex.py](C:/Users/Nelson/Downloads/GR_python/gr_latex.py): LaTeX/PDF report builder
