# GR Calculator - Google Colab Edition

Symbolic General Relativity analysis powered by SymPy, with optional GPU numerical post-processing.

## Files

| File | Purpose |
|---|---|
| `GR_Colab.ipynb` | Main notebook |
| `gr_main.py` | Local driver and configuration |
| `gr_metric_library.py` | Built-in metric registry |
| `gr_tensors.py` | Symbolic tensor engine |
| `gr_latex.py` | LaTeX report assembly |
| `gr_numerics.py` | Numerical evaluation and plotting |
| `gr_warp.py` | Warp-document helpers and comparisons |

## Colab Use Step by Step

1. Open `GR_Colab.ipynb` in Google Colab.
2. Run **Cell 1** to install LaTeX and Python dependencies.
3. Run **Cell 2** to clone the GitHub branch and import the modules.
4. Edit **Cell 3**.
5. Run **Cell 4** for the symbolic calculation.
6. Run **Cell 5** to generate the report files.
7. If you want numerics and plots, continue with **Cell 6**, **Cell 7**, and **Cell 8**.

## What Each Important Cell Does

- **Cell 1** installs packages
- **Cell 2** loads the project from GitHub
- **Cell 3** is where you choose the metric/profile/report options
- **Cell 4** runs the symbolic GR computation
- **Cell 5** writes `gr_report.pdf` and, if enabled, `gr_comparison_report.pdf`

## Standard Warp-Document Workflow

In **Cell 3**, the usual configuration is:

```python
VARIANT = 'variant_a'
PROFILE_MODE = 'document_generic'
RUN_DOCUMENT_COMPARISON = True
GENERATE_COMPARISON_REPORT = True
```

This means:

- `gr_report.pdf` is the direct output of the symbolic run
- `gr_comparison_report.pdf` is optional and only used for checks against external formulas

## If You Want a Brand-New Metric

For a completely new metric outside the predefined warp variants, use the local driver [gr_main.py](C:/Users/Nelson/Downloads/GR_python/GR_python_colab/gr_main.py) with the same workflow as the desktop version:

1. set `METRIC_KEY = 'custom'`
2. define any extra symbols/functions in Section 1.2
3. fill `CUSTOM_METRIC_CONFIG`
4. run the script locally, or adapt the same metric definition into a notebook cell

Two commented examples are already included in [gr_main.py](C:/Users/Nelson/Downloads/GR_python/GR_python_colab/gr_main.py):

- a diagonal ansatz
- a metric with a `dt dr` cross-term

Reusable built-in metrics are stored in [gr_metric_library.py](C:/Users/Nelson/Downloads/GR_python/GR_python_colab/gr_metric_library.py). Useful examples include Schwarzschild, Reissner-Nordstrom, static de Sitter, and a Morris-Thorne wormhole template.

## GPU and Parallel Notes

- Symbolic SymPy calculations run on CPU.
- GPU support accelerates only numerical post-processing.
- Parallel symbolic execution is available on Linux/Colab for selected stages.
