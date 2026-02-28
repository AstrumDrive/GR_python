# General Relativity Symbolic Calculator

A highly automated, symbolic General Relativity calculator and report generator powered by SymPy and LaTeX.

## About

This project is a powerful symbolic calculation engine tailored specifically for researchers, students, and enthusiasts in General Relativity. By leveraging SymPy's symbolic math capabilities, it eliminates the tedious and error-prone process of manually deriving tensor components (like Christoffel symbols and the Riemann tensor) by hand. 

Whether you are exploring classic solutions (Schwarzschild, Kerr, FLRW) or testing novel spacetime metrics (e.g., warp drives or wormholes), this calculator provides a robust, automated pipeline to extract the physics directly from the geometry. 

## Overview

This tool takes a user-defined spacetime metric (like Schwarzschild, FLRW, or any custom metric) and automatically computes all the essential geometric and physical quantities in General Relativity:

*   **Christoffel Symbols** (Levi-Civita connection)
*   **Riemann Curvature Tensor**
*   **Ricci Tensor** & **Ricci Scalar**
*   **Einstein Tensor**
*   **Curvature Invariants** (Kretschmann scalar and Weyl conformal tensor)
*   **Orthonormal Frame Analysis** (Tetrads & Stress-Energy components)
*   **Energy Conditions** (Null, Weak, Strong, Dominant)
*   **Geodesic Equations**
*   **Symmetries** (Killing vector detection for cyclic coordinates)
*   **Conservation Checks** (Bianchi identity verification)

Finally, it compiles all these results into a beautifully formatted, easily readable **LaTeX PDF report**.

## Requirements

*   **Python 3.x**
*   **SymPy** (`pip install sympy`)
*   **LaTeX distribution** with `pdflatex` in your system `PATH` (e.g., MiKTeX on Windows, TeX Live on Linux, MacTeX on macOS).

## How to Use

1. **Define Your Metric:**
   Open `gr_main.py` and navigate to **SECTION 1 — USER INPUT**.
   This is the *only* section you need to modify. You can:
   *   Define your coordinate symbols (e.g., `t, r, theta, phi`).
   *   Define parameters (e.g., Mass `M`) or symbolic functions (e.g., scale factor `a(t)`).
   *   Enter your covariant metric $g_{\mu\nu}$ as a SymPy `Matrix`.

2. **Run the Calculator:**
   Execute the main script from your terminal:
   ```bash
   python gr_calculator.py
   ```
   *(Alternatively, you can run `python gr_main.py` directly).*

3. **View the Results:**
   The script will perform the symbolic math (progress is printed to the console) and then generate `gr_report.tex`. It will automatically attempt to compile it into `gr_report.pdf` using `pdflatex`. Open `gr_report.pdf` to see your full analysis!

## Customisation and Computation Flags

Inside `gr_main.py` (Section 1), there are flags to control the computation time, as some tensors (like the Kretschmann scalar or Weyl tensor) can take several minutes to compute for complex symbolic metrics.

*   `FAST_MODE = True`: Skips the heavy Kretschmann and Weyl computations, making the script run much faster. Recommended for your first attempt with a new metric.
*   `COMPUTE_TETRAD = True`: Automatically computes the orthonormal tetrad via ADM decomposition. You can also provide a custom tetrad manually.
*   `OUTPUT_FILENAME = "gr_report"`: Change this to name the output LaTeX and PDF files.

## Project Structure

*   `gr_calculator.py`: The entry-point script. Run this file.
*   `gr_main.py`: Contains the **User Input** section, manages the computation pipeline, and calls the PDF compiler.
*   `gr_tensors.py`: The math engine. Contains all the symbolic tensor calculus functions.
*   `gr_latex.py`: The formatting engine. Converts SymPy expressions to LaTeX and constructs the final document.
