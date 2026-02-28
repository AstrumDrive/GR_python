# -*- coding: utf-8 -*-
"""
gr_calculator.py — General Relativity Symbolic Calculator

Run this file to compute all GR quantities and generate a PDF report:
    python gr_calculator.py

To customise the metric or computation flags, edit Section 1 in gr_main.py.

Module layout:
  gr_tensors.py  — All tensor computation functions (Christoffel, Riemann, ...)
  gr_latex.py    — LaTeX helpers and report assembly (assemble_report)
  gr_main.py     — User input (Section 1), pipeline, PDF compiler, entry point
  gr_calculator.py — This thin dispatcher (just calls gr_main.main)
"""

from gr_main import main


if __name__ == '__main__':
    main()