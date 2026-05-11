# -*- coding: utf-8 -*-
"""
gr_calculator.py — General Relativity Symbolic Calculator

Run this file to compute all GR quantities and generate a PDF report:
    python gr_calculator.py

For interactive help:
    python gr_calculator.py --help
    python gr_help.py examples

To customise the metric or computation flags, edit Section 1 in gr_main.py.

Module layout:
  gr_tensors.py  — All tensor computation functions (Christoffel, Riemann, ...)
  gr_latex.py    — LaTeX helpers and report assembly (assemble_report)
  gr_main.py     — User input (Section 1), pipeline, PDF compiler, entry point
  gr_calculator.py — This thin dispatcher (just calls gr_main.main)
"""

import sys

from gr_main import main
from gr_help import main as help_main


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in {'-h', '--help', 'help'}:
        raise SystemExit(help_main(sys.argv[2:]))
    main()
