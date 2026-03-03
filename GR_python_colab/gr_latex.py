# -*- coding: utf-8 -*-
"""
gr_latex.py — LaTeX Helper Utilities and Report Assembly

All LaTeX generation utilities and the assemble_report() function.
Imported by gr_main.py.

Sections contained:
  0 — Imports
  2 — Helper Utilities (progress, coord_latex, split_long_equation, ...)
  4 — LaTeX Generation Utilities (make_latex_header, make_tcolorbox, ...)
  6 — Report Assembly (assemble_report)
"""

# ==============================================================================
# SECTION 0 — IMPORTS
# ==============================================================================

import sympy as sp
from sympy import (
    symbols, Function, Matrix, sqrt, sin, cos, exp,
    diff, simplify, cancel, trigsimp, Rational, pi,
    latex, zeros, eye, S, Abs, Integer
)
import os
import sys
import time
from datetime import datetime

# progress(), _COORD_LATEX_MAP, and coord_latex() are all defined in
# gr_tensors (the lowest-level module) to avoid circular imports.
# gr_latex imports from gr_tensors; gr_tensors never imports from gr_latex.
from gr_tensors import progress, _COORD_LATEX_MAP, coord_latex


# ==============================================================================
# SECTION 2 — HELPER UTILITIES
# ==============================================================================


def idx_lat(coords_list, i):
    """Return the LaTeX label for coordinate index i."""
    return coord_latex(coords_list[i])


def expr_to_latex(expr, subs=None):
    """
    Convert a SymPy expression to a LaTeX string.

    Steps:
      1. Apply cancel() to simplify rational functions.
      2. Call sp.latex() to get the LaTeX string.
      3. Apply user-defined string substitutions (subs dict).

    Parameters
    ----------
    expr : SymPy expression
    subs : dict, optional
        String-level replacements applied after latex().

    Returns
    -------
    str : LaTeX string
    """
    try:
        simplified = cancel(expr)
    except Exception:
        simplified = expr
    lat = latex(simplified)
    if subs:
        for old, new in subs.items():
            lat = lat.replace(old, new)
    return lat


def matrix_to_latex(mat, env='pmatrix'):
    """
    Convert a SymPy Matrix to a LaTeX matrix environment.

    Parameters
    ----------
    mat : sympy.Matrix
    env : str
        LaTeX matrix environment name ('pmatrix', 'bmatrix', 'vmatrix', ...)

    Returns
    -------
    str : LaTeX string for the matrix
    """
    rows = []
    for i in range(mat.rows):
        row_entries = []
        for j in range(mat.cols):
            entry = mat[i, j]
            try:
                entry = cancel(entry)
            except Exception:
                pass
            row_entries.append(latex(entry))
        rows.append(' & '.join(row_entries))
    body = ' \\\\\n    '.join(rows)
    return f'\\begin{{{env}}}\n    {body}\n\\end{{{env}}}'


def add_dmath(expr_latex_str):
    """
    Wrap a LaTeX expression in a breqn dmath* environment.
    dmath* automatically breaks long equations across lines — essential
    for verbose Christoffel/Riemann components.

    Returns list of strings (LaTeX lines).
    """
    return [
        r'\begin{dmath*}',
        f'  {expr_latex_str}',
        r'\end{dmath*}',
    ]


def add_align(lhs, rhs):
    """Return lines for a single align* equation  lhs = rhs."""
    return [
        r'\begin{align*}',
        f'  {lhs} &= {rhs}',
        r'\end{align*}',
    ]


# ==============================================================================
# OVERFLOW-PREVENTION THRESHOLDS
# Calibrated against observed overflows in gr_report.log:
#   LATEX_MEDIUM_THRESHOLD = 400  — expressions longer than this are split
#     (smallest overflow observed: 785-char Ricci scalar → 498pt overflow)
#   MATRIX_ENTRY_THRESHOLD = 150  — any matrix entry longer than this triggers
#     component-list mode instead of pmatrix
#     (g entries ≤126 chars: OK; ginv entries ≥199 chars: 132pt overflow)
# ==============================================================================
LATEX_MEDIUM_THRESHOLD = 400   # chars: above this → use multline* splitting
MATRIX_ENTRY_THRESHOLD = 150   # chars: any single matrix entry above → component list


def _extract_single_fraction(s):
    """
    If s is exactly one \\frac{N}{D} (possibly followed by whitespace),
    return (numerator_str, denominator_str, remainder_str).
    Uses brace-balanced extraction to handle nested \\frac{}{} inside.

    Returns (None, None, s) if s does not start with \\frac{.

    This is used by split_long_equation to factor long expressions as
    (1/D)*(N) and split N across multiple lines with multline*.
    """
    s = s.strip()
    if not s.startswith(r'\frac{'):
        return None, None, s

    # Extract numerator: everything between the first pair of braces after \frac
    i = 6           # index right after '\frac{'
    depth = 1
    while i < len(s) and depth > 0:
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
        i += 1
    numerator = s[6:i - 1]   # content between the braces

    # Extract denominator: the second brace group
    if i >= len(s) or s[i] != '{':
        return numerator, None, s[i:]
    j = i + 1
    depth = 1
    while j < len(s) and depth > 0:
        if s[j] == '{':
            depth += 1
        elif s[j] == '}':
            depth -= 1
        j += 1
    denominator = s[i + 1:j - 1]
    remainder   = s[j:]
    return numerator, denominator, remainder


def _find_toplevel_plusminus(s):
    """
    Return a list of indices where '+' or '-' appears at brace-depth 0 in s.

    These positions are safe line-break points in a LaTeX math sum because
    TeX can insert a line break at top-level binary operators.
    Indices pointing at the very first character (i=0) are excluded since a
    leading '-' or '+' belongs to the first term, not a split point.

    Parameters
    ----------
    s : str — LaTeX math string (the right-hand side of an equation)

    Returns
    -------
    list of int — character indices of top-level +/- split points
    """
    depth  = 0
    points = []
    for i, c in enumerate(s):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        elif c in '+-' and depth == 0 and i > 0:
            points.append(i)
    return points


def split_long_equation(lhs, rhs, inside_tcolorbox=False):
    """
    Produce LaTeX lines for  'lhs = rhs'  with automatic overflow prevention.

    Strategy (applied in priority order):
    ─────────────────────────────────────
    1. Short (len ≤ LATEX_MEDIUM_THRESHOLD):
         → dmath* (breqn handles line-breaking for non-monolithic expressions)

    2. Single fraction  \\frac{N}{D}  where N has top-level +/− split points:
         → multline* written as  \\frac{1}{D}\\Bigl( N_1 \\\\ + N_2 \\\\ ... \\Bigr)
         Mathematically equivalent; splits the numerator across lines.

    3. Top-level sum (not a fraction) with top-level +/− split points:
         → multline*: split at each +/− sign, one term per line.

    4. Monolithic (no split points — a single nested fraction or product):
         → {\\sloppy \\begin{dmath*}...\\end{dmath*}}
         Relaxes TeX's strictness; may still cause slight overflow for
         pathologically long atomic tokens, but avoids hard overflow.

    Both dmath* and multline* work correctly inside tcolorbox[breakable].

    Parameters
    ----------
    lhs              : str  — LaTeX for the left-hand side (e.g. r'\\mathcal{K}')
    rhs              : str  — LaTeX for the right-hand side expression
    inside_tcolorbox : bool — reserved for future context-specific adjustments

    Returns
    -------
    list of str — LaTeX source lines
    """
    rhs = rhs.strip()
    n   = len(rhs)

    # ── Case 1: short enough for dmath* ──────────────────────────────────────
    if n <= LATEX_MEDIUM_THRESHOLD:
        return add_dmath(f'{lhs} = {rhs}')

    # ── Case 2: single \\frac{N}{D} with splittable numerator ─────────────────
    num, den, rest = _extract_single_fraction(rhs)
    if num is not None and den is not None and not rest.strip():
        splits = _find_toplevel_plusminus(num)
        if splits:
            # Slice numerator into segments at each split point
            segs, prev = [], 0
            for sp in splits:
                segs.append(num[prev:sp])
                prev = sp
            segs.append(num[prev:])

            # Build align* with the factored form  (1/D) * (N1 + N2 + ...)
            # align* aligns every continuation line at the & marker so that
            # all terms sit flush under the = sign — far cleaner than multline*
            # which left-aligns line 1, centres middle lines, right-aligns last.
            out = [r'\begin{align*}']
            out.append(f'  {lhs} &= \\frac{{1}}{{{den}}}\\Bigl({segs[0].strip()} \\\\')
            for seg in segs[1:-1]:
                out.append(f'  &\\quad {seg.strip()} \\\\')
            # Last segment: close the big parenthesis (no trailing \\)
            out.append(f'  &\\quad {segs[-1].strip()}\\Bigr)')
            out.append(r'\end{align*}')
            return out

    # ── Case 3: top-level sum (not a fraction) ────────────────────────────────
    top_splits = _find_toplevel_plusminus(rhs)
    if top_splits:
        segs, prev = [], 0
        for sp in top_splits:
            segs.append(rhs[prev:sp])
            prev = sp
        segs.append(rhs[prev:])

        # align* keeps all continuation lines left-aligned at a consistent
        # &\quad indent — much cleaner than multline* staircase layout.
        out = [r'\begin{align*}']
        out.append(f'  {lhs} &= {segs[0].strip()} \\\\')
        for seg in segs[1:-1]:
            out.append(f'  &\\quad {seg.strip()} \\\\')
        out.append(f'  &\\quad {segs[-1].strip()}')
        out.append(r'\end{align*}')
        return out

    # ── Case 4: monolithic fallback — \\sloppy + dmath* ──────────────────────
    # \sloppy relaxes TeX's inter-word and inter-character spacing constraints,
    # giving breqn more room to break the line. This avoids a hard crash/overflow
    # for atomic tokens that truly cannot be split (e.g. a single enormous symbol).
    return [
        r'{\sloppy',
        r'\begin{dmath*}',
        f'  {lhs} = {rhs}',
        r'\end{dmath*}',
        r'}',
    ]


def matrix_is_complex(mat, threshold=MATRIX_ENTRY_THRESHOLD):
    """
    Return True if any entry of mat has a LaTeX representation longer
    than `threshold` characters (checked after cancel()).

    Used to decide between pmatrix display and component-list display.

    Parameters
    ----------
    mat       : sympy.Matrix
    threshold : int — character length cutoff (default: MATRIX_ENTRY_THRESHOLD)

    Returns
    -------
    bool
    """
    for i in range(mat.rows):
        for j in range(mat.cols):
            entry = mat[i, j]
            try:
                entry = cancel(entry)
            except Exception:
                pass
            if len(latex(entry)) > threshold:
                return True
    return False


def matrix_to_component_list(mat, row_labels, col_labels, tensor_name):
    """
    Display a matrix as a list of individual non-zero component equations
    instead of a pmatrix block.

    Used when matrix entries are too long to display in a pmatrix without
    overflowing the page (triggered by matrix_is_complex()).

    Each non-zero entry T_{row col} = expr is rendered via split_long_equation,
    so even very long entries are handled gracefully.

    Parameters
    ----------
    mat         : sympy.Matrix
    row_labels  : list of str  — LaTeX index label for each row    (e.g. ['t','r',r'\\theta',r'\\varphi'])
    col_labels  : list of str  — LaTeX index label for each column
    tensor_name : str          — LaTeX symbol for the tensor (e.g. 'g', 'R', 'G')

    Returns
    -------
    list of str — LaTeX lines (sequence of dmath*/multline* blocks)
    """
    out = [r'\par\noindent\textit{Non-zero components (matrix form exceeds page width):}']
    has_any = False
    for i in range(mat.rows):
        for j in range(mat.cols):
            entry = mat[i, j]
            try:
                entry = cancel(entry)
            except Exception:
                pass
            if entry == S.Zero:
                continue
            has_any = True
            lhs_str = f'{tensor_name}_{{{row_labels[i]}{col_labels[j]}}}'
            rhs_str = latex(entry)
            out += split_long_equation(lhs_str, rhs_str)
    if not has_any:
        out = [r'\par\noindent All components vanish (zero tensor).']
    return out


def get_nonzero_christoffel(Gamma, dim):
    """
    Return list of (λ, μ, ν, value) for non-zero Christoffel symbols.
    Only returns upper triangle μ ≤ ν (using lower-index symmetry).
    """
    result = []
    for lam in range(dim):
        for mu in range(dim):
            for nu in range(mu, dim):
                val = Gamma[lam][mu][nu]
                if val != S.Zero:
                    result.append((lam, mu, nu, val))
    return result


def get_independent_riemann(R_riem, dim):
    """
    Return list of (ρ, σ, μ, ν, value) for independent non-zero
    Riemann components. Only μ < ν (antisymmetry in last pair).
    """
    result = []
    for rho in range(dim):
        for sig in range(dim):
            for mu in range(dim):
                for nu in range(mu + 1, dim):
                    val = R_riem[rho][sig][mu][nu]
                    if val != S.Zero:
                        result.append((rho, sig, mu, nu, val))
    return result


# ==============================================================================


# ==============================================================================
# SECTION 4 — LATEX GENERATION UTILITIES
# ==============================================================================

def make_latex_header(metric_name, description, author):
    """
    Build the complete LaTeX preamble, document settings, and title block.

    Packages used
    -------------
    amsmath, amssymb, mathtools  — standard math environments
    physics                      — \dd, \partial, tensor notation
    breqn                        — automatic line-breaking of long equations
    geometry                     — page margins
    hyperref                     — clickable TOC links
    booktabs                     — professional tables
    tcolorbox                    — coloured summary boxes
    lmodern, microtype           — improved fonts and typography
    inputenc, fontenc             — UTF-8 and T1 font encoding

    Returns
    -------
    list of str — LaTeX lines from \documentclass to \maketitle
    """
    lines = []
    lines.append(r'\documentclass[11pt,a4paper]{article}')
    lines.append(r'\usepackage[utf8]{inputenc}')
    lines.append(r'\usepackage[T1]{fontenc}')
    lines.append(r'\usepackage{lmodern}')
    lines.append(r'\usepackage{microtype}')
    lines.append(r'\usepackage{amsmath,amssymb,amsfonts,amsthm}')
    lines.append(r'\usepackage{mathtools}')
    lines.append(r'\usepackage{physics}')
    lines.append(r'\usepackage{breqn}')
    lines.append(r'\usepackage{bm}')
    lines.append(r'\usepackage[margin=2.5cm]{geometry}')
    lines.append(r'\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}')
    lines.append(r'\usepackage{booktabs}')
    lines.append(r'\usepackage{array}')
    lines.append(r'\usepackage{longtable}')
    lines.append(r'\usepackage[most]{tcolorbox}')
    lines.append(r'\usepackage{xcolor}')
    lines.append(r'\usepackage{graphicx}')          # \resizebox fallback for extreme cases
    lines.append(r'')
    lines.append(r'% ---- Custom colours ----')
    lines.append(r'\definecolor{boxblue}{RGB}{220,235,250}')
    lines.append(r'\definecolor{boxgreen}{RGB}{220,245,225}')
    lines.append(r'\definecolor{boxred}{RGB}{250,225,220}')
    lines.append(r'')
    lines.append(r'% ---- Tensor notation shortcuts ----')
    lines.append(r'\newcommand{\Ghat}{\hat{G}}')
    lines.append(r'\newcommand{\Rhat}{\hat{R}}')
    lines.append(r'\newcommand{\christoffel}[3]{\Gamma^{#1}{}_{#2 #3}}')
    lines.append(r'\newcommand{\riemann}[4]{R^{#1}{}_{#2 #3 #4}}')
    lines.append(r'')
    # Title
    lines.append(r'\title{%')
    lines.append(r'  \textbf{General Relativity Analysis} \\[0.6em]')
    lines.append(r'  \large\textit{' + metric_name + r' Metric} \\[0.3em]')
    lines.append(r'  \normalsize ' + description)
    lines.append(r'}')
    lines.append(r'\author{' + author + r'}')
    lines.append(r'\date{' + datetime.now().strftime('%B %d, %Y') + r'}')
    lines.append(r'')
    lines.append(r'\begin{document}')
    lines.append(r'\allowdisplaybreaks')             # allow page-breaks inside align environments
    lines.append(r'\maketitle')
    lines.append(r'\tableofcontents')
    lines.append(r'\newpage')
    lines.append(r'')
    return lines


def make_latex_footer():
    """Return the closing \\end{document} line."""
    return ['', r'\end{document}']


def make_tcolorbox(title, content_lines, color='boxblue'):
    """
    Wrap content in a tcolorbox with a title.

    Parameters
    ----------
    title         : str   — box title
    content_lines : list  — LaTeX lines to place inside the box
    color         : str   — background colour name

    Returns
    -------
    list of str — LaTeX lines
    """
    lines = [
        r'\begin{tcolorbox}[colback=' + color + r',colframe=black!40,'
        r'title=\textbf{' + title + r'},breakable]',
    ]
    lines.extend(content_lines)
    lines.append(r'\end{tcolorbox}')
    return lines


# ==============================================================================


# ==============================================================================
# SECTION 6 — REPORT ASSEMBLY
# ==============================================================================

def assemble_report(results, coords, dim, metric_name, description,
                    author, latex_subs_dict, g_metric_orig,
                    e_tetrad=None):
    """
    Convert all computed results into a complete LaTeX document.

    Structure of the generated PDF
    --------------------------------
    Abstract
    1. Metric Tensor and Conventions
    2. Christoffel Symbols
    3. Riemann Curvature Tensor
    4. Ricci Tensor
    5. Ricci Scalar
    6. Einstein Tensor
    7. Curvature Invariants (Kretschmann, Weyl)
    8. Orthonormal Frame Analysis
    9. Energy Conditions
    10. Geodesic Equations
    11. Conservation and Consistency Checks

    Parameters
    ----------
    results          : dict — output of run_computations()
    coords           : list — coordinate symbols
    dim              : int
    metric_name      : str
    description      : str
    author           : str
    latex_subs_dict  : dict — string-level LaTeX substitutions
    g_metric_orig    : sympy.Matrix — original metric (for display)
    e_tetrad         : sympy.Matrix or None

    Returns
    -------
    list of str — lines of the complete LaTeX document
    """
    R = results
    RL = []    # report_lines

    # ---- Header ----
    RL += make_latex_header(metric_name, description, author)

    # ---- Abstract ----
    RL.append(r'\begin{abstract}')
    RL.append(r'\sloppy')   # relax line-breaking in abstract to prevent text overflow
    RL.append(
        r'This document presents a complete symbolic General Relativity analysis '
        r'of the \textbf{' + metric_name + r'} metric. All quantities were computed '
        r'analytically using SymPy. The report includes: the metric and its inverse, '
        r'Christoffel symbols, Riemann and Ricci tensors, Ricci scalar, Einstein tensor, '
        r'curvature invariants (Kretschmann scalar, Weyl tensor), orthonormal-frame '
        r'stress-energy components, energy conditions, geodesic equations, and '
        r'conservation law verification.'
    )
    RL.append(r'\end{abstract}')
    RL.append(r'\newpage')

    def lat(expr):
        """Shorthand: expression → LaTeX string with user subs."""
        return expr_to_latex(expr, latex_subs_dict)

    # =========================================================================
    # SECTION 1 — METRIC
    # =========================================================================
    RL.append(r'\section{Metric Tensor and Conventions}')

    RL.append(r'\subsection{Coordinate System and Conventions}')
    # Build coordinate list as a SINGLE math-mode span to avoid the
    # "Missing $ inserted" error caused by toggling $...$ around each symbol.
    # e.g. for [t, r, θ, φ] → $(t,\,r,\,\theta,\,\varphi)$
    coord_str = r',\,'.join(latex(c) for c in coords)
    RL.append(
        r'Coordinates: $(' + coord_str + r')$. '
        r'Spacetime signature: $(-,+,+,+)$. '
        r'Natural units: $G = c = 1$. '
        r'Greek indices run over $0,1,\ldots,' + str(dim - 1) + r'$; '
        r'Latin hat-indices label orthonormal frame legs.'
    )

    RL.append(r'\subsection{Covariant Metric $g_{\mu\nu}$}')
    RL.append(r'The line element is $ds^2 = g_{\mu\nu}\,dx^\mu dx^\nu$.')
    _idx = [idx_lat(coords, i) for i in range(dim)]
    if matrix_is_complex(R['g']):
        RL += matrix_to_component_list(R['g'], _idx, _idx, 'g')
    else:
        RL.append(r'\begin{equation}')
        RL.append(r'  g_{\mu\nu} = ' + matrix_to_latex(R['g']))
        RL.append(r'\end{equation}')

    RL.append(r'\subsection{Contravariant Metric $g^{\mu\nu}$}')
    RL.append(r'Defined by $g^{\mu\rho}\,g_{\rho\nu} = \delta^\mu{}_\nu$.')
    if matrix_is_complex(R['ginv']):
        RL += matrix_to_component_list(R['ginv'], _idx, _idx, 'g^')
    else:
        RL.append(r'\begin{equation}')
        RL.append(r'  g^{\mu\nu} = ' + matrix_to_latex(R['ginv']))
        RL.append(r'\end{equation}')

    RL.append(r'\subsection{Metric Determinant}')
    RL += split_long_equation(r'\det(g)', lat(R['det_g']))

    # =========================================================================
    # SECTION 2 — CHRISTOFFEL SYMBOLS
    # =========================================================================
    RL.append(r'\section{Christoffel Symbols}')
    RL.append(
        r'The Christoffel symbols of the second kind (Levi-Civita connection) are'
    )
    RL.append(r'\begin{equation}')
    RL.append(
        r'  \christoffel{\lambda}{\mu}{\nu} = \frac{1}{2}\,g^{\lambda\sigma}'
        r'\!\left(\partial_\mu g_{\nu\sigma} + \partial_\nu g_{\mu\sigma}'
        r'- \partial_\sigma g_{\mu\nu}\right).'
    )
    RL.append(r'\end{equation}')
    RL.append(
        r'They are symmetric in the lower pair: '
        r'$\christoffel{\lambda}{\mu}{\nu} = \christoffel{\lambda}{\nu}{\mu}$. '
        r'Only non-zero independent components (with $\mu \le \nu$) are listed.'
    )

    nz_chris = get_nonzero_christoffel(R['Gamma'], dim)
    if not nz_chris:
        RL += make_tcolorbox(
            'Result',
            [r'All Christoffel symbols vanish identically. The spacetime is flat.'],
            'boxgreen'
        )
    else:
        RL.append(rf'Total non-zero independent components: \textbf{{{len(nz_chris)}}}.')
        # Each component rendered individually so split_long_equation can apply
        # multline* for long entries (the old single align* block had no line-breaking).
        for (lam, mu, nu, val) in nz_chris:
            lhs = (r'\christoffel{' + idx_lat(coords, lam) + r'}{'
                   + idx_lat(coords, mu) + r'}{'
                   + idx_lat(coords, nu) + r'}')
            RL += split_long_equation(lhs, lat(val))

    # =========================================================================
    # SECTION 3 — RIEMANN TENSOR
    # =========================================================================
    RL.append(r'\section{Riemann Curvature Tensor}')
    RL.append(
        r'The Riemann tensor measures the failure of parallel transport around '
        r'closed loops — the intrinsic curvature of spacetime:'
    )
    RL.append(r'\begin{equation}')
    RL.append(
        r'  \riemann{\rho}{\sigma}{\mu}{\nu} = '
        r'\partial_\mu\christoffel{\rho}{\nu}{\sigma}'
        r'- \partial_\nu\christoffel{\rho}{\mu}{\sigma}'
        r'+ \christoffel{\rho}{\mu}{\lambda}\christoffel{\lambda}{\nu}{\sigma}'
        r'- \christoffel{\rho}{\nu}{\lambda}\christoffel{\lambda}{\mu}{\sigma}.'
    )
    RL.append(r'\end{equation}')
    RL.append(
        r'Antisymmetry: $\riemann{\rho}{\sigma}{\mu}{\nu} = -\riemann{\rho}{\sigma}{\nu}{\mu}$. '
        r'Only components with $\mu < \nu$ are listed.'
    )

    nz_riem = get_independent_riemann(R['R_riem'], dim)
    if not nz_riem:
        RL += make_tcolorbox(
            'Result',
            [r'All Riemann components vanish. The spacetime is flat (Minkowski).'],
            'boxgreen'
        )
    else:
        RL.append(rf'Total independent non-zero Riemann components: \textbf{{{len(nz_riem)}}}.')
        for (rho, sig, mu, nu, val) in nz_riem:
            lhs = (r'\riemann{' + idx_lat(coords, rho) + r'}{'
                   + idx_lat(coords, sig) + r'}{'
                   + idx_lat(coords, mu) + r'}{'
                   + idx_lat(coords, nu) + r'}')
            RL += split_long_equation(lhs, lat(val))

    # =========================================================================
    # SECTION 4 — RICCI TENSOR
    # =========================================================================
    RL.append(r'\section{Ricci Tensor}')
    RL.append(
        r'The Ricci tensor is the trace of the Riemann tensor:'
        r' $R_{\mu\nu} = R^\rho{}_{\mu\rho\nu}$. '
        r'It encodes how volumes change under parallel transport '
        r'and is directly sourced by the stress-energy.'
    )
    _idx = [idx_lat(coords, i) for i in range(dim)]
    if matrix_is_complex(R['Ric']):
        RL += matrix_to_component_list(R['Ric'], _idx, _idx, 'R')
    else:
        RL.append(r'\begin{equation}')
        RL.append(r'  R_{\mu\nu} = ' + matrix_to_latex(R['Ric']))
        RL.append(r'\end{equation}')

    # =========================================================================
    # SECTION 5 — RICCI SCALAR
    # =========================================================================
    RL.append(r'\section{Ricci Scalar (Curvature Scalar)}')
    RL.append(
        r'The Ricci scalar $\mathcal{R} = g^{\mu\nu} R_{\mu\nu}$ is the '
        r'full contraction of the Ricci tensor. It provides a single-number '
        r'measure of curvature. It vanishes for vacuum spacetimes.'
    )
    RL += make_tcolorbox(
        r'Ricci scalar',
        split_long_equation(r'\mathcal{R}', lat(R['R_scalar']), inside_tcolorbox=True),
        'boxblue'
    )

    # =========================================================================
    # SECTION 6 — EINSTEIN TENSOR
    # =========================================================================
    RL.append(r'\section{Einstein Tensor}')
    RL.append(
        r'The Einstein tensor $G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2}\,g_{\mu\nu}\,\mathcal{R}$ '
        r'is divergence-free by the contracted Bianchi identity. '
        r'The Einstein field equations read $G_{\mu\nu} = 8\pi T_{\mu\nu}$.'
    )
    _idx = [idx_lat(coords, i) for i in range(dim)]
    if matrix_is_complex(R['G']):
        RL += matrix_to_component_list(R['G'], _idx, _idx, 'G')
    else:
        RL.append(r'\begin{equation}')
        RL.append(r'  G_{\mu\nu} = ' + matrix_to_latex(R['G']))
        RL.append(r'\end{equation}')

    # =========================================================================
    # SECTION 7 — CURVATURE INVARIANTS
    # =========================================================================
    RL.append(r'\section{Curvature Invariants}')

    # 7a — Kretschmann
    RL.append(r'\subsection{Kretschmann Scalar}')
    RL.append(
        r'The Kretschmann scalar $\mathcal{K} = R^{\mu\nu\rho\sigma} R_{\mu\nu\rho\sigma}$ '
        r'is a quadratic curvature invariant. Unlike $\mathcal{R}$, it '
        r'remains nonzero even in vacuum. It diverges at genuine (physical) '
        r'curvature singularities, distinguishing them from coordinate singularities. '
        r'For the Schwarzschild metric: $\mathcal{K} = 48M^2/r^6$.'
    )
    if R['K'] is not None:
        RL += make_tcolorbox(
            r'Kretschmann scalar',
            split_long_equation(r'\mathcal{K}', lat(R['K']), inside_tcolorbox=True),
            'boxblue'
        )
    else:
        RL.append(r'\textit{Not computed (FAST\_MODE enabled or flag set to False).}')

    # 7b — Weyl tensor
    RL.append(r'\subsection{Weyl Conformal Tensor}')
    RL.append(
        r'The Weyl tensor $C_{\rho\sigma\mu\nu}$ is the trace-free part of the '
        r'Riemann tensor, representing pure gravitational (tidal / wave) '
        r'degrees of freedom independent of the local matter distribution. '
        r'It vanishes in conformally flat spacetimes and is equal to the '
        r'Riemann tensor in vacuum.'
    )
    if R['Weyl'] is not None:
        weyl_nonzero = [
            (rho, sig, mu, nu, R['Weyl'][rho][sig][mu][nu])
            for rho in range(dim)
            for sig in range(dim)
            for mu in range(dim)
            for nu in range(mu + 1, dim)
            if cancel(R['Weyl'][rho][sig][mu][nu]) != S.Zero
        ]
        if not weyl_nonzero:
            RL += make_tcolorbox(
                'Result',
                [r'All Weyl components vanish. The spacetime is conformally flat.'],
                'boxgreen'
            )
        else:
            RL.append(rf'Independent non-zero Weyl components: \textbf{{{len(weyl_nonzero)}}}.')
            for (rho, sig, mu, nu, val) in weyl_nonzero:
                lhs = (r'C_{' + idx_lat(coords, rho)
                       + r'\,' + idx_lat(coords, sig)
                       + r'\,' + idx_lat(coords, mu)
                       + r'\,' + idx_lat(coords, nu) + r'}')
                RL += split_long_equation(lhs, lat(val))
    else:
        RL.append(r'\textit{Not computed (FAST\_MODE, flag off, or $n \neq 4$).}')

    # =========================================================================
    # SECTION 8 — ORTHONORMAL FRAME
    # =========================================================================
    RL.append(r'\section{Orthonormal Frame Analysis}')

    _method   = R.get('tetrad_method')
    _verified = R.get('tetrad_verified')
    _e_contra = R.get('e_tetrad_computed')
    _e_cov    = R.get('e_cov')
    _tet_N    = R.get('tetrad_N')
    _tet_beta = R.get('tetrad_beta')
    _tet_E    = R.get('tetrad_E')
    _residual = R.get('tetrad_residual')

    _hat  = [r'\hat{' + idx_lat(coords, i) + r'}' for i in range(dim)]
    _midx = [idx_lat(coords, i) for i in range(dim)]  # coordinate index labels

    if _method is None:
        # Tetrad entirely skipped
        RL.append(
            r'\textit{Orthonormal frame analysis was skipped. '
            r'Set \texttt{COMPUTE\_TETRAD = True} in Section~1 '
            r'(or supply \texttt{e\_tetrad} manually) to enable this section.}'
        )
    else:
        # ------------------------------------------------------------------
        # 8.1 — Tetrad Construction Method
        # ------------------------------------------------------------------
        RL.append(r'\subsection{Tetrad Construction}')
        _method_desc = {
            'diagonal':
                r'The metric is \textbf{fully diagonal}. '
                r'The tetrad was constructed using the standard formula '
                r'$e^{\hat{0}}{}_\mu = \sqrt{-g_{00}}\,\delta^0_\mu$ and '
                r'$e^{\hat{i}}{}_\mu = \sqrt{g_{ii}}\,\delta^i_\mu$ (no sum).',
            'adm_shift_diagonal_spatial':
                r'The metric has an \textbf{ADM shift vector} with a diagonal '
                r'spatial block ($\gamma_{ij} = 0$ for $i \ne j$). '
                r'The tetrad was built via the ADM decomposition: '
                r'lapse $N$, shift $\beta^i$, and spatial triad $E^{\hat{a}}{}_i = '
                r'\sqrt{\gamma_{ii}}\,\delta^{\hat{a}}_i$ (no sum).',
            'adm_shift_cholesky':
                r'The metric has an \textbf{ADM shift vector} with a non-diagonal '
                r'spatial block. The spatial triad $E^{\hat{a}}{}_i$ was obtained '
                r'via Cholesky factorisation $\gamma_{ij} = E^{\hat{a}}{}_i E_{\hat{a}j}$.',
            'user_supplied':
                r'A \textbf{user-supplied} tetrad was provided in Section~1. '
                r'The code verified orthonormality and projected the Einstein tensor.',
        }
        RL.append(_method_desc.get(_method,
            r'Tetrad constructed by method: \texttt{' + _method.replace('_', r'\_') + r'}.'))

        # ------------------------------------------------------------------
        # 8.2 — ADM Lapse and Shift  (only for ADM methods)
        # ------------------------------------------------------------------
        if _method in ('adm_shift_diagonal_spatial', 'adm_shift_cholesky') \
                and _tet_N is not None:
            RL.append(r'\subsection{ADM Lapse and Shift}')
            RL.append(
                r'The ADM decomposition gives lapse $N$ and shift $\beta^i$:'
            )
            RL += split_long_equation(r'N', lat(_tet_N))
            if _tet_beta is not None:
                for i in range(dim - 1):
                    coord_i = idx_lat(coords, i + 1)
                    RL += split_long_equation(
                        r'\beta^{' + coord_i + r'}',
                        lat(cancel(_tet_beta[i, 0]))
                    )

        # ------------------------------------------------------------------
        # 8.3 — Spatial Triad  (only for ADM methods)
        # ------------------------------------------------------------------
        if _method in ('adm_shift_diagonal_spatial', 'adm_shift_cholesky') \
                and _tet_E is not None:
            RL.append(r'\subsection{Spatial Triad $E^{\hat{a}}{}_i$}')
            RL.append(
                r'The spatial triad satisfies '
                r'$\gamma_{ij} = \delta_{\hat{a}\hat{b}}\,E^{\hat{a}}{}_i\,E^{\hat{b}}{}_j$.'
            )
            _shat  = [r'\hat{' + idx_lat(coords, i + 1) + r'}' for i in range(dim - 1)]
            _sidx  = [idx_lat(coords, i + 1) for i in range(dim - 1)]
            if matrix_is_complex(_tet_E, threshold=MATRIX_ENTRY_THRESHOLD):
                RL += matrix_to_component_list(_tet_E, _shat, _sidx, r'E')
            else:
                RL.append(r'\begin{equation}')
                RL.append(r'  E^{\hat{a}}{}_i = ' + matrix_to_latex(_tet_E))
                RL.append(r'\end{equation}')

        # ------------------------------------------------------------------
        # 8.4 — Contravariant Tetrad e^μ_â
        # ------------------------------------------------------------------
        RL.append(r'\subsection{Contravariant Tetrad $e^\mu{}_{\hat{a}}$}')
        RL.append(
            r'Entry $[\mu,\,\hat{a}]$ of the matrix below is $e^\mu{}_{\hat{a}}$. '
            r'It satisfies $g_{\mu\nu}\,e^\mu{}_{\hat{a}}\,e^\nu{}_{\hat{b}} = '
            r'\eta_{\hat{a}\hat{b}}$.'
        )
        if _e_contra is not None:
            if matrix_is_complex(_e_contra, threshold=MATRIX_ENTRY_THRESHOLD):
                RL += matrix_to_component_list(_e_contra, _midx, _hat, r'e')
            else:
                RL.append(r'\begin{equation}')
                RL.append(r'  e^\mu{}_{\hat{a}} = ' + matrix_to_latex(_e_contra))
                RL.append(r'\end{equation}')

        # ------------------------------------------------------------------
        # 8.5 — Coframe e^â_μ  (only when auto-computed)
        # ------------------------------------------------------------------
        if _e_cov is not None and _method != 'user_supplied':
            RL.append(r'\subsection{Coframe $e^{\hat{a}}{}_\mu$}')
            RL.append(
                r'Entry $[\hat{a},\,\mu]$ of the matrix below is $e^{\hat{a}}{}_\mu$ '
                r'(the dual frame / coframe).'
            )
            if matrix_is_complex(_e_cov, threshold=MATRIX_ENTRY_THRESHOLD):
                RL += matrix_to_component_list(_e_cov, _hat, _midx, r'e')
            else:
                RL.append(r'\begin{equation}')
                RL.append(r'  e^{\hat{a}}{}_\mu = ' + matrix_to_latex(_e_cov))
                RL.append(r'\end{equation}')

        # ------------------------------------------------------------------
        # 8.6 — Orthonormality Verification
        # ------------------------------------------------------------------
        RL.append(r'\subsection{Orthonormality Verification}')
        if _verified is True:
            RL.append(
                r'\begin{tcolorbox}[colback=green!8!white,colframe=green!50!black,'
                r'title=Orthonormality Verified]'
            )
            RL.append(
                r'$g_{\mu\nu}\,e^\mu{}_{\hat{a}}\,e^\nu{}_{\hat{b}} = '
                r'\eta_{\hat{a}\hat{b}} = \operatorname{diag}(-1,+1,+1,+1)$ \quad'
                r'\textbf{\large\checkmark}'
            )
            RL.append(r'\end{tcolorbox}')
        elif _verified is False:
            RL.append(
                r'\begin{tcolorbox}[colback=red!8!white,colframe=red!60!black,'
                r'title=Orthonormality FAILED]'
            )
            RL.append(
                r'The residual $g_{\mu\nu}e^\mu{}_{\hat{a}}e^\nu{}_{\hat{b}} - '
                r'\eta_{\hat{a}\hat{b}}$ is \textbf{not zero}. '
                r'Check the tetrad or simplification assumptions.'
            )
            if _residual is not None:
                if matrix_is_complex(_residual, threshold=MATRIX_ENTRY_THRESHOLD):
                    RL += matrix_to_component_list(_residual, _hat, _hat, r'\Delta')
                else:
                    RL.append(r'\begin{equation}')
                    RL.append(r'  \Delta_{\hat{a}\hat{b}} = ' + matrix_to_latex(_residual))
                    RL.append(r'\end{equation}')
            RL.append(r'\end{tcolorbox}')
        else:
            RL.append(r'\textit{Verification was not performed.}')

        # ------------------------------------------------------------------
        # 8.7 — Orthonormal-Frame Einstein Tensor
        # ------------------------------------------------------------------
        RL.append(r'\subsection{Orthonormal-Frame Einstein Tensor $\hat{G}_{\hat{a}\hat{b}}$}')
        RL.append(
            r'The orthonormal-frame Einstein tensor is defined by '
            r'$\hat{G}_{\hat{a}\hat{b}} = e^\mu{}_{\hat{a}}\,e^\nu{}_{\hat{b}}\,G_{\mu\nu}$.'
        )
        if R['G_ortho'] is not None:
            if matrix_is_complex(R['G_ortho']):
                RL += matrix_to_component_list(R['G_ortho'], _hat, _hat, r'\hat{G}')
            else:
                RL.append(r'\begin{equation}')
                RL.append(r'  \hat{G}_{\hat{a}\hat{b}} = ' + matrix_to_latex(R['G_ortho']))
                RL.append(r'\end{equation}')
        else:
            RL.append(r'\textit{(Einstein tensor projection unavailable.)}')

    # =========================================================================
    # SECTION 9 — ENERGY CONDITIONS
    # =========================================================================
    RL.append(r'\section{Energy Conditions}')
    if R['energy_conditions'] is not None:
        ec = R['energy_conditions']
        RL.append(
            r'From $G_{\hat{a}\hat{b}} = 8\pi T_{\hat{a}\hat{b}}$ the '
            r'stress-energy components in the orthonormal frame are:'
        )

        RL.append(r'\subsection{Stress-Energy Components}')
        ec_lines = []
        ec_lines += split_long_equation(r'\rho', lat(ec['rho']), inside_tcolorbox=True)
        for i, p in enumerate(ec['pressures']):
            ec_lines += split_long_equation(f'p_{i+1}', lat(p), inside_tcolorbox=True)
        RL += make_tcolorbox('Physical stress-energy', ec_lines, 'boxblue')

        RL.append(r'\subsection{Null Energy Condition (NEC)}')
        RL.append(
            r'NEC requires $\rho + p_i \geq 0$ for all principal pressures $p_i$. '
            r'Violation implies exotic (negative-energy) matter.'
        )
        for i, nec_expr in enumerate(ec['NEC']):
            RL += split_long_equation(rf'\rho + p_{i+1}', lat(nec_expr))

        RL.append(r'\subsection{Weak Energy Condition (WEC)}')
        RL.append(r'WEC: $\rho \geq 0$ (energy density non-negative) and NEC.')

        RL.append(r'\subsection{Strong Energy Condition (SEC)}')
        RL.append(r'SEC: $\rho + \sum_i p_i \geq 0$.')
        RL += split_long_equation(r'\rho + \textstyle\sum_i p_i', lat(ec['SEC']))

        RL.append(r'\subsection{Dominant Energy Condition (DEC)}')
        RL.append(
            r'DEC: $\rho \geq |p_i|$ — energy flows causally, '
            r'no superluminal energy transport.'
        )
        RL.append(
            r'\textit{(The DEC requires knowledge of the sign of $\rho$ relative '
            r'to $|p_i|$; substitute explicit parameter values to evaluate.)}'
        )
    else:
        RL.append(
            r'\textit{Energy conditions require an orthonormal tetrad. '
            r'Set \texttt{COMPUTE\_TETRAD = True} (or supply \texttt{e\_tetrad} manually) '
            r'in Section~1 to enable this section.}'
        )

    # =========================================================================
    # SECTION 10 — GEODESIC EQUATIONS
    # =========================================================================
    RL.append(r'\section{Geodesic Equations}')
    RL.append(
        r'The geodesic equation for an affinely parameterised curve $x^\mu(\lambda)$ is'
    )
    RL.append(r'\begin{equation}')
    RL.append(
        r'  \frac{d^2 x^\mu}{d\lambda^2} + '
        r'\christoffel{\mu}{\alpha}{\beta}\,\dot{x}^\alpha\,\dot{x}^\beta = 0,'
        r'\qquad \dot{x}^\mu \equiv \frac{dx^\mu}{d\lambda}.'
    )
    RL.append(r'\end{equation}')

    if R['geodesics'] is not None:
        RL.append(r'Each component written as $\ddot{x}^\mu = \ldots\,:$')
        for (mu, accel, u_syms) in R['geodesics']:
            coord_sym = coords[mu]
            lhs = r'\ddot{' + coord_latex(coord_sym) + r'}'
            if cancel(accel) == S.Zero:
                RL.append(
                    f'${lhs} = 0$ '
                    rf'(geodesic in the ${coord_latex(coord_sym)}$-direction is free).\\'
                )
            else:
                # Replace abstract velocity symbols in the expression
                rhs_latex = lat(accel)
                # Map SymPy printed dot_x symbols to LaTeX \dot{x}
                for i, c in enumerate(coords):
                    raw = f'dot_{{{coord_latex(c)}}}'
                    pretty = r'\dot{' + coord_latex(c) + r'}'
                    rhs_latex = rhs_latex.replace(raw, pretty)
                RL += split_long_equation(lhs, f'-({rhs_latex})')

    # Killing vectors / conserved quantities
    if R.get('killing'):
        RL.append(r'\subsection{Conserved Quantities (Killing Vectors)}')
        RL.append(
            r'Coordinates that do not appear in $g_{\mu\nu}$ give rise to '
            r'Killing vectors and conserved momenta along geodesics:'
        )
        RL.append(r'\begin{itemize}')
        for (i, cyc_coord) in R['killing']:
            cname = coord_latex(cyc_coord)
            RL.append(
                r'\item $\partial/\partial ' + cname + r'$ is a Killing vector. '
                r'Conserved quantity: $p_{' + cname + r'} = g_{'
                + cname + r'\nu}\,\dot{x}^\nu = \mathrm{const}$.'
            )
        RL.append(r'\end{itemize}')

    # =========================================================================
    # SECTION 11 — CONSERVATION AND CONSISTENCY CHECKS
    # =========================================================================
    RL.append(r'\section{Conservation and Consistency Checks}')

    # Bianchi identity
    RL.append(r'\subsection{Contracted Bianchi Identity: $\nabla_\mu G^{\mu\nu} = 0$}')
    RL.append(
        r'This is a geometric identity that guarantees stress-energy conservation '
        r'$\nabla_\mu T^{\mu\nu} = 0$ via the field equations.'
    )
    bianchi = R['bianchi']
    all_zero_b = all(cancel(b) == S.Zero for b in bianchi)
    if all_zero_b:
        RL += make_tcolorbox(
            r'Bianchi Identity',
            [r'\textbf{VERIFIED:} $\nabla_\mu G^{\mu\nu} = 0$ for all $\nu$. '
             r'The Einstein tensor is divergence-free. $\checkmark$'],
            'boxgreen'
        )
    else:
        RL += make_tcolorbox(
            r'Bianchi Identity --- Simplification Incomplete',
            [
                r'\textbf{Note:} SymPy could not reduce all components of '
                r'$\nabla_\mu G^{\mu\nu}$ to zero using \texttt{cancel()} '
                r'followed by \texttt{simplify()}. '
                r'This is \emph{not necessarily a code error} --- it is a '
                r'known limitation of SymPy for metrics containing unspecified '
                r'symbolic functions such as $B(r)$, $\beta(r)$, $f(r)$, etc. '
                r'\\[4pt]'
                r'\textbf{What this means physically:} The Bianchi identity '
                r'$\nabla_\mu G^{\mu\nu}=0$ is guaranteed to hold for '
                r'\emph{any} smooth metric by the second Bianchi identity for '
                r'the Riemann tensor --- it is a mathematical identity, not '
                r'an equation of motion. '
                r'If the residuals below are large symbolic expressions '
                r'involving derivatives of abstract metric functions, they '
                r'almost certainly vanish but SymPy lacks a simplification '
                r'rule to confirm it automatically. '
                r'\\[4pt]'
                r'\textbf{What to do:} '
                r'(1)~Substitute a \emph{specific} form of every metric '
                r'function (e.g.\ set $B(r)=1$ and $\beta(r)=\sqrt{2M/r}$ '
                r'for Schwarzschild-PG) --- the residuals will collapse to '
                r'zero numerically. '
                r'(2)~If the residuals contain only small \emph{rational} '
                r'numbers (not long symbolic sums), that would indicate a '
                r'genuine computation error and should be reported.'
            ],
            'boxred'
        )
        for nu, val in enumerate(bianchi):
            if cancel(val) != S.Zero:
                lhs_b = r'\nabla_\mu G^{\mu ' + idx_lat(coords, nu) + r'}'
                RL += split_long_equation(lhs_b, lat(val))

    # Trace consistency
    RL.append(r'\subsection{Ricci Scalar Trace Consistency}')
    RL.append(r'Verify $g^{\mu\nu} R_{\mu\nu} = \mathcal{R}$ (cross-check).')
    tc = R.get('trace_check', S.Zero)
    if cancel(tc) == S.Zero:
        RL += make_tcolorbox(
            'Trace Check',
            [r'\textbf{VERIFIED:} $g^{\mu\nu} R_{\mu\nu} = \mathcal{R}$. $\checkmark$'],
            'boxgreen'
        )
    else:
        RL += make_tcolorbox(
            'Trace Check — WARNING',
            [r'\textbf{Residual:} ' + lat(tc)],
            'boxred'
        )

    # Metric inverse check
    RL.append(r'\subsection{Metric Inverse Identity}')
    RL.append(r'Verify $g^{\mu\rho}\,g_{\rho\nu} = \delta^\mu{}_\nu$.')
    product_check = R['ginv'] * R['g']
    id_residual = product_check - eye(dim)
    max_res = max(abs(cancel(id_residual[i, j]))
                  for i in range(dim) for j in range(dim))
    if max_res == S.Zero:
        RL += make_tcolorbox(
            'Metric Inverse',
            [r'\textbf{VERIFIED:} $g^{\mu\rho}\,g_{\rho\nu} = \delta^\mu{}_\nu$. $\checkmark$'],
            'boxgreen'
        )
    else:
        RL.append(r'\textit{Max residual entry: }' + lat(max_res))

    RL += make_latex_footer()
    return RL


# ==============================================================================