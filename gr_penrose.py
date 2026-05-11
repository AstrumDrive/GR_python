# -*- coding: utf-8 -*-
"""
gr_penrose.py — Penrose-Carter Conformal Diagrams

Generates qualitative Penrose-Carter (conformal) diagrams for standard
spacetimes using matplotlib.  Each diagram is drawn in (T, X) conformal
coordinates where the entire spacetime is mapped to a finite region.

Spacetimes implemented
-----------------------
  • Minkowski         — full conformal diamond
  • Schwarzschild     — Kruskal-Szekeres maximal extension (4 regions)
  • Reissner-Nordström — RN extension with Cauchy horizon
  • Kerr              — Boyer-Lindquist Carter diagram
  • de Sitter         — static-patch conformal square
  • Anti-de Sitter    — global AdS (infinite tower suppressed)
  • Generic 1-horizon — simple Schwarzschild-like template

Usage
-----
    fig = draw_penrose_diagram('schwarzschild', M=1.0, output_path='penrose.pdf')

Notes
-----
The diagrams are qualitative / schematic: exact coordinate transformations to
conformal coordinates are provided for reference in comments, but the diagram
geometry is drawn analytically for maximum clarity and without numerical
integration of null geodesics (which would be needed for pixel-level accuracy).

References
----------
• Carter (1966), Phys. Rev. 141, 1242
• Hawking & Ellis, "The Large Scale Structure of Space-Time" (1973) App. B
• Wald, "General Relativity" (1984) App. D
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch, PathPatch
    from matplotlib.path import Path
    import matplotlib.lines as mlines
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

from gr_tensors import progress


# ==============================================================================
# COLOUR PALETTE
# ==============================================================================

_COLORS = {
    'region_ext':   '#ddeeff',   # exterior region (pale blue)
    'region_int':   '#ffe5e5',   # interior / future (pale red)
    'region_past':  '#ffe5e5',   # past interior
    'region_par':   '#e8f5e9',   # parallel exterior (pale green)
    'region_ergo':  '#fff9c4',   # ergosphere (pale yellow)
    'horizon':      '#cc0000',   # event horizon (red)
    'cauchy':       '#ff6600',   # Cauchy horizon (orange)
    'singularity':  '#222222',   # singularity (near-black)
    'infinity_sp':  '#2266cc',   # spacelike infinity (blue)
    'infinity_nu':  '#22aa44',   # null infinity (green)
    'infinity_tm':  '#aa22cc',   # timelike infinity (purple)
    'axis':         '#888888',
}


# ==============================================================================
# LOW-LEVEL DRAWING HELPERS
# ==============================================================================

def _draw_diamond(ax, vertices, facecolor, edgecolor='none', zorder=1, alpha=0.8):
    """Fill a polygon defined by a list of (X, T) vertex pairs."""
    xs = [v[0] for v in vertices] + [vertices[0][0]]
    ts = [v[1] for v in vertices] + [vertices[0][1]]
    ax.fill(xs, ts, facecolor=facecolor, edgecolor=edgecolor,
            zorder=zorder, alpha=alpha)


def _draw_line(ax, p1, p2, color, lw=2, ls='-', zorder=3, label=None):
    xs = [p1[0], p2[0]]
    ts = [p1[1], p2[1]]
    ax.plot(xs, ts, color=color, lw=lw, ls=ls, zorder=zorder, label=label)


def _draw_singularity(ax, p1, p2, zorder=4):
    """Draw a spacelike singularity as a thick zigzag line."""
    xs = np.linspace(p1[0], p2[0], 60)
    ts = np.linspace(p1[1], p2[1], 60)
    amp = 0.04 * abs(p2[0] - p1[0])
    ts_zig = ts + amp * np.sin(np.linspace(0, 8 * np.pi, 60))
    ax.plot(xs, ts_zig, color=_COLORS['singularity'], lw=3, zorder=zorder)


def _label(ax, X, T, text, fontsize=9, color='black', ha='center', va='center'):
    ax.text(X, T, text, fontsize=fontsize, color=color,
            ha=ha, va=va, zorder=6,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))


def _add_null_rays(ax, n=4, x_range=(-1, 1), t_range=(-1, 1), color='#aaaaaa', alpha=0.3):
    """Draw a background grid of ingoing and outgoing null rays."""
    xs = np.linspace(x_range[0], x_range[1], n)
    for x0 in xs:
        # outgoing: T = X − x0
        X_vals = np.array([x_range[0], x_range[1]])
        T_vals = X_vals - x0
        mask = (T_vals >= t_range[0]) & (T_vals <= t_range[1])
        if mask.any():
            ax.plot(X_vals, T_vals, color=color, lw=0.5, alpha=alpha, zorder=0)
        # ingoing: T = −X + x0
        T_vals2 = -X_vals + x0
        mask2 = (T_vals2 >= t_range[0]) & (T_vals2 <= t_range[1])
        if mask2.any():
            ax.plot(X_vals, T_vals2, color=color, lw=0.5, alpha=alpha, zorder=0)


def _setup_axes(ax, title, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2)):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=11, pad=8)


# ==============================================================================
# INDIVIDUAL PENROSE DIAGRAMS
# ==============================================================================

def _draw_minkowski(ax):
    """Minkowski spacetime — full conformal diamond."""
    # Conformal map: T = arctan(t+r) + arctan(t-r), X = arctan(t+r) − arctan(t-r)
    # Range: |T| + |X| ≤ π, i.e. diamond with vertices at (0,±π), (±π,0)
    # Normalise to unit diamond
    v_top   = (0.0,  1.0)
    v_bot   = (0.0, -1.0)
    v_right = (1.0,  0.0)
    v_left  = (-1.0, 0.0)

    _draw_diamond(ax, [v_right, v_top, v_left, v_bot], _COLORS['region_ext'], zorder=1)

    # Null boundaries (I^+ and I^−)
    _draw_line(ax, v_right, v_top,  _COLORS['infinity_nu'], lw=2, label=r'$\mathcal{I}^+$')
    _draw_line(ax, v_left,  v_top,  _COLORS['infinity_nu'], lw=2)
    _draw_line(ax, v_right, v_bot,  _COLORS['infinity_nu'], lw=2, ls='--', label=r'$\mathcal{I}^-$')
    _draw_line(ax, v_left,  v_bot,  _COLORS['infinity_nu'], lw=2, ls='--')

    # Corner points
    ax.plot(*v_top,   'o', color=_COLORS['infinity_tm'], ms=6, zorder=5)
    ax.plot(*v_bot,   'o', color=_COLORS['infinity_tm'], ms=6, zorder=5)
    ax.plot(*v_right, 's', color=_COLORS['infinity_sp'], ms=6, zorder=5)
    ax.plot(*v_left,  's', color=_COLORS['infinity_sp'], ms=6, zorder=5)

    _add_null_rays(ax, n=6)

    _label(ax,  0.0,  0.0, r'Minkowski',        fontsize=9)
    _label(ax,  0.0,  1.05, r'$i^+$',           fontsize=8, color=_COLORS['infinity_tm'])
    _label(ax,  0.0, -1.05, r'$i^-$',           fontsize=8, color=_COLORS['infinity_tm'])
    _label(ax,  1.1,  0.0,  r'$i^0$',           fontsize=8, color=_COLORS['infinity_sp'])
    _label(ax, -1.1,  0.0,  r'$i^0$',           fontsize=8, color=_COLORS['infinity_sp'])
    _label(ax,  0.6,  0.6,  r'$\mathcal{I}^+$', fontsize=8, color=_COLORS['infinity_nu'])
    _label(ax,  0.6, -0.6,  r'$\mathcal{I}^-$', fontsize=8, color=_COLORS['infinity_nu'])

    _setup_axes(ax, 'Minkowski Spacetime')


def _draw_schwarzschild(ax, label_M=True):
    """
    Kruskal-Szekeres maximal extension of Schwarzschild.
    Four regions: I (exterior), II (future), III (parallel exterior), IV (past).
    """
    # Corners of the Kruskal diamond
    v_top   = (0.0,  1.0)   # future singularity centre
    v_bot   = (0.0, -1.0)   # past singularity centre
    v_right = (1.0,  0.0)   # right spatial infinity i^0
    v_left  = (-1.0, 0.0)   # left spatial infinity i^0

    # Region I: exterior (right wedge)
    _draw_diamond(ax, [(0,0), v_right, (1,1), (0,1)],  # right upper triangle
                  _COLORS['region_ext'], zorder=1)
    _draw_diamond(ax, [(0,0), v_right, (1,-1), (0,-1)],  # right lower
                  _COLORS['region_ext'], zorder=1)
    # Region II: future interior
    _draw_diamond(ax, [(0,0), (1,1), v_top, (-1,1)],
                  _COLORS['region_int'], zorder=1)
    # Region III: parallel exterior (left)
    _draw_diamond(ax, [(0,0), v_left, (-1,1), (0,1)],
                  _COLORS['region_par'], zorder=1)
    _draw_diamond(ax, [(0,0), v_left, (-1,-1), (0,-1)],
                  _COLORS['region_par'], zorder=1)
    # Region IV: past interior
    _draw_diamond(ax, [(0,0), (1,-1), v_bot, (-1,-1)],
                  '#e8f4e8', zorder=1)

    # Event horizons (H^+ and H^−)
    _draw_line(ax, (0,0), (1,1),   _COLORS['horizon'], lw=2, label=r'$\mathcal{H}^+$')
    _draw_line(ax, (0,0), (-1,1),  _COLORS['horizon'], lw=2)
    _draw_line(ax, (0,0), (1,-1),  _COLORS['horizon'], lw=2, ls='--', label=r'$\mathcal{H}^-$')
    _draw_line(ax, (0,0), (-1,-1), _COLORS['horizon'], lw=2, ls='--')

    # Singularities
    _draw_singularity(ax, (-1, 1), (1, 1))
    _draw_singularity(ax, (-1,-1), (1,-1))

    # Null infinities
    _draw_line(ax, v_right, (1,1),   _COLORS['infinity_nu'], lw=1.5)
    _draw_line(ax, v_right, (1,-1),  _COLORS['infinity_nu'], lw=1.5, ls='--')
    _draw_line(ax, v_left,  (-1,1),  _COLORS['infinity_nu'], lw=1.5)
    _draw_line(ax, v_left,  (-1,-1), _COLORS['infinity_nu'], lw=1.5, ls='--')

    _add_null_rays(ax, n=5)

    _label(ax,  0.6,  0.0, 'I\n(exterior)',     fontsize=8, color='#003366')
    _label(ax, -0.6,  0.0, 'III\n(parallel)',   fontsize=8, color='#005500')
    _label(ax,  0.0,  0.7, 'II\n(future)',      fontsize=8, color='#660000')
    _label(ax,  0.0, -0.7, 'IV\n(past)',        fontsize=8, color='#004400')
    _label(ax,  0.0,  1.08, 'singularity',      fontsize=7, color=_COLORS['singularity'])
    _label(ax,  0.0, -1.08, 'singularity',      fontsize=7, color=_COLORS['singularity'])
    _label(ax,  1.0,  0.0,  r'$i^0$',           fontsize=7, color=_COLORS['infinity_sp'])
    _label(ax, -1.0,  0.0,  r'$i^0$',           fontsize=7, color=_COLORS['infinity_sp'])

    _setup_axes(ax, 'Schwarzschild (Kruskal extension)', xlim=(-1.3,1.3), ylim=(-1.3,1.3))


def _draw_reissner_nordstrom(ax):
    """
    Reissner-Nordström — shows two levels of the infinite tower.
    Outer horizon H^±, inner (Cauchy) horizon CH^±.
    """
    # Level 0: exterior region I
    _draw_diamond(ax, [(0,0), (1,0.5), (1,1), (0.5,1)],
                  _COLORS['region_ext'], zorder=1)
    # Between horizons region II
    _draw_diamond(ax, [(0,0), (1,0.5), (0.5,1), (0,1)],
                  _COLORS['region_int'], zorder=1)
    # Interior region III (beyond Cauchy horizon)
    _draw_diamond(ax, [(0,1), (0.5,1), (0.5,1.5), (0,1.5)],
                  '#f3e5f5', zorder=1)

    # Outer horizons
    _draw_line(ax, (0,0),   (1,0.5), _COLORS['horizon'], lw=2)
    _draw_line(ax, (0,0),   (-1,0.5), _COLORS['horizon'], lw=2)

    # Cauchy (inner) horizons
    _draw_line(ax, (0,1),   (0.5,1),  _COLORS['cauchy'], lw=2)
    _draw_line(ax, (0,1),   (-0.5,1), _COLORS['cauchy'], lw=2)

    # No spacelike singularity at the boundary in RN (it's timelike)
    _draw_line(ax, (0.5,1), (0.5,1.5), _COLORS['singularity'], lw=3)
    _draw_line(ax, (-0.5,1),(-0.5,1.5),_COLORS['singularity'], lw=3)

    # Labels
    _label(ax, 0.8, 0.25, 'I\n(exterior)',    fontsize=7, color='#003366')
    _label(ax, 0.4, 0.7,  'II\n(between)',    fontsize=7, color='#660000')
    _label(ax, 0.3, 1.25, 'III\n(interior)',  fontsize=7, color='#440044')
    _label(ax, 0.55,1.3,  'singularity',      fontsize=6, color=_COLORS['singularity'])

    _label(ax, 0.65, 0.45, r'$H^+$',  fontsize=7, color=_COLORS['horizon'])
    _label(ax, 0.3,  0.95, r'$CH^+$', fontsize=7, color=_COLORS['cauchy'])

    _setup_axes(ax, 'Reissner-Nordström (two levels)', xlim=(-1.1,1.1), ylim=(-0.3,1.8))


def _draw_kerr(ax):
    """
    Kerr Carter diagram — schematic with ergosphere region.
    """
    # Exterior region I
    _draw_diamond(ax, [(0,0), (1,0.5), (1,1), (0,1)],
                  _COLORS['region_ext'], zorder=1)
    # Ergosphere (shaded region within exterior, near horizon)
    _draw_diamond(ax, [(0,0), (0.5,0.25), (0.5,0.6), (0,0.4)],
                  _COLORS['region_ergo'], zorder=2, alpha=0.7)
    # Between horizons
    _draw_diamond(ax, [(0,0), (0.5,0.25), (0,0.5)],
                  _COLORS['region_int'], zorder=1)
    # Interior
    _draw_diamond(ax, [(0,0.5), (0.5,0.25), (0.5,0.6), (0,0.8)],
                  '#f3e5f5', zorder=1)

    # Outer horizon
    _draw_line(ax, (0,0), (1,0.5),  _COLORS['horizon'], lw=2)
    _draw_line(ax, (0,0), (-1,0.5), _COLORS['horizon'], lw=2)
    # Inner horizon
    _draw_line(ax, (0,0), (0.5,0.25),  _COLORS['cauchy'], lw=2, ls='--')
    _draw_line(ax, (0,0), (-0.5,0.25), _COLORS['cauchy'], lw=2, ls='--')
    # Static limit (ergosphere boundary)
    _draw_line(ax, (0,0), (0.75,0.5),  _COLORS['region_ergo'][:-2]+'00',
               color='goldenrod', lw=1.5, ls=':', label='static limit')

    _label(ax, 0.8, 0.6, 'I\n(exterior)', fontsize=7)
    _label(ax, 0.3, 0.3, 'ergo',          fontsize=6, color='#997700')
    _label(ax, 0.2, 0.15,'II',            fontsize=7, color='#660000')
    _label(ax, 0.25,0.55, 'III',          fontsize=7, color='#440044')
    _label(ax, 0.65,0.35, r'$H^+$ (outer)', fontsize=6, color=_COLORS['horizon'])
    _label(ax, 0.3, 0.05, r'$H^-$ (inner)', fontsize=6, color=_COLORS['cauchy'])

    _setup_axes(ax, 'Kerr (Carter diagram, schematic)', xlim=(-1.2,1.2), ylim=(-0.2,1.2))


def _draw_de_sitter(ax):
    """
    de Sitter spacetime — static patch conformal square.
    """
    # Conformal diagram: square with cosmological horizon diagonals
    v_tr = (1.0,  1.0)
    v_tl = (-1.0, 1.0)
    v_br = (1.0, -1.0)
    v_bl = (-1.0,-1.0)
    v_top = (0.0,  1.0)
    v_bot = (0.0, -1.0)

    # Left static patch
    _draw_diamond(ax, [(-1,0), v_tl, v_top, (0,0)], _COLORS['region_ext'], zorder=1)
    _draw_diamond(ax, [(-1,0), v_bl, v_bot, (0,0)], _COLORS['region_ext'], zorder=1)
    # Right static patch
    _draw_diamond(ax, [(1,0), v_tr, v_top, (0,0)], _COLORS['region_par'], zorder=1)
    _draw_diamond(ax, [(1,0), v_br, v_bot, (0,0)], _COLORS['region_par'], zorder=1)
    # Top/bottom cosmological regions
    _draw_diamond(ax, [v_top, v_tr, (0,0), v_tl], _COLORS['region_int'], zorder=1)
    _draw_diamond(ax, [v_bot, v_br, (0,0), v_bl], '#e8f5e9', zorder=1)

    # Cosmological horizons
    _draw_line(ax, (0,0), v_tr, _COLORS['horizon'], lw=2)
    _draw_line(ax, (0,0), v_tl, _COLORS['horizon'], lw=2)
    _draw_line(ax, (0,0), v_br, _COLORS['horizon'], lw=2, ls='--')
    _draw_line(ax, (0,0), v_bl, _COLORS['horizon'], lw=2, ls='--')

    # Boundaries (spacelike)
    _draw_line(ax, v_tl, v_tr, _COLORS['infinity_sp'], lw=2)
    _draw_line(ax, v_bl, v_br, _COLORS['infinity_sp'], lw=2, ls='--')
    _draw_line(ax, v_tl, v_bl, _COLORS['infinity_sp'], lw=2)
    _draw_line(ax, v_tr, v_br, _COLORS['infinity_sp'], lw=2)

    _add_null_rays(ax, n=5)

    _label(ax, -0.6, 0.0,  'Observer\n(left)',  fontsize=7)
    _label(ax,  0.6, 0.0,  'Observer\n(right)', fontsize=7)
    _label(ax,  0.0, 0.7,  'Expanding\nregion', fontsize=7, color='#660000')
    _label(ax,  0.0,-0.7,  'Contracting',       fontsize=7, color='#004400')
    _label(ax,  0.0, 1.1,  r'$\mathcal{I}^+$',  fontsize=8)
    _label(ax,  0.0,-1.1,  r'$\mathcal{I}^-$',  fontsize=8)
    _label(ax,  0.55, 0.55, r'$\mathcal{H}$',   fontsize=8, color=_COLORS['horizon'])

    _setup_axes(ax, 'de Sitter Spacetime', xlim=(-1.3,1.3), ylim=(-1.3,1.3))


def _draw_anti_de_sitter(ax):
    """
    Global Anti-de Sitter — timelike boundary (vertical strips).
    """
    # AdS: r goes from 0 (centre) to ∞ (boundary)
    # Conformal time: T ∈ (−π/2, π/2), spatial: χ ∈ [0, π/2]
    # Rectangle: X ∈ [0, 1], T ∈ [−1, 1]

    _draw_diamond(ax, [(0,-1),(1,-1),(1,1),(0,1)], _COLORS['region_ext'], zorder=1, alpha=0.6)

    # Timelike boundary (right edge)
    _draw_line(ax, (1,-1),(1,1), _COLORS['infinity_tm'], lw=3, label='timelike boundary')
    # Origin (axis)
    _draw_line(ax, (0,-1),(0,1), _COLORS['axis'], lw=2, ls='--', label='r = 0')
    # Spacelike top/bottom (Cauchy surfaces)
    _draw_line(ax, (0,1),(1,1),  _COLORS['infinity_sp'], lw=1.5, ls=':')
    _draw_line(ax, (0,-1),(1,-1),_COLORS['infinity_sp'], lw=1.5, ls=':')

    # Some null geodesics (bounce off boundary)
    for x0 in [0.0, 0.3, 0.6]:
        xs = [0, 1, 0, 1, 0]
        ts = [x0, x0+1, x0+2, x0+3, x0+4]
        xs_t = [x for x,t in zip(xs,ts) if -1<=t<=1]
        ts_t = [t for x,t in zip(xs,ts) if -1<=t<=1]
        if len(xs_t) >= 2:
            ax.plot(xs_t, ts_t, color='#aaaaaa', lw=0.8, alpha=0.5, zorder=0)

    _add_null_rays(ax, n=0)

    _label(ax, 0.5, 0.0,  'AdS interior',         fontsize=8)
    _label(ax, 1.12, 0.0, 'timelike\nboundary',    fontsize=7, color=_COLORS['infinity_tm'], ha='left')
    _label(ax, -0.05,0.0, 'r=0',                   fontsize=7, color=_COLORS['axis'], ha='right')

    _setup_axes(ax, 'Global Anti-de Sitter', xlim=(-0.3,1.5), ylim=(-1.3,1.3))


# ==============================================================================
# DISPATCH TABLE
# ==============================================================================

_DIAGRAM_REGISTRY = {
    'minkowski':          _draw_minkowski,
    'schwarzschild':      _draw_schwarzschild,
    'reissner_nordstrom': _draw_reissner_nordstrom,
    'kerr':               _draw_kerr,
    'de_sitter':          _draw_de_sitter,
    'anti_de_sitter':     _draw_anti_de_sitter,
}


# ==============================================================================
# PUBLIC API
# ==============================================================================

def draw_penrose_diagram(spacetime_key, output_path=None, show=False, ax=None, **kwargs):
    """
    Draw the Penrose-Carter conformal diagram for a named spacetime.

    Parameters
    ----------
    spacetime_key : str
        One of: 'minkowski', 'schwarzschild', 'reissner_nordstrom',
                'kerr', 'de_sitter', 'anti_de_sitter'
    output_path   : str or None — if given, save as PNG/PDF
    show          : bool — call plt.show()
    ax            : matplotlib Axes or None — if given, draw into existing axes
    **kwargs      : passed to the specific diagram function

    Returns
    -------
    fig : matplotlib.Figure or None (if matplotlib not available)
    """
    if not _MPL_OK:
        progress("  gr_penrose: matplotlib not available — cannot draw Penrose diagram.")
        return None

    key = spacetime_key.lower().strip()
    if key not in _DIAGRAM_REGISTRY:
        progress(f"  gr_penrose: Unknown spacetime '{key}'. "
                 f"Available: {sorted(_DIAGRAM_REGISTRY.keys())}")
        return None

    progress(f"Drawing Penrose-Carter diagram for: {key} ...")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))
    else:
        fig = ax.figure

    _DIAGRAM_REGISTRY[key](ax, **kwargs)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        progress(f"  Penrose diagram saved → {output_path}")

    if show:
        plt.show()

    return fig


def draw_all_penrose_diagrams(output_path=None, show=False):
    """
    Draw all six built-in Penrose diagrams in a 2×3 grid.

    Parameters
    ----------
    output_path : str or None
    show        : bool

    Returns
    -------
    fig : matplotlib.Figure or None
    """
    if not _MPL_OK:
        progress("  gr_penrose: matplotlib not available.")
        return None

    progress("Drawing all Penrose diagrams (2×3 panel)...")

    keys = ['minkowski', 'schwarzschild', 'reissner_nordstrom',
            'kerr', 'de_sitter', 'anti_de_sitter']
    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    axes_flat = axes.flatten()

    for ax, key in zip(axes_flat, keys):
        _DIAGRAM_REGISTRY[key](ax)

    plt.tight_layout(pad=2.0)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        progress(f"  All Penrose diagrams saved → {output_path}")

    if show:
        plt.show()

    return fig


def list_penrose_spacetimes():
    """Return list of keys for which Penrose diagrams are implemented."""
    return sorted(_DIAGRAM_REGISTRY.keys())
