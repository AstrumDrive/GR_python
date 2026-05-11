# -*- coding: utf-8 -*-
"""
Interactive help and executable examples for GR_python.

Use from a terminal:
    python gr_help.py
    python gr_help.py topics
    python gr_help.py metrics
    python gr_help.py example fast_minkowski_run
    python gr_help.py validate

Use from Spyder, Colab, or another Python session:
    from gr_help import gr_help, validate_examples
    gr_help()
    gr_help("examples")
    validate_examples()
"""

from collections import OrderedDict
import sys

import sympy as sp
from sympy import Function, Matrix, pi, sin, symbols


TOPICS = OrderedDict(
    [
        (
            "start",
            {
                "title": "Start here",
                "body": [
                    "1. Pick a metric key in gr_main.py, usually METRIC_KEY = 'schwarzschild'.",
                    "2. For a quick first run set FAST_MODE = True.",
                    "3. Run: python gr_calculator.py",
                    "4. Outputs are gr_report.tex and, if pdflatex exists, gr_report.pdf.",
                    "5. For notebook use, open GR_python_colab/GR_Colab.ipynb and run cells in order.",
                ],
            },
        ),
        (
            "metrics",
            {
                "title": "Metric selection",
                "body": [
                    "Use gr_metric_library.list_builtin_metric_keys() to see the built-in registry.",
                    "Use gr_metric_library.select_metric(key, coords) to retrieve a metric config.",
                    "Use METRIC_KEY = 'custom' plus CUSTOM_METRIC_CONFIG for a one-off metric.",
                    "Use METRIC_KEY = 'warp_doc_variant_b_alpha' for the full-spatial VdB/PG metric with alpha(r).",
                ],
            },
        ),
        (
            "flags",
            {
                "title": "Computation flags",
                "body": [
                    "FAST_MODE=True skips heavy invariants such as Weyl and Kretschmann.",
                    "COMPUTE_TETRAD=True enables the orthonormal frame and energy-condition block.",
                    "COMPUTE_HORIZONS=True enables horizon/static-limit diagnostics.",
                    "COMPUTE_ADM31=True enables 3+1 ADM diagnostics.",
                    "COMPUTE_MATTER=True enables scalar/Maxwell/spin-2 matter modules.",
                    "COMPUTE_GEODESIC_NUM=True enables numerical geodesic integration.",
                ],
            },
        ),
        (
            "tetrad",
            {
                "title": "Tetrad convention",
                "body": [
                    "Diagonal metrics use the canonical coordinate-aligned static tetrad.",
                    "Metrics with g_{0i} shift terms use the ADM/Eulerian tetrad adapted to t=const slices.",
                    "User-supplied tetrads must be matrices; do not set e_tetrad=True.",
                    "The code verifies g_{mu nu} e^mu_a e^nu_b = eta_ab before projection.",
                ],
            },
        ),
        (
            "reports",
            {
                "title": "Reports",
                "body": [
                    "gr_latex.assemble_report() builds the main symbolic LaTeX report.",
                    "gr_warp.compare_document_formulas() compares a symbolic run against document formulas.",
                    "Generated gr_report.tex/pdf are local outputs and intentionally ignored by Git.",
                ],
            },
        ),
        (
            "validation",
            {
                "title": "Validation scripts",
                "body": [
                    "python validate_vdb_alpha_document.py checks the generalized VdB alpha document identities.",
                    "python validate_magic_beta_document.py checks the magic beta / positive-energy identities.",
                    "python gr_help.py validate runs the lightweight examples in this help system.",
                ],
            },
        ),
    ]
)


EXAMPLES = OrderedDict(
    [
        (
            "list_metrics",
            {
                "title": "List built-in metrics",
                "code": """from gr_metric_library import list_builtin_metric_keys

print(list_builtin_metric_keys())
""",
                "expect": "A sorted list including 'schwarzschild' and 'warp_doc_variant_b_alpha'.",
            },
        ),
        (
            "select_schwarzschild",
            {
                "title": "Select a built-in metric",
                "code": """from sympy import symbols
from gr_metric_library import select_metric

t, r, theta, phi = symbols('t r theta phi', real=True)
cfg = select_metric('schwarzschild', [t, r, theta, phi])
print(cfg['metric_name'])
print(cfg['g_metric'])
""",
                "expect": "A metric config dictionary with g_metric, g_inv_metric, e_tetrad, and metadata.",
            },
        ),
        (
            "fast_minkowski_run",
            {
                "title": "Run the symbolic pipeline in fast mode",
                "code": """import sympy as sp
import gr_main as gm
from gr_metric_library import select_metric

t, r, theta, phi = sp.symbols('t r theta phi', real=True)
coords = [t, r, theta, phi]
cfg = select_metric('minkowski_spherical', coords)
results = gm.run_computations(
    cfg['g_metric'], coords, 4,
    compute_weyl_flag=False,
    compute_kretschmann_flag=False,
    compute_geodesics_flag=False,
    compute_killing_flag=False,
    compute_tetrad_flag=True,
    fast_mode=True,
    compute_horizons_flag=False,
    compute_penrose_flag=False,
)
print(results['R_scalar'])
print(results['tetrad_method'], results['tetrad_verified'])
""",
                "expect": "R_scalar = 0, tetrad_method = diagonal, tetrad_verified = True.",
            },
        ),
        (
            "horizons_schwarzschild",
            {
                "title": "Find Schwarzschild horizons",
                "code": """import sympy as sp
from gr_metric_library import select_metric
from gr_horizons import find_horizons

t, r, theta, phi = sp.symbols('t r theta phi', real=True)
M = sp.symbols('M', positive=True)
coords = [t, r, theta, phi]
cfg = select_metric('schwarzschild', coords, {'M': M})
g = cfg['g_metric']
h = find_horizons(g, g.inv(), coords)
print(h['horizon_roots'])
""",
                "expect": "The event-horizon root [2*M].",
            },
        ),
        (
            "matter_zero_scalar",
            {
                "title": "Compute scalar-field stress energy",
                "code": """import sympy as sp
from gr_metric_library import select_metric
from gr_matter import compute_scalar_stress_energy

t, r, theta, phi = sp.symbols('t r theta phi', real=True)
coords = [t, r, theta, phi]
cfg = select_metric('minkowski_spherical', coords)
matter = compute_scalar_stress_energy(sp.Integer(0), cfg['g_metric'], cfg['g_metric'].inv(), coords)
print(matter['T_cov'])
""",
                "expect": "The zero scalar field has a zero stress-energy tensor.",
            },
        ),
        (
            "vdb_alpha_identities",
            {
                "title": "Inspect generalized VdB alpha identities",
                "code": """import sympy as sp
from gr_warp import document_vdb_alpha_formulas

r = sp.symbols('r', positive=True)
alpha = sp.Function('alpha', positive=True)(r)
B = sp.Function('B', positive=True)(r)
beta = sp.Function('beta')(r)
formulas = document_vdb_alpha_formulas(r, alpha, B, beta)
print(formulas['alpha_magic'])
print(formulas['j_r'])
""",
                "expect": "The magic lapse and the GR_python-sign radial flux formula.",
            },
        ),
        (
            "penrose_keys",
            {
                "title": "List Penrose diagram templates",
                "code": """from gr_penrose import list_penrose_spacetimes

print(list_penrose_spacetimes())
""",
                "expect": "A list including minkowski, schwarzschild, kerr, de_sitter, and anti_de_sitter.",
            },
        ),
    ]
)


def _print_lines(lines):
    for line in lines:
        print(line)


def _print_topic(topic):
    data = TOPICS[topic]
    print(f"\n{data['title']}")
    print("-" * len(data["title"]))
    _print_lines(data["body"])


def _print_example(name):
    data = EXAMPLES[name]
    print(f"\n{name}: {data['title']}")
    print("-" * (len(name) + len(data["title"]) + 2))
    print(data["code"].rstrip())
    print("\nExpected:")
    print(data["expect"])


def gr_help(topic=None):
    """
    Print a compact help topic.

    Parameters
    ----------
    topic : str or None
        None, 'topics', 'examples', one topic key, or 'example:<name>'.
    """
    if topic is None:
        print("GR_python help")
        print("==============")
        print("Use gr_help('topics') to list topics.")
        print("Use gr_help('examples') to list runnable examples.")
        print("Use gr_help('example:fast_minkowski_run') to print one example.")
        print("Terminal equivalents: python gr_help.py topics | examples | validate")
        return

    if topic == "topics":
        print("Available help topics:")
        for key, data in TOPICS.items():
            print(f"  {key:12s} {data['title']}")
        return

    if topic == "examples":
        print("Available runnable examples:")
        for key, data in EXAMPLES.items():
            print(f"  {key:24s} {data['title']}")
        return

    if topic.startswith("example:"):
        name = topic.split(":", 1)[1]
        if name not in EXAMPLES:
            raise KeyError(f"Unknown example {name!r}. Use gr_help('examples').")
        _print_example(name)
        return

    if topic not in TOPICS:
        raise KeyError(f"Unknown help topic {topic!r}. Use gr_help('topics').")
    _print_topic(topic)


def _check(label, condition):
    print(f"[{'OK' if condition else 'FAIL'}] {label}")
    return bool(condition)


def validate_examples():
    """
    Run lightweight smoke checks for the examples documented in this module.

    The intent is not to re-run every expensive symbolic module. It verifies that
    the beginner-facing commands still import, execute, and produce the expected
    structural results.
    """
    from gr_horizons import find_horizons
    from gr_matter import compute_scalar_stress_energy
    from gr_metric_library import list_builtin_metric_keys, select_metric
    from gr_penrose import list_penrose_spacetimes
    from gr_warp import document_vdb_alpha_formulas
    import gr_main as gm

    t, r, theta, phi = symbols("t r theta phi", real=True)
    coords = [t, r, theta, phi]
    checks = []

    keys = list_builtin_metric_keys()
    checks.append(_check("metric registry exposes schwarzschild", "schwarzschild" in keys))
    checks.append(_check("metric registry exposes warp_doc_variant_b_alpha", "warp_doc_variant_b_alpha" in keys))

    cfg = select_metric("schwarzschild", coords)
    checks.append(_check("select_metric returns Schwarzschild metadata", cfg["metric_name"] == "Schwarzschild"))
    checks.append(_check("Schwarzschild metric is 4x4", cfg["g_metric"].shape == (4, 4)))

    minkowski = select_metric("minkowski_spherical", coords)
    results = gm.run_computations(
        minkowski["g_metric"],
        coords,
        4,
        compute_weyl_flag=False,
        compute_kretschmann_flag=False,
        compute_geodesics_flag=False,
        compute_killing_flag=False,
        compute_tetrad_flag=True,
        fast_mode=True,
        compute_horizons_flag=False,
        compute_penrose_flag=False,
    )
    checks.append(_check("fast Minkowski run has R_scalar = 0", sp.simplify(results["R_scalar"]) == 0))
    checks.append(_check("automatic diagonal tetrad verifies", results["tetrad_verified"] is True))

    M = symbols("M", positive=True)
    schwarzschild = select_metric("schwarzschild", coords, {"M": M})
    horizons = find_horizons(schwarzschild["g_metric"], schwarzschild["g_metric"].inv(), coords)
    checks.append(_check("Schwarzschild horizon root is 2*M", horizons["horizon_roots"] == [2 * M]))

    matter = compute_scalar_stress_energy(sp.Integer(0), minkowski["g_metric"], minkowski["g_metric"].inv(), coords)
    checks.append(_check("zero scalar stress tensor is zero", matter["T_cov"] == sp.zeros(4)))

    alpha = Function("alpha", positive=True)(r)
    B = Function("B", positive=True)(r)
    beta = Function("beta")(r)
    formulas = document_vdb_alpha_formulas(r, alpha, B, beta)
    checks.append(_check("VdB alpha formulas expose alpha_magic", formulas["alpha_magic"].has(B)))
    sign_residual = sp.simplify(
        sp.cancel(8 * pi * formulas["j_r"] + 2 * beta * formulas["V_tilde"] / (alpha * B))
    )
    checks.append(_check("VdB radial flux uses GR_python sign", sign_residual == 0))

    penrose_keys = list_penrose_spacetimes()
    checks.append(_check("Penrose templates include schwarzschild", "schwarzschild" in penrose_keys))

    passed = sum(checks)
    total = len(checks)
    print(f"\nHelp example checks passed: {passed}/{total}")
    return passed == total


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        gr_help()
        return 0

    command = argv[0]
    if command in {"-h", "--help", "help"}:
        gr_help()
        return 0
    if command == "topics":
        gr_help("topics")
        return 0
    if command == "examples":
        gr_help("examples")
        return 0
    if command == "example":
        if len(argv) < 2:
            print("Usage: python gr_help.py example <example_name>")
            return 2
        gr_help(f"example:{argv[1]}")
        return 0
    if command == "validate":
        return 0 if validate_examples() else 1
    if command in TOPICS:
        gr_help(command)
        return 0

    print(f"Unknown help command: {command!r}")
    print("Try: python gr_help.py topics")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
