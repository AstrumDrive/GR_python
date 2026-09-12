# -*- coding: utf-8 -*-
"""
Interactive help and executable examples for GR_python.

Use from a terminal:
    python gr_help.py
    python gr_help.py topics
    python gr_help.py metrics
    python gr_help.py baez
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
            "baez",
            {
                "title": "Baez tutorial core quantities",
                "body": [
                    "BAEZ_TUTORIAL_COVERAGE.md maps the quantities in John Baez's GR tutorial to the public API.",
                    "It includes metric contractions, causal classification, index raising/lowering, torsion, parallel transport, and geodesic deviation.",
                    "Run python validate_baez_tutorial.py for a focused symbolic and numerical smoke check.",
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
        (
            "inventory",
            {
                "title": "Public-function inventory",
                "body": [
                    "GUIA_COMANDOS_ES.md groups the public API by module with examples.",
                    "Use gr_help('examples') for the executable examples that are validated automatically.",
                    "Use python gr_help.py validate after edits to catch documentation/code drift.",
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
            "petrov_schwarzschild",
            {
                "title": "Classify Schwarzschild as Petrov type D",
                "code": """import sympy as sp
from gr_metric_library import select_metric
from gr_tensors import (
    compute_christoffel, compute_riemann, compute_ricci,
    compute_ricci_scalar, compute_weyl, compute_tetrad_adm,
)
from gr_petrov import build_np_tetrad, compute_weyl_scalars, classify_petrov

t, r, theta, phi = sp.symbols('t r theta phi', real=True)
M = sp.symbols('M', positive=True)
coords = [t, r, theta, phi]
cfg = select_metric('schwarzschild', coords, {'M': M})
g = cfg['g_metric']
ginv = cfg['g_inv_metric'] or g.inv()
Gamma = compute_christoffel(g, ginv, coords)
Riem = compute_riemann(Gamma, coords)
Ric = compute_ricci(Riem)
R = compute_ricci_scalar(Ric, ginv)
C = compute_weyl(Riem, Ric, g, ginv, R)
e_contra, *_ = compute_tetrad_adm(g, ginv, coords)
np_tetrad = build_np_tetrad(e_contra)
psi = compute_weyl_scalars(C, np_tetrad, g)
print(psi['Psi2'])
print(classify_petrov(psi)['type'])
""",
                "expect": "Psi2 = -M/r**3 and Petrov type D.",
            },
        ),
        (
            "horizons_kerr",
            {
                "title": "Find Kerr horizons and static limit",
                "code": """import sympy as sp
from gr_metric_library import select_metric
from gr_horizons import find_horizons

t, r, theta, phi = sp.symbols('t r theta phi', real=True)
M, a = sp.symbols('M a', positive=True)
coords = [t, r, theta, phi]
cfg = select_metric('kerr', coords, {'M': M, 'a': a})
g = cfg['g_metric']
ginv = cfg['g_inv_metric'] or g.inv()
h = find_horizons(g, ginv, coords)
print(h['horizon_roots'])
print(h['static_limit_roots'])
""",
                "expect": "Horizon roots [M + sqrt(M**2 - a**2), M - sqrt(M**2 - a**2)] and equatorial static limit [2*M].",
            },
        ),
        (
            "numeric_geodesic_schwarzschild",
            {
                "title": "Integrate a short Schwarzschild circular geodesic",
                "code": """import math
import sympy as sp
from gr_metric_library import select_metric
from gr_tensors import compute_christoffel
from gr_geodesic_numeric import (
    lambdify_christoffel, lambdify_metric,
    integrate_geodesic, check_conserved,
)

t, r, theta, phi = sp.symbols('t r theta phi', real=True)
M = sp.symbols('M', positive=True)
coords = [t, r, theta, phi]
cfg = select_metric('schwarzschild', coords, {'M': M})
g = cfg['g_metric']
ginv = cfg['g_inv_metric'] or g.inv()
Gamma = compute_christoffel(g, ginv, coords)
Gamma_num, remaining = lambdify_christoffel(Gamma, coords, parameter_subs={M: 1.0})
g_num, ginv_num = lambdify_metric(g, ginv, coords, parameter_subs={M: 1.0})
r0 = 10.0
u_t = 1.0 / math.sqrt(1.0 - 3.0 / r0)
u_phi = math.sqrt(1.0 / r0**3) * u_t
sol = integrate_geodesic(
    Gamma_num, [0.0, r0, math.pi/2, 0.0], [u_t, 0.0, 0.0, u_phi],
    (0.0, 50.0), rtol=1e-8, atol=1e-10, max_step=1.0,
)
diag = check_conserved(sol, g_num, ginv_num, killing_indices=[0, 3])
print(sol.success)
print(diag['norm_drift'])
""",
                "expect": "Successful RK45 integration with very small norm drift.",
            },
        ),
        (
            "maxwell_reissner",
            {
                "title": "Build Coulomb F_mn and Maxwell stress energy",
                "code": """import sympy as sp
from gr_metric_library import select_metric
from gr_matter import build_faraday_tensor, compute_maxwell_stress_energy

t, r, theta, phi = sp.symbols('t r theta phi', real=True)
Q = sp.symbols('Q', real=True)
coords = [t, r, theta, phi]
cfg = select_metric('reissner_nordstrom', coords, {'Q': Q})
g = cfg['g_metric']
ginv = cfg['g_inv_metric'] or g.inv()
F = build_faraday_tensor([Q/r, 0, 0, 0], coords)
T = compute_maxwell_stress_energy(F, g, ginv, coords)
print(F[0, 1], F[1, 0])
print(T['T_trace'])
""",
                "expect": "F_01 = Q/r**2, F_10 = -Q/r**2, and Maxwell trace = 0.",
            },
        ),
        (
            "adm_flrw",
            {
                "title": "Extract ADM variables for flat FLRW",
                "code": """import sympy as sp
from gr_metric_library import select_metric
from gr_tensors import compute_christoffel
from gr_adm31 import extract_adm_variables, compute_extrinsic_curvature

t, r, theta, phi = sp.symbols('t r theta phi', real=True)
coords = [t, r, theta, phi]
cfg = select_metric('frw_flat', coords)
g = cfg['g_metric']
ginv = cfg['g_inv_metric'] or g.inv()
Gamma = compute_christoffel(g, ginv, coords)
adm = extract_adm_variables(g, ginv, coords)
K = compute_extrinsic_curvature(g, ginv, Gamma, coords, adm)
print(adm['N'])
print(adm['beta_contra'])
print(sp.simplify(K['K_trace']))
""",
                "expect": "N = 1, beta^i = [0, 0, 0], and K = -3*a'(t)/a(t) in GR_python's ADM convention.",
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
        (
            "penrose_all",
            {
                "title": "Draw the 2x3 Penrose diagram panel",
                "code": """from gr_penrose import draw_all_penrose_diagrams

fig = draw_all_penrose_diagrams(output_path='all_penrose.pdf', show=False)
print(type(fig).__name__)
print(len(fig.axes))
""",
                "expect": "A matplotlib Figure with six axes, saved to all_penrose.pdf.",
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
    import math

    from gr_adm31 import extract_adm_variables, compute_extrinsic_curvature
    from gr_geodesic_numeric import (
        lambdify_christoffel, lambdify_metric, integrate_geodesic, check_conserved,
    )
    from gr_horizons import find_horizons
    from gr_matter import (
        build_faraday_tensor, compute_maxwell_stress_energy, compute_scalar_stress_energy,
    )
    from gr_metric_library import list_builtin_metric_keys, select_metric
    from gr_penrose import draw_all_penrose_diagrams, list_penrose_spacetimes
    from gr_petrov import build_np_tetrad, compute_weyl_scalars, classify_petrov, verify_np_tetrad
    from gr_tensors import (
        compute_christoffel, compute_riemann, compute_ricci,
        compute_ricci_scalar, compute_tetrad_adm, compute_weyl,
    )
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

    # Extended examples from the public-function inventory.
    g = schwarzschild["g_metric"]
    ginv = schwarzschild["g_inv_metric"] or g.inv()
    Gamma = compute_christoffel(g, ginv, coords)
    Riem = compute_riemann(Gamma, coords)
    Ric = compute_ricci(Riem)
    R_scalar = compute_ricci_scalar(Ric, ginv)
    C_weyl = compute_weyl(Riem, Ric, g, ginv, R_scalar)
    e_contra, *_ = compute_tetrad_adm(g, ginv, coords)
    np_tetrad = build_np_tetrad(e_contra)
    np_ok, _ = verify_np_tetrad(np_tetrad, g)
    psi = compute_weyl_scalars(C_weyl, np_tetrad, g)
    petrov = classify_petrov(psi)
    checks.append(_check("Schwarzschild NP tetrad verifies", np_ok is True))
    checks.append(_check("Schwarzschild Psi2 = -M/r^3", sp.simplify(psi["Psi2"] + M / r**3) == 0))
    checks.append(_check("Schwarzschild Petrov type is D", petrov["type"] == "D"))

    a = symbols("a", positive=True)
    kerr = select_metric("kerr", coords, {"M": M, "a": a})
    g_kerr = kerr["g_metric"]
    ginv_kerr = kerr["g_inv_metric"] or g_kerr.inv()
    kerr_h = find_horizons(g_kerr, ginv_kerr, coords)
    checks.append(_check(
        "Kerr horizons are r_+ and r_-",
        kerr_h["horizon_roots"] == [M + sp.sqrt(M**2 - a**2), M - sp.sqrt(M**2 - a**2)],
    ))
    checks.append(_check("Kerr equatorial static limit is 2*M", kerr_h["static_limit_roots"] == [2 * M]))

    Gamma_num, remaining = lambdify_christoffel(Gamma, coords, parameter_subs={M: 1.0})
    g_num, ginv_num = lambdify_metric(g, ginv, coords, parameter_subs={M: 1.0})
    r0 = 10.0
    u_t = 1.0 / math.sqrt(1.0 - 3.0 / r0)
    u_phi = math.sqrt(1.0 / r0**3) * u_t
    sol = integrate_geodesic(
        Gamma_num,
        [0.0, r0, math.pi / 2, 0.0],
        [u_t, 0.0, 0.0, u_phi],
        (0.0, 50.0),
        rtol=1e-8,
        atol=1e-10,
        max_step=1.0,
    )
    diag = check_conserved(sol, g_num, ginv_num, killing_indices=[0, 3])
    checks.append(_check("numeric geodesic integration succeeds", sol.success is True))
    checks.append(_check("numeric geodesic norm drift is tiny", diag["norm_drift"] < 1e-8))
    checks.append(_check("numeric geodesic parameters fully substituted", len(remaining) == 0))

    Q = symbols("Q", real=True)
    rn = select_metric("reissner_nordstrom", coords, {"M": M, "Q": Q})
    g_rn = rn["g_metric"]
    ginv_rn = rn["g_inv_metric"] or g_rn.inv()
    F = build_faraday_tensor([Q / r, 0, 0, 0], coords)
    T_em = compute_maxwell_stress_energy(F, g_rn, ginv_rn, coords)
    checks.append(_check("Coulomb F_01 = Q/r^2", sp.simplify(F[0, 1] - Q / r**2) == 0))
    checks.append(_check("Coulomb F_10 = -Q/r^2", sp.simplify(F[1, 0] + Q / r**2) == 0))
    checks.append(_check("Maxwell stress-energy is traceless", sp.simplify(T_em["T_trace"]) == 0))

    flrw = select_metric("frw_flat", coords)
    g_flrw = flrw["g_metric"]
    ginv_flrw = flrw["g_inv_metric"] or g_flrw.inv()
    Gamma_flrw = compute_christoffel(g_flrw, ginv_flrw, coords)
    adm = extract_adm_variables(g_flrw, ginv_flrw, coords)
    K = compute_extrinsic_curvature(g_flrw, ginv_flrw, Gamma_flrw, coords, adm)
    a_t = Function("a", positive=True)(t)
    checks.append(_check("FLRW ADM lapse is 1", adm["N"] == 1))
    checks.append(_check("FLRW ADM shift is zero", list(adm["beta_contra"]) == [0, 0, 0]))
    checks.append(_check(
        "FLRW K_trace follows GR_python sign convention",
        sp.simplify(K["K_trace"] + 3 * sp.diff(a_t, t) / a_t) == 0,
    ))

    fig = draw_all_penrose_diagrams(output_path=None, show=False)
    checks.append(_check("Penrose all-panel returns six axes", hasattr(fig, "axes") and len(fig.axes) == 6))

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
