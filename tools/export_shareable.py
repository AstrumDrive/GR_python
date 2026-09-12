#!/usr/bin/env python3
"""Generate the shareable GR_python snapshot from this repository.

The shareable copy used to be a hand-maintained fork.  That is how a defect in
the ADM module came to live in two places at once, and how 290 lines of
finished work (the Baez tutorial coverage) ended up sitting uncommitted in the
copy rather than in the source.  This script replaces the hand maintenance: the
snapshot becomes a generated artifact, and the only thing anyone edits is the
policy below.

Usage
-----
    python tools/export_shareable.py --check      report drift, write nothing
    python tools/export_shareable.py --write      regenerate the snapshot

Both modes refuse to touch anything if the source tree is dirty, and --write
also refuses if the DESTINATION has uncommitted or untracked work.  That second
guard is the one that matters: regenerating over unrecorded work is precisely
the accident this script exists to prevent.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# POLICY.  This is the whole editable surface.  Everything else is mechanism.
# ---------------------------------------------------------------------------

# Tracked paths that must NOT go out to colleagues.  One reason per line.
EXCLUDE = [
    ("grsuite_adapter.py", "internal integration with the GRSuite project"),
    ("GRSUITE_INTEROP.md", "documents that same internal integration"),
    ("tests/test_grsuite_adapter.py", "tests the internal integration"),
    ("astra_jobs/*", "internal job wrappers for the ASTRUM cluster"),
    ("reviews/*", "internal review records, not user documentation"),
    ("astrum_report_*.png", "screenshots of an internal report"),
    ("tools/export_shareable.py", "this script maintains the snapshot, "
                                  "it does not belong inside it"),
]

# Paths that exist only in the snapshot and must survive regeneration.
# The script never writes or deletes these.
PRESERVE = [
    ".gitattributes",
    "AGENTS.md",
    "START_HERE.md",
    "requirements.txt",
]

PROVENANCE = "SHAREABLE_FROM.txt"

DEFAULT_DEST = Path(__file__).resolve().parents[2] / "GR_python_shareable"


# ---------------------------------------------------------------------------
# mechanism
# ---------------------------------------------------------------------------

def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed in {repo}:\n{out.stderr}")
    return out.stdout


def is_excluded(path: str) -> str | None:
    for pattern, reason in EXCLUDE:
        if fnmatch.fnmatch(path, pattern):
            return reason
    return None


def content_hash(path: Path) -> str:
    """md5 of the content with CR stripped, so line endings are not drift."""
    return hashlib.md5(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def planned_files(src: Path) -> list[str]:
    tracked = [p for p in git(src, "ls-files").splitlines() if p]
    return sorted(p for p in tracked if is_excluded(p) is None)


def dest_state(dest: Path) -> tuple[list[str], list[str]]:
    tracked = [p for p in git(dest, "ls-files").splitlines() if p]
    untracked = [p for p in git(dest, "ls-files", "--others",
                                "--exclude-standard").splitlines() if p]
    return sorted(tracked), sorted(untracked)


def require_clean(repo: Path, label: str, allow_untracked: bool) -> None:
    dirty = [l for l in git(repo, "status", "--porcelain").splitlines() if l]
    if allow_untracked:
        dirty = [l for l in dirty if not l.startswith("??")]
    if dirty:
        print(f"\nREFUSING: {label} has uncommitted work:", file=sys.stderr)
        for line in dirty:
            print(f"    {line}", file=sys.stderr)
        print("\nCommit it or discard it first.  Regenerating over unrecorded\n"
              "work is how the Baez coverage was nearly lost.", file=sys.stderr)
        raise SystemExit(2)


def report(src: Path, dest: Path) -> int:
    plan = planned_files(src)
    tracked, untracked = dest_state(dest)
    preserve = set(PRESERVE)

    should_exist = set(plan) | preserve
    missing = [p for p in plan if not (dest / p).exists()]
    stale = [p for p in tracked if p not in should_exist and p != PROVENANCE]
    changed = [
        p for p in plan
        if (dest / p).exists()
        and content_hash(src / p) != content_hash(dest / p)
    ]
    absent_preserved = [p for p in PRESERVE if not (dest / p).exists()]

    print(f"source      {src}")
    print(f"destination {dest}")
    print(f"commit      {git(src, 'rev-parse', 'HEAD').strip()}")
    print()
    print(f"  files the policy exports        {len(plan)}")
    print(f"  excluded by policy              "
          f"{len([p for p in git(src, 'ls-files').splitlines() if p and is_excluded(p)])}")
    print(f"  preserved (snapshot-only)       {len(PRESERVE)}")
    print()

    def show(label: str, items: list[str]) -> None:
        if items:
            print(f"  {label} ({len(items)})")
            for p in items:
                print(f"      {p}")

    show("missing from the snapshot", missing)
    show("out of date in the snapshot", changed)
    show("in the snapshot but no longer exported", stale)
    show("preserved file absent, will not be created", absent_preserved)
    if untracked:
        print(f"  UNTRACKED in the snapshot ({len(untracked)}) "
              f"-- these block a rewrite")
        for p in untracked:
            print(f"      {p}")

    drift = len(missing) + len(changed) + len(stale)
    print()
    if drift == 0 and not untracked:
        print("  in sync")
    else:
        print(f"  drift: {drift} file(s)")
    return drift


def write(src: Path, dest: Path) -> None:
    plan = planned_files(src)
    tracked, _ = dest_state(dest)
    preserve = set(PRESERVE)

    for rel in plan:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)

    should_exist = set(plan) | preserve | {PROVENANCE}
    removed = []
    for rel in tracked:
        if rel not in should_exist:
            path = dest / rel
            if path.exists():
                path.unlink()
                removed.append(rel)

    commit = git(src, "rev-parse", "HEAD").strip()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# GR_python shareable snapshot",
        f"# generated from the canonical repository at commit {commit}",
        f"# on {stamp} UTC by tools/export_shareable.py",
        "#",
        "# Do not edit this snapshot by hand.  Changes belong in the canonical",
        "# repository; run the exporter again to bring them across.",
        "#",
        "# Hashes are md5 of the content with CR stripped, so line endings do",
        "# not count as drift.  Verify with:",
        "#   python tools/export_shareable.py --check",
        "#",
        f"# Files preserved from the snapshot and not generated: "
        f"{', '.join(PRESERVE)}",
        "",
    ]
    for rel in plan:
        lines.append(f"{content_hash(src / rel)}  {rel}")
    (dest / PROVENANCE).write_text("\n".join(lines) + "\n", encoding="utf-8",
                                   newline="\n")

    print(f"  exported {len(plan)} file(s)")
    if removed:
        print(f"  removed {len(removed)} no longer exported:")
        for rel in removed:
            print(f"      {rel}")
    print(f"  wrote {PROVENANCE}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true",
                      help="report drift and exit non-zero if any; writes nothing")
    mode.add_argument("--write", action="store_true",
                      help="regenerate the snapshot")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"snapshot repository (default: {DEFAULT_DEST})")
    args = ap.parse_args()

    src = Path(__file__).resolve().parents[1]
    dest = args.dest.resolve()
    if not (dest / ".git").exists():
        raise SystemExit(f"{dest} is not a git repository")

    # The source must be recorded before anything is shipped from it.
    require_clean(src, "the canonical repository", allow_untracked=True)

    if args.check:
        return 1 if report(src, dest) else 0

    # The destination must have nothing unrecorded, tracked or not.
    require_clean(dest, "the snapshot", allow_untracked=False)
    print()
    write(src, dest)
    print("\nReview the result and commit it inside the snapshot repository.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
