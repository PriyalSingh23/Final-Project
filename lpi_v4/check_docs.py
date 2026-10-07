#!/usr/bin/env python3
"""check_docs.py -- do the commands this repo *documents* actually run?

Why this exists: ``lpi_eval.py`` spent a commit reading ``args.md`` while its
``--md`` definition was gone, and ``lpi_train.py`` still tells you to run
``link_test.py``, which nobody wrote.  Both are invisible to the test suite --
they live in the space between a docstring and a shell line, and by the time
you notice, you have "debugged" a metric gate that was never the problem.

So the space between documentation and argparse gets checked like code:

    python lpi_v4/check_docs.py            # exit 1 on any mismatch
    python lpi_v4/check_docs.py -v         # show every reference it accepted

Rules, deliberately conservative -- a doc check that cries wolf gets ignored:
  * only lines that look like a command (they mention ``python``) are parsed;
  * only ``--flags`` are checked, never positional arguments or values;
  * a script referenced by name anywhere in the repo is resolved by filename,
    so ``python lpi_v4/lpi_eval.py`` and ``lpi_v4\\lpi_eval.py`` are the same
    target; an unparseable script (the legacy ``priyal gg.py``) is skipped.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: files whose command lines are user-facing promises
DOCS = ["README.md", "FIELD_TEST_CHECKLIST.md", "AUTOMATION.md", "FINDINGS.md",
        "lpi_v4/README.md", "lpi_v4/v4.bat",
        ".github/workflows/ci.yml", ".github/workflows/nightly-train.yml",
        "colab/make_notebook.py", "colab/LPI_v4_Colab_Training.ipynb"]

SCRIPT_RE = re.compile(r"([\w./\\\\%{}$-]*?(\w+)\.py)((?:[^|#])*)")
PY_MENTION = re.compile(r"python[3]?|python\.exe|%LPI_PYTHON%|\bCMD\b|!\w")


def repo_scripts(root: str = REPO) -> dict[str, str]:
    """basename -> path, first match wins (lpi_v4/ sorts before the legacy root
    only by luck, so prefer the v4 copy explicitly when both exist)."""
    out: dict[str, str] = {}
    v4 = os.path.join(root, "lpi_v4")
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "__pycache__", ".venv", "node_modules")]
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(dirpath, f)
                if f not in out or p.startswith(v4 + os.sep):
                    out[f] = p
    return out


def argparse_flags(path: str) -> set[str] | None:
    """The dests argparse defines in *path*.

    None means "nothing to check": the file does not parse (the legacy
    ``priyal gg.py``) or it declares no optional flags at all, so it is not
    an argparse entry point and a flag mismatch would be a false alarm.
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except SyntaxError:
        return None
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument":
            for a in n.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                        and a.value.startswith("--"):
                    out.add(a.value[2:].replace("-", "_"))
            for kw in n.keywords:
                if kw.arg == "dest" and isinstance(kw.value, ast.Constant) \
                        and isinstance(kw.value.value, str):
                    out.add(kw.value.value)
    if not out:
        # no optional flags at all: this script is not argparse-driven (the legacy
        # webapp reads sys.argv, or takes none), so flag-checking it is meaningless
        return None
    return out


def doc_text(root: str, rel: str) -> str:
    """The command-bearing text of one doc: markdown/bat as-is, notebook sources
    joined, shell continuations glued so a flag on the next line still counts."""
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        return ""
    text = open(p, encoding="utf-8", errors="ignore").read()
    if rel.endswith(".ipynb"):
        import json
        cells = json.loads(text).get("cells", [])
        text = "\n".join("\n".join(c.get("source", [])) for c in cells)
    elif rel.endswith(".py") and "make_notebook" in rel:
        text = "\n".join(s.value for s in ast.walk(ast.parse(text))
                         if isinstance(s, ast.Constant) and isinstance(s.value, str))
    return re.sub(r"\\\s*\n[ \t]*", " ", text)


def scan(root: str = REPO) -> tuple[list, list, int]:
    """-> (missing_scripts, bad_flags, n_references_checked)"""
    scripts = repo_scripts(root)
    cache: dict[str, set[str] | None] = {}

    def flags(name: str):
        if name not in cache:
            cache[name] = argparse_flags(scripts[name]) if name in scripts else None
        return cache[name]

    missing, bad, n = [], [], 0
    sources = [(rel, doc_text(root, rel)) for rel in DOCS]
    # ...and the two places inside lpi_v4/*.py where a command line is user-facing:
    # printed hints ("next: python lpi_eval.py ...") and the module docstring, which
    # is what --help and the editor show.
    v4 = os.path.join(root, "lpi_v4")
    for f in sorted(os.listdir(v4)):
        if not f.endswith(".py"):
            continue
        src = open(os.path.join(v4, f), encoding="utf-8", errors="ignore").read()
        hints = "\n".join(m.group(1) for m in
                          re.finditer(r'print\(f?"([^"]{0,300}?)"', src))
        sources.append((f"lpi_v4/{f} (printed hints)", hints))
        try:
            doc = ast.get_docstring(ast.parse(src)) or ""
        except SyntaxError:
            doc = ""
        sources.append((f"lpi_v4/{f} (module docstring)", doc))

    for rel, text in sources:
        for raw in text.splitlines():
            # one line can hold two commands (`tx.py … && rx.py …`); attribute the
            # flags to the command they follow, not to the first script on the line
            for line in re.split(r"&&|\|\||;|\|", raw):
                if not PY_MENTION.search(line):
                    continue
                for m in SCRIPT_RE.finditer(line):
                    name = m.group(2) + ".py"
                    if name not in scripts:
                        missing.append((rel, name, line.strip()[:110]))
                        continue
                    defined = flags(name)
                    if defined is None:
                        continue        # legacy/no-argparse script: nothing to check
                    n += 1
                    for fl in re.findall(r"--([\w-]+)", m.group(3)):
                        if fl.replace("-", "_") not in defined:
                            bad.append((rel, name, "--" + fl, line.strip()[:110]))
    return missing, bad, n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="list every reference that was accepted")
    ap.add_argument("--allow", default="",
                    help="comma-separated script names to skip entirely")
    args = ap.parse_args(argv)

    missing, bad, n = scan()
    skip = {s.strip() for s in args.allow.split(",") if s.strip()}
    missing = [x for x in missing if x[1] not in skip]
    bad = [x for x in bad if x[1] not in skip]

    if args.verbose:
        for rel, name, ln in missing:
            print(f"MISSING  {rel}: {name}\n         {ln}")
        for rel, name, fl, ln in bad:
            print(f"BADFLAG  {rel}: {name} {fl}\n         {ln}")
    if not missing and not bad:
        print(f"[check_docs] OK: {n} documented commands reference real flags")
        return 0
    if missing:
        print(f"[check_docs] {len(missing)} command(s) point at a script that "
              f"does not exist:")
        for rel, name, ln in missing:
            print(f"  {rel}: {name}   <- {ln}")
    if bad:
        print(f"[check_docs] {len(bad)} flag(s) a documented command passes but "
              f"argparse does not define:")
        for rel, name, fl, ln in bad:
            print(f"  {name} {fl}   ({rel})  <- {ln}")
    print("[check_docs] fix the doc, or add the flag -- both are one line, "
          "and either is cheaper than a user guessing which is wrong")
    return 1


if __name__ == "__main__":
    sys.exit(main())
