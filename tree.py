#!/usr/bin/env python3
"""
tree.py — print directory structure (folders + files) without external deps.

Usage examples:
  python tree.py .                           # current repo
  python tree.py /path/to/repo --max-depth 3
  python tree.py . --exclude .git node_modules __pycache__ .venv
  python tree.py . --sizes --ascii -o repo_tree.txt
"""

from pathlib import Path
import argparse, os, sys, fnmatch

def should_exclude(path: Path, patterns):
    name = path.name
    rel = str(path)
    for pat in patterns:
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat):
            return True
    return False

def format_size(n):
    for unit in ("B","KB","MB","GB","TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0

def print_tree(root: Path, prefix="", depth=0, max_depth=None, excludes=(), ascii_only=False, show_sizes=False):
    if max_depth is not None and depth > max_depth:
        return

    entries = []
    try:
        with os.scandir(root) as it:
            for e in it:
                p = Path(e.path)
                if should_exclude(p, excludes):
                    continue
                entries.append((p, e.is_dir(follow_symlinks=False)))
    except PermissionError:
        print(prefix + "[permission denied]")
        return

    # Sort: dirs first, then files (case-insensitive)
    entries.sort(key=lambda t: (not t[1], t[0].name.lower()))

    branch_mid = "|-- " if ascii_only else "├── "
    branch_end = "`-- " if ascii_only else "└── "
    pipe_keep  = "|   " if ascii_only else "│   "
    pipe_blank = "    "

    for idx, (p, is_dir) in enumerate(entries):
        is_last = idx == len(entries) - 1
        connector = branch_end if is_last else branch_mid
        size_str = ""
        if show_sizes and not is_dir:
            try:
                size_str = f" ({format_size(p.stat().st_size)})"
            except OSError:
                pass
        print(prefix + connector + p.name + ("/" if is_dir else "") + size_str)
        if is_dir and (max_depth is None or depth < max_depth):
            new_prefix = prefix + (pipe_blank if is_last else pipe_keep)
            print_tree(
                p,
                prefix=new_prefix,
                depth=depth + 1,
                max_depth=max_depth,
                excludes=excludes,
                ascii_only=ascii_only,
                show_sizes=show_sizes,
            )

def main():
    parser = argparse.ArgumentParser(description="Print directory structure (like `tree`).")
    parser.add_argument("path", nargs="?", default=".", help="Root directory (default: current dir)")
    parser.add_argument("--max-depth", type=int, help="Limit recursion depth (levels under root)")
    parser.add_argument("--exclude", nargs="*", default=[".git","__pycache__","node_modules",".venv"],
                        help="Names or glob patterns to exclude")
    parser.add_argument("--ascii", action="store_true", help="Use ASCII characters only")
    parser.add_argument("--sizes", action="store_true", help="Show file sizes")
    parser.add_argument("--output", "-o", help="Write output to a file")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"Path not found: {root}", file=sys.stderr)
        sys.exit(1)

    def run_and_capture():
        print(root.name + "/")
        print_tree(
            root,
            prefix="",
            depth=1,
            max_depth=args.max_depth,
            excludes=args.exclude,
            ascii_only=args.ascii,
            show_sizes=args.sizes,
        )

    if args.output:
        from io import StringIO
        buf, old = StringIO(), sys.stdout
        sys.stdout = buf
        try:
            run_and_capture()
        finally:
            sys.stdout = old
        Path(args.output).write_text(buf.getvalue(), encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        run_and_capture()

if __name__ == "__main__":
    main()
