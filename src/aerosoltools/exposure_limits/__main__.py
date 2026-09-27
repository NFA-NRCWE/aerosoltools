"""Command line: check for / download a newer Danish limit-value order.

Usage::

    python -m aerosoltools.exposure_limits check
    python -m aerosoltools.exposure_limits fetch --latest -o limits.json
    python -m aerosoltools.exposure_limits fetch 2026/613 -o limits.json
    python -m aerosoltools.exposure_limits show [limits.json]

``check`` reports whether the order behind the bundled list is still in force
(and which order replaced it if not); ``fetch`` downloads and parses an order;
``show`` prints a saved (or the bundled) list.
"""

from __future__ import annotations

import argparse
import sys

from . import (
    BUNDLED_ELI,
    RetsinformationError,
    fetch_exposure_limits,
    find_current_order,
    load_exposure_limits,
)


def _print_list(limits) -> None:
    """Print a list's source and its limits as a table."""
    src = limits.source
    print(f"{src.label} — {src.title}")
    print(f"  {src.eli}  (status {src.status or '?'}, retrieved {src.retrieved})")
    print(
        f"  {len(limits)} particulate limits (Bilag 2, Afsnit {', '.join(src.sections)})"
    )
    reasons: dict[str, int] = {}
    for _name, reason in limits.excluded:
        reasons[reason] = reasons.get(reason, 0) + 1
    for reason, count in reasons.items():
        print(f"  left out: {count} × {reason}")
    table = limits.to_dataframe()[
        ["Substance", "CAS", "8-h limit", "Short-term limit", "Unit", "Remarks"]
    ]
    print(table.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    """Run the command line; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m aerosoltools.exposure_limits",
        description="Danish occupational exposure limits from Retsinformation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Is the order still in force?")
    check.add_argument("eli", nargs="?", default=BUNDLED_ELI, help="order to check")
    fetch = sub.add_parser("fetch", help="Download and parse an order.")
    fetch.add_argument("eli", nargs="?", default=BUNDLED_ELI, help="e.g. 2026/613")
    fetch.add_argument(
        "--latest", action="store_true", help="follow to the order in force now"
    )
    fetch.add_argument("-o", "--output", help="save the parsed list as JSON")
    show = sub.add_parser("show", help="Print a saved or the bundled list.")
    show.add_argument("path", nargs="?", help="list saved with fetch -o")
    args = parser.parse_args(argv)

    # The limits contain Danish letters and unit glyphs (mg/m³); keep a
    # non-UTF-8 console from failing on them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        if args.command == "check":
            found = find_current_order(args.eli)
            for step in found.chain:
                state = "in force" if step.in_force else "replaced"
                print(f"{step.label}: {state}  {step.eli}")
            for amendment in found.amendments:
                print(f"  amended by {amendment.label} ({amendment.title})")
            return 0
        if args.command == "fetch":
            eli = find_current_order(args.eli).current.eli if args.latest else args.eli
            limits = fetch_exposure_limits(eli)
            _print_list(limits)
            if args.output:
                limits.to_json(args.output)
                print(f"Saved to {args.output}")
            return 0
        _print_list(load_exposure_limits(args.path))
        return 0
    except (RetsinformationError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
