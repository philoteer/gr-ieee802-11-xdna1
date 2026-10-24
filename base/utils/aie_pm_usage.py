#!/usr/bin/env python3
"""Report AIE program, data, and stack memory usage from linker outputs.

Examples:
    aie_pm_usage.py ../kernel/build/aie.mlir.prj --kernel frame_equalizer_preprocess \
        --kernel frame_equalizer_equalize
    aie_pm_usage.py build/main_core_0_2.elf.map --top 10

The map's PM ``Size`` describes its address space, not the physical memory
available to an AIE core. This tool compares PM ``Used`` with a 16 KiB limit
by default. DM usage and allocated stack come from the ELF map; maximum stack
usage comes from the matching ``.elf.calltree`` file when available.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PM_LIMIT = 16 * 1024
CORE_RE = re.compile(r"main_core_(.+)\.elf\.map$")
USED_RE = re.compile(r"^\s*Used\s*=\s*(\d+)\s*$")
MAX_STACK_RE = re.compile(r"^Maximum stack size\s*:\s*(\d+)\s*$")
FUNCTION_RE = re.compile(
    r"^\s*0x[0-9a-fA-F]+\.\.0x[0-9a-fA-F]+"
    r"\s+\(\s*(\d+) items\)\s*:\s*(.*?)"
    r"\s+\(Function,\s*(?:Global|Local),\s*\.text[^)]*\)"
)


@dataclass(frozen=True)
class Function:
    name: str
    size: int


@dataclass(frozen=True)
class MapUsage:
    path: Path
    core: str
    pm_used: int
    dm_used: int
    stack_allocated: int
    stack_maximum: int | None
    functions: tuple[Function, ...]
    text: str


def discover_maps(paths: list[Path]) -> list[Path]:
    """Return unique ELF map files found in files and directories."""
    maps: set[Path] = set()
    for path in paths:
        if not path.exists():
            raise ValueError(f"path does not exist: {path}")
        if path.is_dir():
            maps.update(candidate.resolve() for candidate in path.rglob("*.elf.map"))
        elif path.name.endswith(".elf.map"):
            maps.add(path.resolve())
        else:
            raise ValueError(f"not an .elf.map file or directory: {path}")
    return sorted(maps)


def parse_memory_used(lines: list[str], name: str, path: Path) -> tuple[int, list[str]]:
    """Return the Used value and body of one named memory-map section."""
    section_start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() == f"Memory map for memory '{name}':"
        ),
        None,
    )
    if section_start is None:
        raise ValueError(f"no {name} memory section in {path}")

    section_lines: list[str] = []
    for line in lines[section_start + 1 :]:
        if line.startswith("Memory map for memory '"):
            break
        section_lines.append(line)

    used_match = next(
        (USED_RE.match(line) for line in section_lines if USED_RE.match(line)), None
    )
    if used_match is None:
        raise ValueError(f"no {name} Used value in {path}")
    return int(used_match.group(1)), section_lines


def parse_maximum_stack(map_path: Path) -> int | None:
    """Read maximum analyzed stack from the map's matching call-tree file."""
    calltree_path = map_path.with_suffix(".calltree")
    if not calltree_path.exists():
        return None
    for line in calltree_path.read_text(errors="replace").splitlines():
        if match := MAX_STACK_RE.match(line.strip()):
            return int(match.group(1))
    raise ValueError(f"no maximum stack size in {calltree_path}")


def parse_map(path: Path) -> MapUsage:
    """Parse memory usage and linked functions from one ELF map."""
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    pm_used, pm_lines = parse_memory_used(lines, "PM", path)
    dm_used, _ = parse_memory_used(lines, "DMb", path)
    stack_allocated, _ = parse_memory_used(lines, "DM_stack", path)

    functions = tuple(
        Function(match.group(2), int(match.group(1)))
        for line in pm_lines
        if (match := FUNCTION_RE.match(line)) is not None
    )
    core_match = CORE_RE.search(path.name)
    core = core_match.group(1) if core_match else path.stem
    return MapUsage(
        path,
        core,
        pm_used,
        dm_used,
        stack_allocated,
        parse_maximum_stack(path),
        functions,
        text,
    )


def format_bytes(value: int) -> str:
    return f"{value:,}"


def print_report(usages: list[MapUsage], limit: int, top: int, kernels: list[str]) -> None:
    rows = []
    for usage in usages:
        matched = [kernel for kernel in kernels if kernel in usage.text]
        kernel_label = ",".join(matched) if kernels else "-"
        free = limit - usage.pm_used
        stack_overflow = (
            usage.stack_maximum is not None
            and usage.stack_maximum > usage.stack_allocated
        )
        if free < 0:
            status = "PM OVERFLOW"
        elif stack_overflow:
            status = "STACK OVERFLOW"
        else:
            status = "OK"
        rows.append(
            (
                usage.core,
                kernel_label,
                format_bytes(usage.pm_used),
                format_bytes(limit),
                format_bytes(free),
                f"{100.0 * usage.pm_used / limit:.2f}%",
                format_bytes(usage.dm_used),
                (
                    format_bytes(usage.stack_maximum)
                    if usage.stack_maximum is not None
                    else "-"
                ),
                format_bytes(usage.stack_allocated),
                status,
            )
        )

    headers = (
        "Core",
        "Kernel",
        "PM used",
        "PM limit",
        "PM free",
        "PM usage",
        "DM used",
        "Stack max",
        "Stack alloc",
        "Status",
    )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        fields = [row[0].ljust(widths[0]), row[1].ljust(widths[1])]
        fields.extend(row[index].rjust(widths[index]) for index in range(2, 9))
        fields.append(row[9].ljust(widths[9]))
        print("  ".join(fields))

    if top:
        for usage in usages:
            print(f"\n{usage.path}:")
            largest = sorted(usage.functions, key=lambda function: function.size, reverse=True)[:top]
            if not largest:
                print("  No linked PM functions found")
                continue
            name_width = max(len(function.name) for function in largest)
            for function in largest:
                print(f"  {format_bytes(function.size):>8}  {function.name:<{name_width}}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report AIE program, data, and stack memory usage."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help=".elf.map files or directories to search recursively",
    )
    parser.add_argument(
        "-k",
        "--kernel",
        action="append",
        default=[],
        metavar="NAME",
        help="show only maps containing NAME; repeat to match additional kernels",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        metavar="N",
        help="show the N largest linked PM functions for each map",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_PM_LIMIT,
        metavar="BYTES",
        help=f"physical PM limit in bytes (default: {DEFAULT_PM_LIMIT})",
    )
    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.top < 0:
        parser.error("--top must be non-negative")

    try:
        map_paths = discover_maps(args.paths)
        if not map_paths:
            raise ValueError("no .elf.map files found")
        usages = [parse_map(path) for path in map_paths]
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.kernel:
        usages = [
            usage
            for usage in usages
            if any(kernel in usage.text for kernel in args.kernel)
        ]
        if not usages:
            names = ", ".join(args.kernel)
            print(f"error: no maps contain requested kernel(s): {names}", file=sys.stderr)
            return 2

    print_report(usages, args.limit, args.top, args.kernel)
    overflow = any(
        usage.pm_used > args.limit
        or (
            usage.stack_maximum is not None
            and usage.stack_maximum > usage.stack_allocated
        )
        for usage in usages
    )
    return 1 if overflow else 0


if __name__ == "__main__":
    raise SystemExit(main())
