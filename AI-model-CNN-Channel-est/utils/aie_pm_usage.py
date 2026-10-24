#!/usr/bin/env python3
"""Report AIE program, data, and stack memory usage from linker outputs.

The AIE toolchain emits, per compute tile, an ``*.elf.map`` linker map and a
matching ``*.elf.calltree`` stack-analysis file.  This tool parses those
artifacts and prints a per-core table of program memory (PM), data memory
(DMb), and stack usage, flagging PM or stack overflows.

Important detail:
    The map's PM ``Size`` describes its address space, not the physical
    memory available to an AIE core.  This tool therefore compares PM ``Used``
    against a 16 KiB limit by default (``--limit`` overrides it).  DM usage
    and allocated stack come from the ELF map; maximum stack usage comes from
    the matching ``.elf.calltree`` file when available.

Exit status:
    0 when all reported cores fit their limits, 1 when any core overflows
    (PM used above the limit or analyzed stack above the allocation), 2 on
    argument/parse errors.

Examples:
    aie_pm_usage.py ../kernel/build/aie.mlir.prj --kernel frame_equalizer_preprocess \
        --kernel frame_equalizer_equalize
    aie_pm_usage.py build/main_core_0_2.elf.map --top 10
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


# Physical program memory available to one AIE compute tile, in bytes.  The
# linker map's "Size" field is the address space, so it cannot be used for
# overflow checks.
DEFAULT_PM_LIMIT = 16 * 1024
# Map file names look like main_core_0_2.elf.map; the capture group is the
# tile coordinates used in the report's "Core" column.
CORE_RE = re.compile(r"main_core_(.+)\.elf\.map$")
# Lines inside a memory section that carry the byte count of live data.
USED_RE = re.compile(r"^\s*Used\s*=\s*(\d+)\s*$")
# Stack-analysis line in the .elf.calltree file.
MAX_STACK_RE = re.compile(r"^Maximum stack size\s*:\s*(\d+)\s*$")
# Function record in the PM section, e.g.
#   0x0000..0x03ff ( 256 items): my_kernel (Function, Global, .text...)
FUNCTION_RE = re.compile(
    r"^\s*0x[0-9a-fA-F]+\.\.0x[0-9a-fA-F]+"
    r"\s+\(\s*(\d+) items\)\s*:\s*(.*?)"
    r"\s+\(Function,\s*(?:Global|Local),\s*\.text[^)]*\)"
)


@dataclass(frozen=True)
class Function:
    """One function linked into a tile's PM section.

    Attributes:
        name: Demangled/raw function name as printed by the linker.
        size: Number of bytes (``items``) the function occupies.
    """

    name: str
    size: int


@dataclass(frozen=True)
class MapUsage:
    """Parsed memory usage for a single compute tile's ELF map.

    Attributes:
        path: Location of the ``*.elf.map`` file this record was parsed from.
        core: Tile label extracted from the file name (for example ``0_2``),
            or the file stem when the name does not match the expected form.
        pm_used: Bytes of program memory in use.
        dm_used: Bytes of data memory (``DMb``) in use.
        stack_allocated: Stack bytes reserved in the ``DM_stack`` section.
        stack_maximum: Maximum analyzed stack from the matching
            ``.calltree`` file, or ``None`` when that file is absent.
        functions: Linked PM functions, in map order.
        text: Full map text, retained for kernel-name filtering.
    """

    path: Path
    core: str
    pm_used: int
    dm_used: int
    stack_allocated: int
    stack_maximum: int | None
    functions: tuple[Function, ...]
    text: str


def discover_maps(paths: list[Path]) -> list[Path]:
    """Expand a mix of map files and directories into a unique sorted list.

    Directory arguments are searched recursively for ``*.elf.map`` files, so
    a single path such as ``aie.mlir.prj`` can cover every tile.  Returned
    paths are resolved to absolute form to deduplicate overlaps (a file
    passed explicitly and also found by a directory scan appears once).

    Args:
        paths: ``.elf.map`` files or directories to search.

    Returns:
        Resolved map paths, sorted for deterministic reporting.

    Raises:
        ValueError: If an argument does not exist, or is neither a
            ``.elf.map`` file nor a directory.
    """
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
    """Extract the ``Used`` byte count and body of one memory-map section.

    Linker maps repeat a block per memory region, introduced by a header line
    of the form ``Memory map for memory 'NAME':`` and terminated by the next
    such header (or end of file).  This function isolates the requested
    section, scans it for the first ``Used = <n>`` line, and returns both the
    count and the section body so callers can mine it for function records.

    Args:
        lines: All lines of the map file, already split.
        name: Memory region name exactly as it appears in the header, for
            example ``PM``, ``DMb``, or ``DM_stack``.
        path: Map file path, used only in error messages.

    Returns:
        A ``(used_bytes, section_lines)`` tuple, where ``section_lines``
        excludes the section header and everything from the next header on.

    Raises:
        ValueError: If no section named ``name`` exists, or the section has
            no ``Used`` line.
    """
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
    """Read the maximum analyzed stack from the map's call-tree file.

    The AIE toolchain writes stack-depth analysis to a sibling file obtained
    by replacing the ``.map`` suffix with ``.calltree``.  That file may be
    missing for partially linked builds, in which case stack analysis is
    simply unavailable rather than an error.

    Args:
        map_path: Path to the ``*.elf.map`` file.

    Returns:
        Maximum stack size in bytes, or ``None`` when the calltree file does
        not exist.

    Raises:
        ValueError: If the calltree file exists but contains no maximum
            stack line (corrupt or unexpected format).
    """
    calltree_path = map_path.with_suffix(".calltree")
    if not calltree_path.exists():
        return None
    for line in calltree_path.read_text(errors="replace").splitlines():
        if match := MAX_STACK_RE.match(line.strip()):
            return int(match.group(1))
    raise ValueError(f"no maximum stack size in {calltree_path}")


def parse_map(path: Path) -> MapUsage:
    """Parse memory usage and linked functions from one ELF map.

    Reads the whole map text, extracts the ``PM`` (program memory), ``DMb``
    (data memory), and ``DM_stack`` sections, mines the PM body for the
    function records emitted by the linker, derives the tile label from the
    file name, and attaches the calltree-derived maximum stack when it is
    available.

    Args:
        path: Path to a ``*.elf.map`` file.

    Returns:
        The fully populated :class:`MapUsage` record for that tile.

    Raises:
        ValueError: If any required memory section or ``Used`` line is
            missing, or the calltree file is malformed.
    """
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    pm_used, pm_lines = parse_memory_used(lines, "PM", path)
    dm_used, _ = parse_memory_used(lines, "DMb", path)
    stack_allocated, _ = parse_memory_used(lines, "DM_stack", path)

    # Function records live in the PM section body; empty lines and other
    # annotations are skipped by the regex.
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
    """Format a byte count with thousands separators for the report table.

    Args:
        value: Byte count to format.

    Returns:
        Decimal string with comma grouping, for example ``"16,384"``.
    """
    return f"{value:,}"


def print_report(usages: list[MapUsage], limit: int, top: int, kernels: list[str]) -> None:
    """Print the per-core memory table and optional largest-function ranking.

    Builds one row per map with PM used/limit/free/percentage, DM used, stack
    maximum and allocation, and an ``OK``/``PM OVERFLOW``/``STACK OVERFLOW``
    status.  The ``Kernel`` column lists requested kernel names whose text
    was found in the map (``-`` when no kernel filter is active).  When
    ``top`` is nonzero, a second listing shows the largest linked PM functions
    for each map, which pinpoints what is consuming program memory.

    Args:
        usages: Parsed map records to report.
        limit: Physical PM limit in bytes used for the free/percentage
            columns and the overflow status.
        top: Number of largest functions to list per map; 0 disables the
            function listing.
        kernels: Kernel names requested on the command line (already used by
            the caller to filter); echoed in the Kernel column.
    """
    rows = []
    for usage in usages:
        # A map matches every requested kernel name found anywhere in it.
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
    # Column widths follow the widest cell in each column.
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        # Text columns are left-aligned; numeric columns are right-aligned.
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
    """Run the memory-usage report command-line tool.

    Discovers and parses the requested maps, optionally filters them to maps
    that mention one or more ``--kernel`` names, prints the report, and
    reports overflow through the exit status.

    Returns:
        ``0`` when all selected maps fit their PM/stack limits, ``1`` when
        any overflow was detected, ``2`` when discovery/parsing or kernel
        filtering failed (message written to stderr).
    """
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
        # Keep only maps whose text mentions at least one requested kernel.
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
