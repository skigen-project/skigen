#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

"""Audit doc/api_registry.json for stale docs, examples, and plot metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _repo_path(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    return repo_root / value


def _guide_path(repo_root: Path, guide: str | None) -> Path | None:
    if not guide:
        return None
    prefix = "/docs/guide/"
    if not guide.startswith(prefix):
        return None
    slug = guide.removeprefix(prefix)
    return repo_root / "doc" / "website" / "docs" / "guide" / f"{slug}.mdx"


def _has_snippet_marker(path: Path, snippet: str) -> bool:
    marker = f"//! [{snippet}]"
    try:
        return marker in path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False


def audit_registry(
    repo_root: Path,
    registry_path: Path,
    rendered_plot_dir: Path | None,
) -> tuple[Audit, dict[str, int]]:
    audit = Audit()
    with registry_path.open(encoding="utf-8") as stream:
        registry = json.load(stream)

    modules = registry.get("modules", {})
    classes = registry.get("classes", [])
    names_seen: set[str] = set()
    plot_count = 0

    for module_name, module in modules.items():
        include_name = module.get("include")
        if include_name and not (repo_root / include_name).is_file():
            audit.error(f"Module {module_name}: include path missing: {include_name}")
        for key in ("test", "benchmark"):
            path_value = module.get(key)
            if path_value and not (repo_root / path_value).is_file():
                audit.error(f"Module {module_name}: {key} path missing: {path_value}")

    for entry in classes:
        name = entry.get("name", "<unnamed>")
        module_name = entry.get("module")
        if name in names_seen:
            audit.error(f"Class {name}: duplicate registry entry")
        names_seen.add(name)

        if module_name not in modules:
            audit.error(f"Class {name}: unknown module {module_name!r}")

        guide = entry.get("guide")
        if entry.get("documented") and not guide:
            audit.error(f"Class {name}: documented=true but guide is missing")
        if guide:
            guide_file = _guide_path(repo_root, guide)
            if guide_file is None:
                audit.error(f"Class {name}: unsupported guide route: {guide}")
            elif not guide_file.is_file():
                audit.error(f"Class {name}: guide file missing: {guide_file.relative_to(repo_root)}")

        example = entry.get("example")
        example_file = _repo_path(repo_root, example)
        if example and (example_file is None or not example_file.is_file()):
            audit.error(f"Class {name}: example file missing: {example}")

        snippet = entry.get("example_snippet")
        if snippet:
            if example_file is None or not example_file.is_file():
                audit.error(f"Class {name}: example_snippet declared without a valid example: {snippet}")
            elif not _has_snippet_marker(example_file, snippet):
                audit.error(f"Class {name}: snippet marker missing in {example}: {snippet}")

        for plot in entry.get("plots", []):
            plot_count += 1
            stem = plot.get("stem")
            title = plot.get("title")
            plot_example = plot.get("example")
            plot_snippet = plot.get("snippet")
            if not stem:
                audit.error(f"Class {name}: plot entry missing stem")
            if not title:
                audit.error(f"Class {name}: plot entry missing title")
            plot_example_file = _repo_path(repo_root, plot_example)
            if not plot_example or plot_example_file is None or not plot_example_file.is_file():
                audit.error(f"Class {name}: plot example file missing: {plot_example}")
            if plot_snippet:
                if plot_example_file is None or not plot_example_file.is_file():
                    audit.error(f"Class {name}: plot snippet declared without valid example: {plot_snippet}")
                elif not _has_snippet_marker(plot_example_file, plot_snippet):
                    audit.error(f"Class {name}: plot snippet marker missing in {plot_example}: {plot_snippet}")
            else:
                audit.error(f"Class {name}: plot entry missing snippet")

            if rendered_plot_dir and stem:
                for suffix in ("light", "dark"):
                    rendered = rendered_plot_dir / f"{stem}_{suffix}.png"
                    if not rendered.is_file():
                        audit.error(f"Class {name}: rendered plot missing: {rendered}")
                    elif rendered.stat().st_size == 0:
                        audit.error(f"Class {name}: rendered plot is empty: {rendered}")

    documented = sum(1 for entry in classes if entry.get("documented") is True)
    with_example = sum(1 for entry in classes if entry.get("example"))
    with_plots = sum(1 for entry in classes if entry.get("plots"))
    summary = {
        "classes": len(classes),
        "documented": documented,
        "with_example": with_example,
        "with_plots": with_plots,
        "plot_entries": plot_count,
    }
    return audit, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="..", help="Repository root path")
    parser.add_argument("--registry", default="api_registry.json", help="Registry JSON path")
    parser.add_argument(
        "--check-rendered-plots",
        default=None,
        help="Directory containing rendered <stem>_{light,dark}.png files",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = (Path.cwd() / registry_path).resolve()
    rendered_plot_dir = Path(args.check_rendered_plots).resolve() if args.check_rendered_plots else None

    audit, summary = audit_registry(repo_root, registry_path, rendered_plot_dir)

    print("API registry audit")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    for warning in audit.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for error in audit.errors:
        print(f"error: {error}", file=sys.stderr)

    return 1 if audit.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
