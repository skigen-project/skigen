#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Skigen Contributors
#
# doxygen2mdx.py — Convert Doxygen XML output to Docusaurus MDX pages.
#
# Usage:
#   python3 doxygen2mdx.py \
#       --xml-dir doxygen_output/xml \
#       --out-dir website/docs/api \
#       --include-style Skigen/LinearModel
#
# Generates one .mdx file per public class, structured to match
# scikit-learn's API reference style.

import argparse
import html
import json
import re
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------------------------------------------------------------------------
# XML text extraction helpers
# ---------------------------------------------------------------------------

def text_of(el):
    """Recursively extract text from a Doxygen XML element, converting
    tags to Markdown/MDX equivalents."""
    if el is None:
        return ""
    parts = []
    if el.text:
        parts.append(el.text)
    for child in el:
        tag = child.tag
        if tag == "computeroutput":
            parts.append(f"`{text_of(child)}`")
        elif tag == "bold":
            parts.append(f"**{text_of(child)}**")
        elif tag == "emphasis":
            parts.append(f"*{text_of(child)}*")
        elif tag == "ref":
            parts.append(f"`{text_of(child)}`")
        elif tag == "ulink":
            url = child.get("url", "")
            parts.append(f"[{text_of(child)}]({url})")
        elif tag == "formula":
            formula = text_of(child).strip()
            # Doxygen wraps display math in \[ ... \] and inline in $ ... $
            if formula.startswith("\\["):
                formula = formula[2:]
                if formula.endswith("\\]"):
                    formula = formula[:-2]
                parts.append(f"\n\n$$\n{formula.strip()}\n$$\n\n")
            elif formula.startswith("$") and formula.endswith("$"):
                parts.append(formula)
            else:
                parts.append(f"${formula}$")
        elif tag == "simplesect":
            kind = child.get("kind", "")
            body = text_of(child).strip()
            if kind == "return":
                parts.append(f"\n\n**Returns:** {body}\n\n")
            elif kind == "note":
                parts.append(f"\n\n:::note\n{body}\n:::\n\n")
            elif kind == "see":
                parts.append(f"\n\n**See also:** {body}\n\n")
            else:
                parts.append(body)
        elif tag == "parameterlist":
            # Handled separately by dedicated extractors
            pass
        elif tag == "para":
            parts.append(text_of(child))
            parts.append("\n\n")
        elif tag == "sect1" or tag == "sect2" or tag == "sect3":
            parts.append(text_of(child))
        elif tag == "title":
            depth = {"sect1": "##", "sect2": "###", "sect3": "###"}.get(
                el.tag, "##"
            )
            parts.append(f"\n\n{depth} {text_of(child)}\n\n")
        elif tag == "itemizedlist":
            for item in child.findall("listitem"):
                item_text = text_of(item).strip()
                parts.append(f"- {item_text}\n")
            parts.append("\n")
        elif tag == "table":
            parts.append(render_table(child))
        elif tag == "programlisting":
            parts.append(render_code_block(child))
        elif tag == "sp":
            parts.append(" ")
        else:
            parts.append(text_of(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def render_table(table_el):
    """Convert a Doxygen <table> element to a Markdown table."""
    rows = table_el.findall("row")
    if not rows:
        return ""
    lines = []
    for i, row in enumerate(rows):
        cells = []
        for entry in row.findall("entry"):
            cells.append(text_of(entry).strip().replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("|" + "|".join(["---"] * len(cells)) + "|")
    return "\n" + "\n".join(lines) + "\n\n"


def render_code_block(listing_el):
    """Convert a Doxygen <programlisting> to a fenced code block."""
    lines = []
    for codeline in listing_el.findall("codeline"):
        line_parts = []
        for hl in codeline:
            t = hl.text or ""
            # Convert <sp/> inside highlight groups
            for sub in hl:
                if sub.tag == "sp":
                    t += " "
                elif sub.tag == "ref":
                    t += sub.text or ""
                else:
                    t += sub.text or ""
                if sub.tail:
                    t += sub.tail
            line_parts.append(t)
            if hl.tail:
                line_parts.append(hl.tail)
        lines.append("".join(line_parts))
    code = "\n".join(lines)
    return f"\n```cpp\n{code}\n```\n\n"


# ---------------------------------------------------------------------------
# C++ type beautifier — simplify Eigen/STL types for display
# ---------------------------------------------------------------------------

# Ordered list of (pattern, replacement) applied sequentially
_TYPE_SIMPLIFICATIONS = [
    # Eigen::Ref wrappers → inner type
    (r"const\s+Eigen::Ref<\s*const\s+(\w+)\s*>\s*&", r"\1"),
    (r"Eigen::Ref<\s*const\s+(\w+)\s*>\s*&", r"\1"),
    (r"Eigen::Ref<\s*(\w+)\s*>\s*&", r"\1"),
    # const ref → type
    (r"const\s+(\w+)\s*&", r"\1"),
    # plain ref
    (r"(\w+)\s*&", r"\1"),
]


def beautify_type(raw_type):
    """Simplify a C++ type string for human-friendly display."""
    t = raw_type.strip()
    # Remove stray backticks (from Doxygen <ref> tags)
    t = t.replace("`", "")
    for pattern, repl in _TYPE_SIMPLIFICATIONS:
        t = re.sub(pattern, repl, t)
    # Collapse multiple spaces
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Parameter / exception extraction
# ---------------------------------------------------------------------------

def extract_params(detail_el):
    """Extract @param entries from a detaileddescription element."""
    params = []
    for plist in detail_el.findall(".//parameterlist[@kind='param']"):
        for item in plist.findall("parameteritem"):
            name_el = item.find(".//parametername")
            desc_el = item.find("parameterdescription")
            name = text_of(name_el).strip() if name_el is not None else ""
            # Strip backticks that text_of() may add from <ref> tags
            name = name.strip("`")
            desc = text_of(desc_el).strip() if desc_el is not None else ""
            # Collapse multi-line descriptions
            desc = re.sub(r"\s*\n\s*", " ", desc)
            params.append((name, desc))
    return params


def extract_exceptions(detail_el):
    """Extract @throws entries."""
    exceptions = []
    for plist in detail_el.findall(".//parameterlist[@kind='exception']"):
        for item in plist.findall("parameteritem"):
            name_el = item.find(".//parametername")
            desc_el = item.find("parameterdescription")
            name = text_of(name_el).strip() if name_el is not None else ""
            name = name.strip("`")
            desc = text_of(desc_el).strip() if desc_el is not None else ""
            desc = re.sub(r"\s*\n\s*", " ", desc)
            exceptions.append((name, desc))
    return exceptions


def extract_return(detail_el):
    """Extract @return text."""
    for ss in detail_el.findall(".//simplesect[@kind='return']"):
        return text_of(ss).strip()
    return ""


def extract_notes(detail_el):
    """Extract @note entries."""
    notes = []
    for ss in detail_el.findall(".//simplesect[@kind='note']"):
        notes.append(text_of(ss).strip())
    return notes


# ---------------------------------------------------------------------------
# API Registry — single source of truth loaded from api_registry.json
# ---------------------------------------------------------------------------

_REGISTRY = None       # loaded lazily
_MODULES = {}          # module_name -> {include, dir_slug, sidebar_label}
_CLASS_INDEX = {}      # short_class_name -> registry entry dict
_GUIDE_LINK_MAP = {}   # snake_case -> guide path


def _load_registry(registry_path=None):
    """Load the registry from JSON.  Called once; cached globally."""
    global _REGISTRY, _MODULES, _CLASS_INDEX, _GUIDE_LINK_MAP
    if _REGISTRY is not None:
        return

    if registry_path is None:
        registry_path = Path(__file__).parent / "api_registry.json"
    with open(registry_path, encoding="utf-8") as f:
        _REGISTRY = json.load(f)

    _MODULES = _REGISTRY["modules"]

    for entry in _REGISTRY["classes"]:
        _CLASS_INDEX[entry["name"]] = entry
        # Build guide link map (CamelCase → snake_case → guide path)
        snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", entry["name"]).lower()
        if entry.get("guide"):
            _GUIDE_LINK_MAP[snake] = entry["guide"]


def registry_lookup(short_name):
    """Return the registry entry for *short_name* or raise if missing."""
    _load_registry()
    if short_name not in _CLASS_INDEX:
        raise RuntimeError(
            f"Class '{short_name}' is NOT registered in api_registry.json. "
            "Every public Skigen class must be listed there."
        )
    return _CLASS_INDEX[short_name]


def resolve_include(location_file):
    """Convert a Doxygen source file path to the Eigen-style include."""
    _load_registry()
    for mod_name, mod in _MODULES.items():
        if location_file and location_file.startswith(f"{mod_name}/"):
            return mod["include"]
    return f"Skigen/{location_file}" if location_file else "Skigen"


def resolve_subdir(location_file):
    """Convert a Doxygen source file path to a kebab-case subdirectory."""
    _load_registry()
    for mod_name, mod in _MODULES.items():
        if location_file and location_file.startswith(f"{mod_name}/"):
            return mod["dir_slug"]
    return "core"


def resolve_guide_link(class_name):
    """Resolve a guide link for a class name like 'ElasticNet'."""
    _load_registry()
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", class_name).lower()
    return _GUIDE_LINK_MAP.get(snake, "")


# ---------------------------------------------------------------------------
# URL slug helpers
# ---------------------------------------------------------------------------

def class_slug(compound_name):
    """Convert 'Skigen::ElasticNet' to 'elastic-net'."""
    short = compound_name.split("::")[-1]
    # CamelCase to kebab-case
    slug = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", short)
    return slug.lower()


# ---------------------------------------------------------------------------
# MDX generation for a single class
# ---------------------------------------------------------------------------

def generate_class_mdx(compounddef, out_dir):
    """Generate an MDX file for a single class compounddef element."""

    compound_name = compounddef.findtext("compoundname", "")
    short_name = compound_name.split("::")[-1]
    brief = text_of(compounddef.find("briefdescription")).strip()

    # Registry lookup — throws if class is not registered
    reg_entry = registry_lookup(short_name)
    sidebar_position = reg_entry.get("sidebar_position")
    plots = reg_entry.get("plots", [])

    # Template params
    tparams = []
    for tp in compounddef.findall("templateparamlist/param"):
        tp_type = text_of(tp.find("type")).strip()
        tp_defval = text_of(tp.find("defval")).strip()
        tparams.append((tp_type, tp_defval))

    # Inheritance
    bases = []
    for base in compounddef.findall("basecompoundref"):
        bases.append(html.unescape(base.text or text_of(base)))

    # Source file for include resolution
    loc_el = compounddef.find("location")
    loc_file = loc_el.get("file", "") if loc_el is not None else ""
    include_header = resolve_include(loc_file)

    # Collect public functions
    public_funcs = []
    for section in compounddef.findall("sectiondef[@kind='public-func']"):
        for member in section.findall("memberdef[@kind='function']"):
            if member.get("prot") != "public":
                continue
            public_funcs.append(member)

    # Separate constructor from methods
    constructor = None
    methods = []
    for func in public_funcs:
        name = func.findtext("name", "")
        if name == short_name:
            constructor = func
        else:
            methods.append(func)

    # Detailed class description
    detail_el = compounddef.find("detaileddescription")
    # Extract just the prose body, excluding structured sections (tables, lists
    # from @defgroup that duplicate the structured output below).
    class_detail = extract_body_text(detail_el) if detail_el is not None else ""

    # Determine output subdirectory from source location
    subdir = resolve_subdir(loc_file)

    slug = class_slug(compound_name)
    out_path = Path(out_dir)
    if subdir:
        out_path = out_path / subdir
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"{slug}.mdx"

    # Build MDX content
    lines = []

    # Frontmatter
    lines.append("---")
    lines.append(f'title: "{compound_name}"')
    lines.append(f"sidebar_label: {short_name}")
    if sidebar_position is not None:
        lines.append(f"sidebar_position: {sidebar_position}")
    lines.append("---")
    lines.append("")
    if plots:
        lines.append("import ExamplePlot from '@site/src/components/ExamplePlot';")
        lines.append("")
    lines.append(f"# {short_name}")
    lines.append("")

    # Include (small metadata line, before the class signature)
    lines.append(f"`#include <{include_header}>`")
    lines.append("")

    # Prominent class signature with template (Qt-style)
    if constructor is not None:
        sig_parts = []
        for p in constructor.findall("param"):
            pname = p.findtext("declname", "")
            pdefval = text_of(p.find("defval")).strip()
            if pname and pdefval:
                val = pdefval
                # Unwrap all Scalar{...} wrappers
                val = re.sub(r"Scalar\{([^}]*)\}", r"\1", val)
                sig_parts.append(f"{pname}={val}")
            elif pname:
                sig_parts.append(pname)
        sig_str = ', '.join(sig_parts)
        lines.append("```cpp")
        if tparams:
            tp_strs = []
            for tp_type, tp_defval in tparams:
                if tp_defval:
                    tp_strs.append(f"{tp_type} = {tp_defval}")
                else:
                    tp_strs.append(tp_type)
            lines.append(f"template <{', '.join(tp_strs)}>")
        lines.append(f"class {compound_name}({sig_str})")
        lines.append("```")
    else:
        lines.append("```cpp")
        if tparams:
            tp_strs = []
            for tp_type, tp_defval in tparams:
                if tp_defval:
                    tp_strs.append(f"{tp_type} = {tp_defval}")
                else:
                    tp_strs.append(tp_type)
            lines.append(f"template <{', '.join(tp_strs)}>")
        lines.append(f"class {compound_name}")
        lines.append("```")
    lines.append("")

    # Brief
    if brief:
        lines.append(brief)
        lines.append("")

    # Class detailed description (objective function, math, etc.)
    if class_detail:
        # Resolve "Read more in the User Guide" to an actual link
        guide_link = resolve_guide_link(short_name)
        if guide_link:
            class_detail = re.sub(
                r"Read more in the (?:@ref \S+ )?\"?User Guide\"?\.?",
                f"Read more in the [User Guide]({guide_link}).",
                class_detail,
            )
        lines.append(class_detail)
        lines.append("")

    lines.append("---")
    lines.append("")

    # Constructor
    if constructor is not None:
        ctor_detail_el = constructor.find("detaileddescription")
        xml_params = {}
        for p in constructor.findall("param"):
            pname = p.findtext("declname", "")
            ptype = text_of(p.find("type")).strip()
            pdefval = text_of(p.find("defval")).strip()
            xml_params[pname] = (ptype, pdefval)

        # Constructor brief
        ctor_brief = text_of(constructor.find("briefdescription")).strip()
        ctor_body = extract_body_text(ctor_detail_el)
        if ctor_body:
            lines.append(ctor_body)
            lines.append("")

        # Constructor params — sklearn-style definition list
        ctor_params = extract_params(ctor_detail_el) if ctor_detail_el is not None else []
        if ctor_params:
            lines.append("**Parameters:**")
            lines.append("")
            for pname, pdesc in ctor_params:
                ptype, pdefval = xml_params.get(pname, ("", ""))
                display_type = beautify_type(ptype)
                default_str = ""
                if pdefval:
                    val = pdefval
                    # Unwrap all Scalar{...} wrappers
                    val = re.sub(r"Scalar\{([^}]*)\}", r"\1", val)
                    # Escape braces for MDX
                    val = val.replace("{", "\\{").replace("}", "\\}")
                    default_str = f", default={val}"
                type_str = f"{display_type}{default_str}"
                if type_str:
                    lines.append(f"- **{pname}** : *{type_str}*")
                else:
                    lines.append(f"- **{pname}**")
                lines.append(f"  {pdesc}")
                lines.append("")

        # Constructor notes
        ctor_notes = extract_notes(ctor_detail_el) if ctor_detail_el is not None else []
        for note in ctor_notes:
            lines.append(f":::note\n{note}\n:::")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Attributes (public accessors that are const, no params, nodiscard)
    accessors = []
    non_accessors = []
    for func in methods:
        name = func.findtext("name", "")
        args = func.findtext("argsstring", "")
        is_const = func.get("const") == "yes"
        is_nodiscard = func.get("nodiscard") == "yes"
        has_no_params = not func.findall("param")
        # _impl suffix = internal CRTP dispatch, show as public API name
        if name.endswith("_impl"):
            non_accessors.append(func)
        elif is_const and has_no_params:
            accessors.append(func)
        else:
            non_accessors.append(func)

    if accessors:
        lines.append("**Attributes:**")
        lines.append("")
        for func in accessors:
            name = func.findtext("name", "")
            ret_type = text_of(func.find("type")).strip()
            display_type = beautify_type(ret_type)
            func_brief = text_of(func.find("briefdescription")).strip()
            if display_type:
                lines.append(f"- **{name}** : *{display_type}*")
            else:
                lines.append(f"- **{name}**")
            if func_brief:
                lines.append(f"  {func_brief}")
            lines.append("")

        # Notes from accessors
        for func in accessors:
            detail = func.find("detaileddescription")
            if detail is not None:
                for note in extract_notes(detail):
                    lines.append(f":::note\n{note}\n:::")
                    lines.append("")

        lines.append("---")
        lines.append("")

    # Methods
    if non_accessors:
        lines.append("## Methods")
        lines.append("")

        for func in non_accessors:
            name = func.findtext("name", "")
            # Strip _impl suffix for display (CRTP pattern)
            display_name = name.replace("_impl", "")

            # sklearn-style heading: method(param1, param2)
            param_names = []
            for p in func.findall("param"):
                pname = p.findtext("declname", "")
                if pname:
                    param_names.append(pname)
            heading_sig = f"{display_name}({', '.join(param_names)})"
            heading_sig = heading_sig.replace("[", "\\[").replace("]", "\\]")
            lines.append(f"### {heading_sig}")
            lines.append("")

            # Brief + body
            func_brief = text_of(func.find("briefdescription")).strip()
            func_detail_el = func.find("detaileddescription")
            func_body = extract_body_text(func_detail_el)
            if func_brief:
                lines.append(func_brief)
                lines.append("")
            if func_body:
                lines.append(func_body)
                lines.append("")

            # Params — definition list with beautified types
            func_params = extract_params(func_detail_el) if func_detail_el is not None else []
            if func_params:
                lines.append("**Parameters:**")
                lines.append("")
                xml_params = {}
                for p in func.findall("param"):
                    pname = p.findtext("declname", "")
                    ptype = text_of(p.find("type")).strip()
                    xml_params[pname] = ptype
                for pname, pdesc in func_params:
                    raw_type = xml_params.get(pname, "")
                    display_type = beautify_type(raw_type)
                    if display_type:
                        lines.append(f"- **{pname}** : *{display_type}*")
                    else:
                        lines.append(f"- **{pname}**")
                    lines.append(f"  {pdesc}")
                    lines.append("")

            # Return
            ret = extract_return(func_detail_el) if func_detail_el is not None else ""
            if ret:
                ret_type = text_of(func.find("type")).strip()
                display_ret = beautify_type(ret_type)
                lines.append("**Returns:**")
                lines.append("")
                if display_ret:
                    lines.append(f"- **result** : *{display_ret}*")
                else:
                    lines.append(f"- **result**")
                lines.append(f"  {ret}")
                lines.append("")

            # Exceptions
            exceptions = extract_exceptions(func_detail_el) if func_detail_el is not None else []
            if exceptions:
                lines.append("**Throws:**")
                lines.append("")
                for ename, edesc in exceptions:
                    lines.append(f"- `{ename}` — {edesc}")
                lines.append("")

            # Notes
            func_notes = extract_notes(func_detail_el) if func_detail_el is not None else []
            for note in func_notes:
                lines.append(f":::note\n{note}\n:::")
                lines.append("")

            lines.append("---")
            lines.append("")

    # Example section — extract @snippet / programlisting from class description
    examples = extract_examples(detail_el)
    if examples:
        lines.append("## Example")
        lines.append("")
        for ex in examples:
            lines.append(ex)
            lines.append("")
        lines.append("---")
        lines.append("")

    if plots:
        lines.append("## Plot examples")
        lines.append("")
        lines.append("These SkigenPlot figures are rendered from registered examples during the documentation build.")
        lines.append("")
        for plot in plots:
            title = plot.get("title", "Generated plot")
            stem = plot["stem"]
            example = plot.get("example")
            snippet = extract_source_snippet(example, plot.get("snippet"))
            lines.append(f"### {title}")
            lines.append("")
            if example:
                lines.append(f"Source example: [`{example}`](https://github.com/skigen-project/skigen/blob/main/{example})")
                lines.append("")
            lines.append(f'<ExamplePlot alt="{html.escape(title)}" stem="{stem}" />')
            lines.append("")
            if snippet:
                lines.append("Plot-generation snippet:")
                lines.append("")
                lines.append(snippet)
                lines.append("")
        lines.append("---")
        lines.append("")

    # Write file
    content = "\n".join(lines)
    # Clean up excessive blank lines
    content = re.sub(r"\n{4,}", "\n\n\n", content)
    out_file.write_text(content, encoding="utf-8")
    print(f"  Generated: {out_file}")
    return str(out_file)


def render_func_signature(func, class_name, display_name=None):
    """Render a C++ function signature as a fenced code block."""
    name = display_name or func.findtext("name", "")
    ret_type = text_of(func.find("type")).strip()
    # Remove stray backticks from type (e.g. from <ref> tags)
    ret_type = ret_type.replace("`", "")
    is_const = func.get("const") == "yes"
    is_explicit = func.get("explicit") == "yes"

    params = []
    for p in func.findall("param"):
        ptype = text_of(p.find("type")).strip()
        pname = p.findtext("declname", "")
        pdefval = text_of(p.find("defval")).strip()
        param_str = ptype
        if pname:
            param_str += f" {pname}"
        if pdefval:
            param_str += f" = {pdefval}"
        params.append(param_str)

    sig_parts = []
    if is_explicit:
        sig_parts.append("explicit ")
    if ret_type:
        sig_parts.append(f"{ret_type} ")
    sig_parts.append(f"{name}(")

    if len(params) <= 2:
        sig_parts.append(", ".join(params))
    else:
        sig_parts.append("\n")
        for i, p in enumerate(params):
            comma = "," if i < len(params) - 1 else ""
            sig_parts.append(f"    {p}{comma}\n")

    sig_parts.append(")")
    if is_const:
        sig_parts.append(" const")
    sig_parts.append(";")

    return "```cpp\n" + "".join(sig_parts) + "\n```"


def extract_examples(detail_el):
    """Extract @snippet / programlisting blocks from the detailed description."""
    if detail_el is None:
        return []
    examples = []
    for listing in detail_el.findall(".//programlisting"):
        code = render_code_block(listing)
        if code.strip():
            # Strip common leading whitespace from the code block
            inner = code.strip()
            # Remove fences to dedent, then re-add
            if inner.startswith("```cpp\n") and inner.endswith("\n```"):
                body = inner[7:-4]  # strip ```cpp\n and \n```
                body = textwrap.dedent(body)
                inner = f"```cpp\n{body}\n```"
            examples.append(inner)
    return examples


def extract_source_snippet(source_path, snippet_name):
    """Extract a //! [snippet] block from a source file as a C++ code block."""
    if not source_path or not snippet_name:
        return ""

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / source_path
    if not path.is_file():
        return ""

    marker = f"//! [{snippet_name}]"
    in_snippet = False
    snippet_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == marker:
            if in_snippet:
                break
            in_snippet = True
            continue
        if in_snippet:
            snippet_lines.append(line)

    if not snippet_lines:
        return ""

    body = textwrap.dedent("\n".join(snippet_lines)).strip("\n")
    return f"```cpp\n{body}\n```"


def extract_body_text(detail_el):
    """Extract the prose body from detaileddescription, excluding params/return/notes/tables."""
    if detail_el is None:
        return ""
    parts = []
    for child in detail_el:
        if child.tag == "para":
            # Skip paras that only contain parameterlist, simplesect, or tables
            has_content = False
            if child.text and child.text.strip():
                has_content = True
            for sub in child:
                if sub.tag not in ("parameterlist", "simplesect", "table",
                                   "programlisting", "sect1", "sect2", "sect3"):
                    has_content = True
            if has_content:
                # Extract just the text, not the structured parts
                text_parts = []
                if child.text:
                    text_parts.append(child.text)
                for sub in child:
                    if sub.tag in ("parameterlist", "simplesect"):
                        continue
                    if sub.tag == "computeroutput":
                        text_parts.append(f"`{text_of(sub)}`")
                    elif sub.tag == "bold":
                        text_parts.append(f"**{text_of(sub)}**")
                    elif sub.tag == "formula":
                        formula = text_of(sub).strip()
                        if formula.startswith("\\["):
                            formula = formula[2:]
                            if formula.endswith("\\]"):
                                formula = formula[:-2]
                            text_parts.append(f"\n\n$$\n{formula.strip()}\n$$\n\n")
                        elif formula.startswith("$") and formula.endswith("$"):
                            text_parts.append(formula)
                        else:
                            text_parts.append(f"${formula}$")
                    elif sub.tag == "ref":
                        text_parts.append(f"`{text_of(sub)}`")
                    elif sub.tag == "ulink":
                        url = sub.get("url", "")
                        text_parts.append(f"[{text_of(sub)}]({url})")
                    if sub.tail:
                        text_parts.append(sub.tail)
                body = "".join(text_parts).strip()
                if body:
                    parts.append(body)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main: process all class XML files
# ---------------------------------------------------------------------------

def generate_sidebars(out_dir):
    """Generate a sidebars.ts apiSidebar section from the registry."""
    _load_registry()

    # Group classes by module, ordered by sidebar_position
    from collections import defaultdict
    groups = defaultdict(list)
    for entry in _REGISTRY["classes"]:
        groups[entry["module"]].append(entry)
    for entries in groups.values():
        entries.sort(key=lambda e: e.get("sidebar_position", 999))

    # Build apiSidebar categories in module order
    categories = []
    # Use the module order from the JSON keys (insertion-ordered in Python 3.7+)
    for mod_name, mod in _MODULES.items():
        if mod_name not in groups:
            continue
        items = []
        for entry in groups[mod_name]:
            slug = class_slug(f"Skigen::{entry['name']}")
            items.append(f"                'api/{mod['dir_slug']}/{slug}'")
        if items:
            categories.append(
                f"        {{\n"
                f"            type: 'category',\n"
                f"            label: '{mod['sidebar_label']}',\n"
                f"            items: [\n"
                + ",\n".join(items) + "\n"
                f"            ],\n"
                f"        }}"
            )

    api_sidebar = "    apiSidebar: [\n" + ",\n".join(categories) + "\n    ],"

    # Write to a partial file that can be imported or copy-pasted
    partial_path = Path(out_dir) / "_apiSidebar.generated.ts"
    partial_path.write_text(
        "// Auto-generated from api_registry.json — do not edit manually.\n"
        "// Copy the apiSidebar block into sidebars.ts.\n\n"
        + api_sidebar + "\n",
        encoding="utf-8",
    )
    print(f"  Generated sidebar partial: {partial_path}")
    return api_sidebar


def main():
    parser = argparse.ArgumentParser(
        description="Convert Doxygen XML to Docusaurus MDX"
    )
    parser.add_argument(
        "--xml-dir", required=True,
        help="Path to Doxygen XML output directory (contains index.xml)"
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory for generated .mdx files"
    )
    parser.add_argument(
        "--classes", nargs="*", default=None,
        help="Only generate pages for these classes (e.g. Skigen::ElasticNet)"
    )
    parser.add_argument(
        "--registry", default=None,
        help="Path to api_registry.json (default: <script_dir>/api_registry.json)"
    )
    parser.add_argument(
        "--generate-sidebars", action="store_true",
        help="Also generate _apiSidebar.generated.ts in out-dir"
    )
    args = parser.parse_args()

    # Load registry
    if args.registry:
        _load_registry(Path(args.registry))
    else:
        _load_registry()

    xml_dir = Path(args.xml_dir)
    out_dir = Path(args.out_dir)

    if not xml_dir.exists():
        print(f"Error: XML directory not found: {xml_dir}", file=sys.stderr)
        sys.exit(1)

    # Parse index.xml to discover class files
    index_tree = ET.parse(xml_dir / "index.xml")
    index_root = index_tree.getroot()

    generated = []
    for compound in index_root.findall("compound"):
        kind = compound.get("kind")
        if kind != "class":
            continue
        name = compound.findtext("name", "")

        # Filter if --classes specified
        if args.classes and name not in args.classes:
            continue

        # Skip internal / base classes
        if "internal" in name.lower():
            continue

        # Skip base classes (Estimator, Predictor, Transformer, Classifier)
        short = name.split("::")[-1]
        if short in ("Estimator", "Predictor", "Transformer", "Classifier"):
            continue

        # Enforce registration — throws if class is unknown
        registry_lookup(short)

        refid = compound.get("refid")
        xml_file = xml_dir / f"{refid}.xml"
        if not xml_file.exists():
            print(f"  Warning: {xml_file} not found, skipping {name}",
                  file=sys.stderr)
            continue

        tree = ET.parse(xml_file)
        compounddef = tree.find(".//compounddef")
        if compounddef is None:
            continue

        path = generate_class_mdx(compounddef, str(out_dir))
        generated.append(path)

    print(f"\nGenerated {len(generated)} API page(s).")

    if args.generate_sidebars:
        generate_sidebars(str(out_dir))


if __name__ == "__main__":
    main()
