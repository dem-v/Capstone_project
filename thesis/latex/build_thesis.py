#!/usr/bin/env python3
"""Build LaTeX sources for the thesis from the canonical Markdown skeleton.

This is an output-preserving *format* conversion: it slices the canonical
`thesis/thesis_skeleton.md`, strips authoring scaffolding (TODO / VERIFIED /
Draft-status / meta-notes), promotes headings to chapter/section/subsection,
reclassifies Table / Figure / Graph / Chart blocks into the template's four
separately-numbered float families, and emits a self-contained `main.tex` via
pandoc plus an IEEE `thebibliography` harvested from `docs/references.md`.

Run from anywhere; paths are resolved relative to the repo root inferred from
this file's location (repo/thesis/latex/build_thesis.py -> repo).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # repo/thesis/latex
REPO = HERE.parent.parent                        # repo
SKELETON = REPO / "thesis" / "thesis_skeleton.md"
REFERENCES = REPO / "docs" / "references.md"
OUT = HERE                                       # write generated files here

# ---------------------------------------------------------------------------
# Small LaTeX text escaper for captions (raw-latex bypasses pandoc escaping).
# ---------------------------------------------------------------------------
def tex_escape(text: str) -> str:
    # convert `code` spans to \texttt{...} first, escaping inside
    def code_repl(m: re.Match) -> str:
        inner = m.group(1)
        inner = inner.replace("\\", "\\textbackslash{}")
        for ch in "&%#_${}":
            inner = inner.replace(ch, "\\" + ch)
        inner = inner.replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")
        return "\\texttt{" + inner + "}"

    parts = re.split(r"`([^`]+)`", text)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:                       # inside backticks
            out.append(code_repl(re.match(r"(.*)", part)))
        else:
            p = part.replace("\\", "\\textbackslash{}")
            for ch in "&%#_${}":
                p = p.replace(ch, "\\" + ch)
            p = p.replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")
            out.append(p)
    return "".join(out)


def fix_img_path(path: str) -> str:
    # skeleton uses ../outputs/...  (relative to thesis/); main.tex lives in
    # thesis/latex/, and we set \graphicspath{{../../}}, so strip leading ../.
    path = path.strip()
    if path.startswith("../"):
        path = path[3:]
    return path


# ---------------------------------------------------------------------------
# Float emission helpers (raw-latex fenced blocks pandoc passes through).
# ---------------------------------------------------------------------------
ENV = {"Figure": "figure", "Graph": "graph", "Chart": "chart"}


def raw_latex(body: str) -> list[str]:
    return ["", "```{=latex}", body.rstrip("\n"), "```", ""]


def emit_image_float(ftype: str, n: int, m: int, caption: str, paths: list[str]) -> list[str]:
    env = ENV[ftype]
    label = f"{env}:{n}-{m}"
    if len(paths) == 1:
        graphics = f"\\includegraphics[width=0.85\\linewidth,height=0.42\\textheight,keepaspectratio]{{{paths[0]}}}"
    else:
        w = round(0.95 / len(paths), 3)
        rows = []
        for p in paths:
            rows.append(f"\\includegraphics[width={w}\\linewidth,height=0.32\\textheight,keepaspectratio]{{{p}}}%")
        graphics = "\n\\hfill\n".join(rows)
    body = (
        f"\\begin{{{env}}}[htbp]\n\\centering\n{graphics}\n"
        f"\\caption{{{tex_escape(caption)}}}\n\\label{{{label}}}\n\\end{{{env}}}"
    )
    return raw_latex(body)


def emit_verbatim_float(ftype: str, n: int, m: int, caption: str, lines: list[str]) -> list[str]:
    env = ENV[ftype]
    label = f"{env}:{n}-{m}"
    verb = "\n".join(lines)
    body = (
        f"\\begin{{{env}}}[htbp]\n\\centering\n"
        f"\\begin{{minipage}}{{0.95\\linewidth}}\n\\scriptsize\n"
        f"\\begin{{verbatim}}\n{verb}\n\\end{{verbatim}}\n\\end{{minipage}}\n"
        f"\\caption{{{tex_escape(caption)}}}\n\\label{{{label}}}\n\\end{{{env}}}"
    )
    return raw_latex(body)


IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
CAP_RE = re.compile(r"^(Table|Figure|Graph|Chart)\s+(\d+)\.(\d+):\s*(.+?)\s*$")


def is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


# ---------------------------------------------------------------------------
# Body processing
# ---------------------------------------------------------------------------
# Authoring-process meta-sentences to delete (they reference the conversion /
# final-writing workflow, not the research). Kept conservative: genuine
# draft-caveat sentences about the research itself are left for the human pass.
STRIP_SENTENCES = [
    "The example paths below point to one balanced-review case and should be "
    "replaced by the final selected thesis-quality case after human figure selection.",
    "The table is intended as a final-writing scaffold; exact run parameters such "
    "as `ig_steps`, `gradshap_samples`, occlusion patch/stride, and "
    "`score_cam_channels_cap` must be filled from the final run metadata.",
]


def strip_scaffolding(lines: list[str]) -> list[str]:
    out = []
    skip_para = False
    for line in lines:
        s = line.strip()
        if skip_para:
            if s == "":
                skip_para = False
            continue
        if s.startswith("TODO") or s.startswith("VERIFIED (") or s.startswith("Draft status:"):
            skip_para = True            # drop this paragraph (until blank line)
            continue
        for sent in STRIP_SENTENCES:
            if sent in line:
                line = line.replace(sent, "").replace("  ", " ").rstrip()
        out.append(line)
    return out


def process_floats(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        mcap = CAP_RE.match(line)
        if not mcap:
            out.append(line)
            i += 1
            continue
        ftype, ch, num, caption = mcap.group(1), int(mcap.group(2)), int(mcap.group(3)), mcap.group(4)
        # advance past caption + blank lines to the content block
        j = i + 1
        while j < n and lines[j].strip() == "":
            j += 1
        if j >= n:
            out.append(line)
            i += 1
            continue
        block = lines[j]
        # --- fenced code block (Figure 3.1 schematic) ---
        if block.strip().startswith("```"):
            fence = block.strip()[:3]
            k = j + 1
            code = []
            while k < n and lines[k].strip() != fence:
                code.append(lines[k])
                k += 1
            k += 1  # skip closing fence
            out.extend(emit_verbatim_float(ftype, ch, num, caption, code))
            i = k
            continue
        # --- pipe table ---
        if is_table_line(block):
            k = j
            tbl = []
            while k < n and (is_table_line(lines[k]) or lines[k].strip() == ""):
                if lines[k].strip() == "":
                    break
                tbl.append(lines[k])
                k += 1
            cells_text = "\n".join(tbl)
            imgs = IMG_RE.findall(cells_text)
            if imgs and ftype in ENV:           # image grid -> image float
                paths = [fix_img_path(p) for p in imgs]
                out.extend(emit_image_float(ftype, ch, num, caption, paths))
                i = k
                continue
            else:                               # data table -> keep + pandoc caption
                out.extend(tbl)
                out.append("")
                out.append(f"  : {caption}")
                out.append("")
                i = k
                continue
        # --- standalone markdown image (maybe followed by a grid: Figure 4.1) ---
        mimg = IMG_RE.search(block)
        if mimg and ftype in ENV:
            paths = [fix_img_path(mimg.group(1))]
            k = j + 1
            # optional following image-grid table belongs to same float
            while k < n and lines[k].strip() == "":
                k += 1
            if k < n and is_table_line(lines[k]):
                tbl = []
                while k < n and is_table_line(lines[k]):
                    tbl.append(lines[k])
                    k += 1
                paths += [fix_img_path(p) for p in IMG_RE.findall("\n".join(tbl))]
            out.extend(emit_image_float(ftype, ch, num, caption, paths))
            i = k
            continue
        # fallback: keep caption as bold paragraph
        out.append(f"**{caption}**")
        i = i + 1
    return out


NUM_PREFIX = re.compile(r"^\d+(?:\.\d+)*\.?\s+(.+)$")


def transform_headings(lines: list[str]) -> list[str]:
    out: list[str] = []
    in_appendix = False
    for line in lines:
        m = re.match(r"^(#{2,4})\s+(.+?)\s*$", line)
        if not m:
            out.append(line)
            continue
        level = len(m.group(1))
        title = m.group(2)
        if level == 2:
            # chapter level, or Appendices / Bibliography markers
            if title.strip() == "Appendices":
                out.extend(["", "```{=latex}", "\\appendix", "```", ""])
                in_appendix = True
                continue
            cm = re.match(r"^Chapter\s+\d+\.\s+(.+)$", title)
            ttl = cm.group(1) if cm else title
            out.append(f"# {ttl}")
        elif level == 3:
            am = re.match(r"^Appendix\s+[A-Z]\.\s+(.+)$", title)
            if am:                                  # appendix -> chapter (A, B, C)
                out.append(f"# {am.group(1)}")
                continue
            if title.lower().startswith("conclusions to chapter"):
                out.append(f"## {title} {{-}}")
                continue
            nm = NUM_PREFIX.match(title)
            ttl = nm.group(1) if nm else title
            out.append(f"## {ttl}")
        elif level == 4:
            nm = NUM_PREFIX.match(title)
            ttl = nm.group(1) if nm else title
            out.append(f"### {ttl}")
    return out


def slice_between(lines: list[str], start_pred, end_pred) -> tuple[int, int]:
    start = end = -1
    for idx, l in enumerate(lines):
        if start == -1 and start_pred(l):
            start = idx
        elif start != -1 and end == -1 and end_pred(l):
            end = idx
            break
    if end == -1:
        end = len(lines)
    return start, end


def build_body(raw_lines: list[str]) -> str:
    # body = from "## Chapter 1." to end, minus the Bibliography prose section
    cstart, _ = slice_between(raw_lines,
                              lambda l: l.strip().startswith("## Chapter 1."),
                              lambda l: False)
    body = raw_lines[cstart:]
    # remove "## Bibliography" ... up to "## Appendices"
    bstart, bend = slice_between(body,
                                 lambda l: l.strip() == "## Bibliography",
                                 lambda l: l.strip() == "## Appendices")
    if bstart != -1:
        body = body[:bstart] + body[bend:]
    body = strip_scaffolding(body)
    body = process_floats(body)
    body = transform_headings(body)
    text = "\n".join(body) + "\n"
    # Prose cross-references whose literal number drifts from the auto-numbered
    # float (the skeleton reserved a Chart 4.1 faithfulness-AUC that is not
    # rendered). Wire them to \ref so they track the real number and survive
    # future figure additions. (Tables/figures have no drift -> left literal.)
    for find, ref in {
        "Chart 4.2 visualizes": "Chart \\ref{chart:4-2} visualizes",
        "Chart 4.3 visualizes": "Chart \\ref{chart:4-3} visualizes",
    }.items():
        text = text.replace(find, ref)
    return text


# ---------------------------------------------------------------------------
# Front-matter slices (abstract, abbreviations, glossary)
# ---------------------------------------------------------------------------
def extract_slice(raw_lines, start_head, end_head, drop_first_para=False):
    s, e = slice_between(raw_lines,
                         lambda l: l.strip() == start_head,
                         lambda l: l.strip() == end_head)
    chunk = raw_lines[s + 1:e]
    if drop_first_para:
        # drop a leading scaffolding paragraph like "Draft abstract (...)"
        k = 0
        while k < len(chunk) and chunk[k].strip() == "":
            k += 1
        if k < len(chunk) and chunk[k].strip().lower().startswith("draft abstract"):
            while k < len(chunk) and chunk[k].strip() != "":
                k += 1
            chunk = chunk[k:]
    return "\n".join(chunk).strip() + "\n"


# ---------------------------------------------------------------------------
# Bibliography from references.md
# ---------------------------------------------------------------------------
def build_abbrev_table(md: str) -> str:
    # paragraphs separated by blank lines, each "ABBR - Full form" (may wrap)
    paras = [re.sub(r"\s+", " ", p.strip()) for p in re.split(r"\n\s*\n", md) if p.strip()]
    rows = []
    for p in paras:
        m = re.match(r"^(.+?)\s+-\s+(.+)$", p)
        if not m:
            continue
        abbr, full = tex_escape(m.group(1)), tex_escape(m.group(2))
        rows.append(f"{abbr} & {full}\\\\")
    body = "\n".join(rows)
    return (
        "\\begin{longtable}{@{}l p{0.72\\linewidth}@{}}\n"
        "\\toprule\n\\textbf{Abbreviation} & \\textbf{Full Form}\\\\\n"
        "\\midrule\n\\endhead\n" + body + "\n\\bottomrule\n\\end{longtable}\n"
    )


def build_bibliography(ref_text: str) -> str:
    # Pair each "IEEE-style entry:" with the nearest preceding <a id="..."> anchor
    # in references.md, and use that id as a stable \bibitem key (e.g. ref-vit,
    # ref-ct-ich). Stable keys mean inserting a reference never renumbers any
    # other entry, and in-text \cite{ref-vit} stays readable. Falls back to a
    # sequential key only if an entry has no anchor.
    items, key, seen = [], None, set()
    for line in ref_text.splitlines():
        m_id = re.search(r'<a id="([^"]+)"', line)
        if m_id:
            key = m_id.group(1)
        m_e = re.search(r"IEEE-style entry:\s*(.+)", line)
        if not m_e:
            continue
        e = m_e.group(1).strip()
        # markdown *italics* -> \emph{}
        e = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", e)
        # escape bare & % # that are not already escaped
        e = e.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
        k = key or f"ref-{len(items) + 1}"
        if k in seen:
            raise ValueError(f"duplicate bibliography key: {k!r}")
        seen.add(k)
        items.append(f"\\bibitem{{{k}}} {e}")
    body = "\n\n".join(items)
    return ("\\begin{thebibliography}{99}\n"
            "\\addcontentsline{toc}{chapter}{Bibliography}\n"
            + body + "\n\\end{thebibliography}\n")


def inject_texttt_breaks(text: str) -> str:
    """Insert \\allowbreak after _ / - . inside every \\texttt{...} so long
    identifiers/paths wrap in narrow columns instead of overflowing. No hyphen
    is added; breaks are zero-width and only taken when needed."""
    # allow escaped chars (\_, \{) and empty groups (\textless{}) in the argument
    tt_re = re.compile(r"\\texttt\{((?:\\.|\{\}|[^{}])*)\}")

    def brk(m: re.Match) -> str:
        inner = m.group(1)
        inner = inner.replace("\\_", "\\_\\allowbreak ")
        for ch in ("/", "-", ".", ","):
            inner = inner.replace(ch, ch + "\\allowbreak ")
        return "\\texttt{" + inner + "}"

    return tt_re.sub(brk, text)


def wrap_wide_tables_landscape(text: str, caption_substrs: list[str]) -> str:
    """Wrap specific wide longtables (matched by caption substring) in a
    landscape page so many-column statistics tables get full width."""
    for sub in caption_substrs:
        # caption text is hard-wrapped by pandoc -> match with flexible whitespace
        pat = re.compile(sub.replace(" ", r"\s+"))
        mm = pat.search(text)
        if mm is None:
            continue
        cap = mm.start()
        begin = text.rfind("\\begin{longtable}", 0, cap)
        end = text.find("\\end{longtable}", cap)
        if begin == -1 or end == -1:
            continue
        end += len("\\end{longtable}")
        text = (text[:begin] + "\\begin{landscape}\n" + text[begin:end]
                + "\n\\end{landscape}" + text[end:])
    return text


def package_figures() -> tuple[int, list[str]]:
    """Copy every referenced image into thesis/latex/figures/ with a flattened
    unique name and rewrite \\includegraphics paths, so the folder is a
    self-contained Overleaf-uploadable project. Returns (copied, missing)."""
    import shutil

    main_tex = (OUT / "main.tex").read_text(encoding="utf-8")
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    inc_re = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)(outputs/[^}]+)\}")
    copied, missing, mapping = 0, [], {}

    def flat_name(rel: str) -> str:
        return rel[len("outputs/"):].replace("/", "__")

    for m in inc_re.finditer(main_tex):
        rel = m.group(2)
        if rel in mapping:
            continue
        src = REPO / rel
        flat = flat_name(rel)
        mapping[rel] = flat
        if src.exists():
            shutil.copyfile(src, figdir / flat)
            copied += 1
        else:
            missing.append(rel)

    def repl(m: re.Match) -> str:
        return m.group(1) + "figures/" + mapping[m.group(2)] + "}"

    main_tex = inc_re.sub(repl, main_tex)
    main_tex = inject_texttt_breaks(main_tex)
    main_tex = wrap_wide_tables_landscape(main_tex, [
        "CXR classifier performance summary",             # Table 4.1 (10 cols)
        "improvement-experiment paired Dice comparison",  # Table 4.6 (10 cols)
    ])
    (OUT / "main.tex").write_text(main_tex, encoding="utf-8")
    return copied, missing


def run_pandoc(md_path: Path, tex_path: Path, fragment=True):
    cmd = ["pandoc", str(md_path), "-o", str(tex_path)]
    if not fragment:
        cmd.append("-s")
    cmd += ["--top-level-division=chapter", "-f",
            "markdown+raw_tex+pipe_tables", "-t", "latex"]
    subprocess.run(cmd, check=True)


def main() -> int:
    raw = SKELETON.read_text(encoding="utf-8")
    raw_lines = raw.splitlines()

    # 1. body
    body_md = build_body(raw_lines)
    (OUT / "_body_clean.md").write_text(body_md, encoding="utf-8")
    run_pandoc(OUT / "_body_clean.md", OUT / "body.tex", fragment=True)

    # 2. front-matter slices -> tex fragments
    abstract_md = extract_slice(raw_lines, "## Abstract",
                                "## List of Abbreviations", drop_first_para=True)
    abbrev_md = extract_slice(raw_lines, "## List of Abbreviations",
                              "## Glossary of Methodological Terms")
    glossary_md = extract_slice(raw_lines, "## Glossary of Methodological Terms",
                                "## Chapter 1. Introduction")
    (OUT / "_abstract.md").write_text(abstract_md, encoding="utf-8")
    (OUT / "_glossary.md").write_text(glossary_md, encoding="utf-8")
    run_pandoc(OUT / "_abstract.md", OUT / "abstract_frag.tex")
    run_pandoc(OUT / "_glossary.md", OUT / "glossary_frag.tex")
    (OUT / "abbrev_frag.tex").write_text(build_abbrev_table(abbrev_md), encoding="utf-8")

    # 3. bibliography
    bib = build_bibliography(REFERENCES.read_text(encoding="utf-8"))
    (OUT / "bibliography.tex").write_text(bib, encoding="utf-8")
    n_refs = bib.count("\\bibitem")

    # 4. final assembly: pandoc standalone with includes + template vars
    cmd = [
        "pandoc", str(OUT / "_body_clean.md"), "-s",
        "--top-level-division=chapter",
        "--number-sections",  # enable native chapter/section numbering so
                              # \counterwithin floats become N.x (not 0.x)
        "-f", "markdown+raw_tex+pipe_tables", "-t", "latex",
        "-V", "documentclass=report",
        "-V", "papersize=a4",
        "-V", "fontsize=12pt",
        "-V", "linestretch=1.5",
        "-V", "indent=true",
        "-V", "geometry:left=2.5cm,right=1.5cm,top=2cm,bottom=2cm",
        "-H", str(OUT / "preamble_extra.tex"),
        "-B", str(OUT / "frontmatter.tex"),
        "-A", str(OUT / "backmatter.tex"),
        "-o", str(OUT / "main.tex"),
    ]
    subprocess.run(cmd, check=True)

    # 5. package figures so thesis/latex/ is a self-contained Overleaf upload
    copied, missing = package_figures()
    print(f"OK  {n_refs} bib entries, main.tex; figures copied={copied}")
    if missing:
        print("WARNING missing image sources (rendered as missing in PDF):")
        for r in missing:
            print("   -", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
