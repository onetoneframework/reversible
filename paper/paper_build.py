"""Inline the SVG figures into the manuscript and emit the print (PDF source) and artifact variants."""

import re
import sys
from pathlib import Path

SP = Path(__file__).resolve().parent
FIG = SP / "paper_fig"
src = (SP / "paper_revised.html").read_text(encoding="utf-8")


def svg_inline(name):
    s = (FIG / (name + ".svg")).read_text(encoding="utf-8")
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    s = re.sub(r"<!DOCTYPE[^>]*>", "", s)
    # drop matplotlib's fixed pt size so CSS width:100% + viewBox scale it
    s = re.sub(r'(<svg[^>]*?)\swidth="[^"]*pt"', r"\1", s, count=1)
    s = re.sub(r'(<svg[^>]*?)\sheight="[^"]*pt"', r"\1", s, count=1)
    return s.strip()


missing = []
def repl(m):
    name = m.group(1)
    if not (FIG / (name + ".svg")).exists():
        missing.append(name)
        return "<p><b>[missing figure %s]</b></p>" % name
    return svg_inline(name)


out = re.sub(r"<!--FIG:([a-z0-9_]+)-->", repl, src)
(SP / "paper_build.html").write_text(out, encoding="utf-8")

# artifact variant: no doctype/html/head/body wrapper, title + style first
art = re.sub(r"<!DOCTYPE html>\s*<html[^>]*>\s*<head>\s*<meta[^>]*>\s*", "", out)
art = art.replace("</head>\n<body>", "").replace("</body>\n</html>", "")
art = art.replace("@page { size: A4; margin: 20mm 18mm 22mm 18mm; }", "")
art = art.replace("html,body{ background:#fff; color:var(--ink);", "body{ background:#fff; color:var(--ink);")
(SP / "paper_artifact.html").write_text(art, encoding="utf-8")
print("built; missing figures:", missing or "none")
print("figures inlined:", len(re.findall(r"<svg", out)))
