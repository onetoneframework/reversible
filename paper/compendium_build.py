"""Inline generated tables and SVG figures into the compendium; emit print and artifact variants."""

import json
import re
from pathlib import Path

SP = Path(__file__).resolve().parent
FIG = SP / "paper_fig"
T = json.load(open(SP / "compendium_tables.json", encoding="utf-8"))
src = (SP / "compendium.html").read_text(encoding="utf-8")
missing = []


def svg_inline(name):
    s = (FIG / (name + ".svg")).read_text(encoding="utf-8")
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    s = re.sub(r"<!DOCTYPE[^>]*>", "", s)
    s = re.sub(r'(<svg[^>]*?)\swidth="[^"]*pt"', r"\1", s, count=1)
    s = re.sub(r'(<svg[^>]*?)\sheight="[^"]*pt"', r"\1", s, count=1)
    return s.strip()


def rfig(m):
    n = m.group(1)
    if not (FIG / (n + ".svg")).exists():
        missing.append("fig:" + n); return "<p><b>[missing figure %s]</b></p>" % n
    return svg_inline(n)


def rtbl(m):
    n = m.group(1)
    if n not in T:
        missing.append("tbl:" + n); return "<p><b>[missing table %s]</b></p>" % n
    return '<div class="gen">' + T[n] + '<p class="cap">Source table <code>%s</code>.</p></div>' % n


out = re.sub(r"<!--FIG:([a-z0-9_]+)-->", rfig, src)
out = re.sub(r"<!--TBL:([A-Za-z0-9_@]+)-->", rtbl, out)
(SP / "compendium_build.html").write_text(out, encoding="utf-8")
art = re.sub(r"<!DOCTYPE html>\s*<html[^>]*>\s*<head>\s*<meta[^>]*>\s*", "", out)
art = art.replace("</head>\n<body>", "").replace("</body>\n</html>", "")
art = art.replace("@page { size: A4; margin: 18mm 16mm 20mm 16mm; }", "")
art = art.replace("html,body{ background:#fff; color:var(--ink);", "body{ background:#fff; color:var(--ink);")
(SP / "compendium_artifact.html").write_text(art, encoding="utf-8")
print("missing:", missing or "none", "| tables inlined:", len(re.findall(r'<div class="gen">', out)), "| figures:", len(re.findall(r"<svg", out)))
