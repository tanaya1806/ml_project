"""Apply the user's edited wording onto the clean IEEE-formatted template (formatting from template, text from user)."""
import sys, re, docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
tpl, usr, out = sys.argv[1:4]
m = docx.Document(tpl); u = docx.Document(usr)
def cp(d): return [p for p in d.paragraphs if p.text.strip() and not re.fullmatch(r"\s*\(\d+\)\s*", p.text)]
U, M = cp(u), cp(m); assert len(U) == len(M), (len(U), len(M))
changed = 0
for a, b in zip(U, M):
    if a.text.strip() == b.text.strip(): continue
    short = len(a.text) < 90 and len(b.text) < 90
    assert not short, (a.text, b.text)          # headings/captions must already match
    rs = b.runs; rs[0].text = a.text.strip()
    for r in rs[1:]: r._r.getparent().remove(r._r)
    changed += 1
for p in m.paragraphs:
    if re.match(r"^Fig\. \d\. ", p.text): p.alignment = WD_ALIGN_PARAGRAPH.CENTER
m.save(out); print("paragraphs replaced:", changed)
