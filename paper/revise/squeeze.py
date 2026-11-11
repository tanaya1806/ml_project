"""Formatting-only tightening (no wording changes): empty spacer lines, heading/reference spacing, figure size, algorithm font."""
import sys, re, docx
from docx.shared import Pt
from docx.oxml.ns import qn
src, out, FIG = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = docx.Document(src)
for p in list(d.paragraphs):
    if not p.text.strip() and not p._p.findall(".//" + qn("w:drawing")) and not p._p.findall(".//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath"):
        pf = p.paragraph_format
        if pf.space_before and pf.space_before.pt >= 5 and p.paragraph_format.space_after is not None and False: pass
        if p._p.getnext() is not None and p._p.getnext().tag == qn("w:p"):
            nxt = p._p.getnext(); txt = "".join(t.text or "" for t in nxt.iter(qn("w:t")))
            if re.match(r"^(IV\.|V\.|IV|V)", txt.strip()) or txt.strip().startswith("IV. RESULTS") or txt.strip().startswith("V. DISCUSSION"):
                p._p.getparent().remove(p._p)
for p in d.paragraphs:
    pf = p.paragraph_format
    t = p.text.strip()
    if re.match(r"^\[\d+\]", t): pf.space_after = Pt(0)
    elif re.match(r"^[IVX]+\. [A-Z]", t) and len(t) < 60:
        pf.space_before = Pt(6); pf.space_after = Pt(3)
    elif re.match(r"^[A-L]\. ", t) and len(t) < 70:
        pf.space_before = Pt(3); pf.space_after = Pt(1)
for inl in d.element.body.iter(qn("wp:inline")):
    ext = inl.find(qn("wp:extent")); cx, cy = int(ext.get("cx")), int(ext.get("cy"))
    if inl.find(".//" + qn("pic:pic")) is None if False else (cx == int(3171825 * 0.7)):
        ncx, ncy = int(cx * FIG), int(cy * FIG); ext.set("cx", str(ncx)); ext.set("cy", str(ncy))
        for e in inl.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}ext"): e.set("cx", str(ncx)); e.set("cy", str(ncy))
for p in d.paragraphs:
    if re.match(r"^Fig\. \d\. ", p.text): p.paragraph_format.space_after = Pt(3)
    if re.match(r"^TABLE [IVX]+\.", p.text): p.paragraph_format.space_before = Pt(3); p.paragraph_format.space_after = Pt(2)
tb = [t for t in d.tables if t.rows[0].cells[0].text.startswith("Algorithm 1")][0]
for r in tb._tbl.iter(qn("w:sz")): r.set(qn("w:val"), "15")
d.save(out)
