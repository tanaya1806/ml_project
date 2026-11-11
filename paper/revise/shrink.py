"""Shrink to six pages: METHODOLOGY heading, drop Experimental Setup (fold essentials into J), renumber tables/sections, scale figures."""
import sys, re, docx
from docx.oxml.ns import qn
src, out, SCALE = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = docx.Document(src); body = d.element.body
def find(start):
    for p in d.paragraphs:
        if p.text.startswith(start): return p
    raise SystemExit("missing " + start)
def run_sub(p, a, b):
    for r in p.runs:
        if a in r.text: r.text = r.text.replace(a, b); return
    raise SystemExit("no run: " + a)
run_sub(find("III. METHOD"), "III. METHOD", "III. METHODOLOGY")
# drop section IV (heading, text, caption, Table I)
h = find("IV. EXPERIMENTAL SETUP"); txt = find("Table I shows"); cap = find("TABLE I. Main")
t1 = [t for t in d.tables if t.rows[0].cells[0].text == "Item"][0]
for el in (h._p, txt._p, cap._p, t1._tbl): el.getparent().remove(el)
# fold essentials into J
J = find("To determine the detectability")
r = J.runs[-1]
r.text = r.text.rstrip() + " All settings were fixed beforehand, the checkpoint with the best validation loss (epoch 4 of 5) was used and test data served only for the final evaluation. Uncertainty is a 95% bootstrap interval (2,000 resamples) with a permutation test (5,000 shuffles); few-shot results give the mean and standard deviation over five draws of ASD scans that share one test set; the chance F1 for flag rate q is 2πq/(π + q), with π = 0.466 the share of ASD patients in the test set."
# section renumber
for a, b in (("V. RESULTS", "IV. RESULTS"), ("VI. DISCUSSION", "V. DISCUSSION"), ("VII. CONCLUSION", "VI. CONCLUSION")): run_sub(find(a), a, b)
# table renumber (ascending, per text node)
# drop the severity table (orig. Table IV); its numbers stay in the text
cap4 = find("TABLE IV. Severity"); t4 = [t for t in d.tables if t.rows[0].cells[0].text == "Evaluation"][0]
for el in (cap4._p, t4._tbl): el.getparent().remove(el)
for t in body.iter(qn("w:t")):
    if t.text and "(See Table IV)" in t.text: t.text = t.text.replace(" (See Table IV)", "")
M = {"II": "I", "III": "II", "V": "III", "VI": "IV"}
for t in body.iter(qn("w:t")):
    if t.text and re.search(r"\b(TABLE|Table) (II|III|IV|V|VI)\b", t.text):
        t.text = re.sub(r"\b(TABLE|Table) (VI|IV|III|II|V)\b", lambda m: m.group(1) + " " + M[m.group(2)], t.text)
# scale figure images (not the Fig. 1 canvas / fallback)
for inl in body.iter(qn("wp:inline")):
    ext = inl.find(qn("wp:extent")); cx, cy = int(ext.get("cx")), int(ext.get("cy"))
    if cy > 2500000 and cy < 4000000 and cx == 3171825:
        ext.set("cx", str(int(cx * SCALE))); ext.set("cy", str(int(cy * SCALE)))
        for e in inl.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}ext"):
            e.set("cx", str(int(cx * SCALE))); e.set("cy", str(int(cy * SCALE)))
d.save(out)
