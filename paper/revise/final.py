"""Final pass for submission: cite Fig. 2 / Table I, add Algorithm 1, then recolour everything black."""
import sys, re, copy, docx
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Pt
src, out = sys.argv[1], sys.argv[2]
d = docx.Document(src); P = d.paragraphs
def sub_in_runs(p, a, b):
    for r in p.runs:
        if a in r.text: r.text = r.text.replace(a, b); return
    raise SystemExit("not in a single run: " + a)
def append_run(p, text):
    r = copy.deepcopy(p.runs[-1]._r); p._p.append(r)
    for t in r.findall(qn("w:t")): t.text = text; t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
assert P[73].text.startswith("The first table shows"); sub_in_runs(P[73], "The first table shows", "Table I shows")
assert P[77].text.startswith("A total of 131"); append_run(P[77], " Fig. 2 shows the score distributions and the ROC curve.")
sub_in_runs(P[85], "(Fig 3)", "(Fig. 3)")
assert P[69].text.startswith("The severity estimator"); append_run(P[69], " Algorithm 1 summarizes the complete procedure with the equations above.")
anchor = P[70]; assert anchor.text.startswith("L. Model-wise")

LINES = [
 ("Algorithm 1.", " Training, anomaly generation and few-shot detection.", True),
 ("Input:", " scans of ABIDE I; 20 seed regions R; window 30, stride 10; edit budget m, steps Q = 5, severity δ; shots k.", False),
 ("Output:", " patient flag and severity band.", False),
 ("1:", " for each scan, build the window graphs G_t by Eq. (1)–(2).", False),
 ("2:", " for each seed r in R, fix the 1-hop subgraph V_r from G_1 and keep it for all windows.", False),
 ("3:", " train on control scans only: embed z_t by Eq. (3)–(4), predict the next embedding by Eq. (5)–(9), minimize L by Eq. (10)–(12).", False),
 ("4:", " score each step A_t by Eq. (13), the patient S by Eq. (14); set thresholds from validation controls by Eq. (15).", False),
 ("5:", " for each control scan, region and severity: repeat Q times, take the gradient of A_t with respect to the edge weights, update the m most influential edges by Eq. (16) and clip the weights to [−1, 1].", False),
 ("6:", " build x, the 20 per-region maximum scores, for k real ASD scans, 30 generated anomalies and the control scans.", False),
 ("7:", " fit the logistic regression head by Eq. (17).", False),
 ("8:", " for a new patient, compute S; flag the patient when S exceeds the patient threshold or the head predicts ASD; assign a low, moderate or high severity band to flagged patients.", False),
]
tbl = d.add_table(rows=len(LINES), cols=1)
W = 5000
tblPr = tbl._tbl.tblPr
for tag in ("w:tblW", "w:tblBorders", "w:tblLayout", "w:tblCellMar"):
    e = tblPr.find(qn(tag))
    if e is not None: tblPr.remove(e)
tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), str(W)); tw.set(qn("w:type"), "dxa"); tblPr.append(tw)
bd = OxmlElement("w:tblBorders")
for side in ("top", "bottom"):
    b = OxmlElement("w:" + side); b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "8"); b.set(qn("w:space"), "0"); b.set(qn("w:color"), "000000"); bd.append(b)
tblPr.append(bd)
cm = OxmlElement("w:tblCellMar")
for side, v in (("top", 15), ("left", 40), ("bottom", 15), ("right", 40)):
    e = OxmlElement("w:" + side); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); cm.append(e)
tblPr.append(cm)
for row, (lab, txt, head) in zip(tbl.rows, LINES):
    c = row.cells[0]; c.width = docx.shared.Twips(W); p = c.paragraphs[0]
    pf = p.paragraph_format; pf.space_before = Pt(0); pf.space_after = Pt(0); pf.line_spacing = 1.0
    for s, bold in ((lab, True), (txt, False)):
        r = p.add_run(s); r.bold = bold; r.font.size = Pt(8); r.font.name = "Times New Roman"
        r._r.get_or_add_rPr().rFonts.set(qn("w:cs"), "Times New Roman")
    if head:
        tcPr = c._tc.get_or_add_tcPr(); tb = OxmlElement("w:tcBorders"); b = OxmlElement("w:bottom")
        b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "4"); b.set(qn("w:space"), "0"); b.set(qn("w:color"), "000000"); tb.append(b); tcPr.append(tb)
anchor._p.addprevious(tbl._tbl)
spacer = copy.deepcopy(anchor._p)
for r in spacer.findall(qn("w:r")): spacer.remove(r)
anchor._p.addprevious(spacer)

# everything black
for el in d.element.body.iter(qn("w:color")):
    if el.get(qn("w:val")) == "FF0000": el.set(qn("w:val"), "000000")
d.save(out)
