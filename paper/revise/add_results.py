"""Insert Section IV-H (comparison with existing detectors): text, Table V, Fig. 4, Fig. 5 and refs [20]-[23]. All new text is red. Usage: add_results.py in.docx out.docx"""
import sys, copy, json, docx
from docx.shared import RGBColor, Inches
from docx.oxml.ns import qn
src, out = sys.argv[1:3]
RED = RGBColor(255, 0, 0)
J = json.load(open("results/extra_baselines.json"))["syn"]
d = docx.Document(src); P = d.paragraphs
def find(s):
    return next(p for p in d.paragraphs if p.text.startswith(s))
def setp(el, texts, red_from=0):
    """set run texts of a copied paragraph; runs at index >= red_from are red; extra runs blanked"""
    p = docx.text.paragraph.Paragraph(el, None)
    for k, r in enumerate(p.runs):
        r.text = texts[k] if k < len(texts) else ""
        if k >= red_from and r.text: r.font.color.rgb = RED
    return p
def clone_after(anchor_el, src_p):
    el = copy.deepcopy(src_p._p); anchor_el.addnext(el); return el
G, body_p, cap3, pic3, cap4 = find("G. Model-wise"), find("To evaluate the models"), find("Fig. 3. "), None, find("TABLE IV.")
pic3 = cap3._p.getprevious()   # paragraph holding the Fig. 3 picture
tbl4 = cap4._p.getnext()
ref_last = find("[19]")
TEXT = ("Table V compares the proposed detector with five existing ones on 100 synthetic scans: the static and persistence scores, a one-class SVM [20], "
        "an isolation forest [21], an autoencoder [22] and a whole-brain GIN+GRU. Half of the scans carry an injected sub-network change. "
        "The proposed detector is best on every metric, with 96.0% accuracy, 0.92 MCC [23] and 0.981 AUROC (Fig. 4a). "
        "The whole-brain model reaches 79.0% and flags 18% of weak changes, against 76% (Fig. 4b). "
        "These scans favour sub-network methods. On the 131 real test patients, all models score 0.45 to 0.56 AUROC with overlapping intervals.")
cur = tbl4
h = clone_after(cur, G); setp(h, ["H. Comparison with existing detectors"], 0); cur = h
t = clone_after(cur, body_p); setp(t, [TEXT], 0); cur = t
# Table V caption + table
c5 = clone_after(cur, cap4); setp(c5, ["TABLE V.", " ", "Comparison with existing detectors on synthetic scans."], 2); cur = c5
ROWS = [("Proposed detector", "temporal_subgraph_detector"), ("Static score", "static_baseline"), ("Persistence score", "persistence_baseline"),
        ("One-class SVM [20]", "ocsvm"), ("Isolation forest [21]", "iforest"), ("Autoencoder [22]", "autoencoder"), ("Whole-brain GIN+GRU", "whole_brain_gin_gru"),
        ("Few-shot head (guided)", "few_shot_structure_aware_anomalies")]
tb = copy.deepcopy(tbl4); cur.addnext(tb); cur = tb
trs = tb.findall(qn("w:tr")); hdr, best, plain = trs[0], trs[1], trs[2]
for tr in trs[1:]: tb.remove(tr)
def fill(tr, vals):
    for tc, v in zip(tr.findall(qn("w:tc")), vals):
        ts = list(tc.iter(qn("w:t")))
        ts[0].text = v
        for x in ts[1:]: x.text = ""
        for rpr in tc.iter(qn("w:rPr")):
            for c in rpr.findall(qn("w:color")): rpr.remove(c)
            col = rpr.makeelement(qn("w:color"), {qn("w:val"): "FF0000"}); rpr.append(col)
fill(hdr, ["Model", "Acc.", "Sens.", "F1", "MCC", "AUROC"])
for i, (nm, k) in enumerate(ROWS):
    m = J[k]; tr = copy.deepcopy(best if i == 0 else plain)
    fill(tr, [nm, f"{m['accuracy']:.3f}", f"{m['sensitivity']:.3f}", f"{m['f1']:.3f}", f"{m['mcc']:.3f}", f"{m['auroc']:.3f}"]); tb.append(tr)
W = [int(x * 1440) for x in (1.28, 0.36, 0.36, 0.34, 0.36, 0.5)]
for gc, w in zip(tb.find(qn("w:tblGrid")).findall(qn("w:gridCol")), W): gc.set(qn("w:w"), str(w))
for tr in tb.findall(qn("w:tr")):
    for tc, w in zip(tr.findall(qn("w:tc")), W):
        tcw = tc.find(qn("w:tcPr")).find(qn("w:tcW")); tcw.set(qn("w:w"), str(w)); tcw.set(qn("w:type"), "dxa")
tb.find(qn("w:tblPr")).find(qn("w:tblW")).set(qn("w:w"), str(sum(W)))
# figures
for png, cap in (("figures_hires/fig_roc_strength_col.png", "Fig. 4. Synthetic-scan ROC and detection by strength."),):
    pe = clone_after(cur, docx.text.paragraph.Paragraph(pic3, None)); pp = docx.text.paragraph.Paragraph(pe, body_p._parent)
    for r in list(pp.runs): r._r.getparent().remove(r._r)
    pp.add_run().add_picture(png, width=Inches(3.4)); cur = pe
    ce = clone_after(cur, cap3); lab, ttl = cap.split(". ", 1); setp(ce, [lab + ". ", ttl], 1); cur = ce
# references
REFS = [
 [("B. Schölkopf ", 0), ("et al.", 1), (", “Estimating the support of a high-dimensional distribution,” ", 0), ("Neural Comput.", 1), (", vol. 13, no. 7, pp. 1443–1471, 2001, doi: 10.1162/089976601750264965.", 0)],
 [("F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation forest,” in ", 0), ("Proc. IEEE Int. Conf. Data Mining (ICDM)", 1), (", Pisa, Italy, 2008, pp. 413–422, doi: 10.1109/ICDM.2008.17.", 0)],
 [("M. Sakurada and T. Yairi, “Anomaly detection using autoencoders with nonlinear dimensionality reduction,” in ", 0), ("Proc. MLSDA Workshop Mach. Learn. Sensory Data Anal.", 1), (", 2014, pp. 4–11, doi: 10.1145/2689746.2689747.", 0)],
 [("D. Chicco and G. Jurman, “The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation,” ", 0), ("BMC Genomics", 1), (", vol. 21, art. 6, 2020, doi: 10.1186/s12864-019-6413-7.", 0)],
]
cur = ref_last._p
for n, parts in enumerate(REFS, 20):
    el = copy.deepcopy(ref_last._p); cur.addnext(el); cur = el; p = docx.text.paragraph.Paragraph(el, None)
    rs = p.runs; base_plain, base_it = copy.deepcopy(rs[2]._r), copy.deepcopy(rs[3]._r)
    for r in rs[2:]: r._r.getparent().remove(r._r)
    rs[0].text = f"[{n}]"; rs[0].font.color.rgb = RED; rs[1].font.color.rgb = RED
    for txt, it in parts:
        r = copy.deepcopy(base_it if it else base_plain); el.append(r)
    for r, (txt, it) in zip(p.runs[2:], parts): r.text = txt; r.font.color.rgb = RED
d.save(out); print("saved", out)
