"""Deck v6 -> v7: paper-aligned title and terms, highlighted novel steps (slide 5), two new comparison slides, updated discussion/conclusion.
Run from repo root: python3 paper/deck/build_v7.py"""
import copy, json, numpy as np
from pptx import Presentation
from pptx.util import Emu, Pt, Inches
from pptx.dml.color import RGBColor
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
from sklearn.metrics import roc_curve
SRC, OUT = "paper/deck/GRAPH_ANOMALY_EndSem_v6.pptx", "paper/deck/GRAPH_ANOMALY_EndSem_v7.pptx"
J = json.load(open("paper/results/extra_baselines.json")); cur = J["curves"]
prs = Presentation(SRC); S = list(prs.slides)
def walk(shs):
    for sh in shs:
        yield sh
        if sh.shape_type == 6: yield from walk(sh.shapes)
def set_text(sh, new):
    """replace the text of the first paragraph's first run, drop other runs/paragraphs"""
    tf = sh.text_frame; p = tf.paragraphs[0]; r = p.runs[0]; r.text = new
    for x in p.runs[1:]: x._r.getparent().remove(x._r)
    for q in tf.paragraphs[1:]: q._p.getparent().remove(q._p)
def frames(slide):
    for sh in walk(slide.shapes):
        if sh.has_text_frame: yield sh.text_frame
        if getattr(sh, "has_table", False) and sh.has_table:
            for row in sh.table.rows:
                for c in row.cells: yield c.text_frame
def sub_runs(slide, pairs):
    n = 0
    for tf in frames(slide):
        if True:
            for p in tf.paragraphs:
                for r in p.runs:
                    for a, b in pairs:
                        if a in r.text: r.text = r.text.replace(a, b); n += 1
    return n
# ---- 1. terminology (guided, as in the paper) and the paper title --------------------------------
TERMS = [("Sub-structure anomaly generation + detection", "Guided anomaly generation + detection"), ("Sub-structure edits are detected", "Guided edits are detected"),
         ("0.40 (sub-structure)", "0.40 (guided)"), ("sub-structure edits are detected", "guided edits are detected"), ("Sub-structure edits detected", "Guided edits detected"),
         ("Guided sub-structure anomaly generation", "Guided anomaly generation")]
for s in S: sub_runs(s, TERMS)
t = [x for x in walk(S[0].shapes) if x.has_text_frame and "FEW-SHOT TEMPORAL" in x.text_frame.text][0]
set_text(t, "FEW-SHOT TEMPORAL SUBGRAPH ANOMALY DETECTION AND STRUCTURE-AWARE ANOMALY GENERATION IN FUNCTIONAL BRAIN NETWORKS FOR AUTISM SPECTRUM DISORDER")
t.text_frame.paragraphs[0].runs[0].font.size = Pt(26)
# chart category / series names (keep the values)
for s in S[7:9]:
    for sh in s.shapes:
        if sh.has_chart:
            ch = sh.chart; cd = CategoryChartData(); cd.categories = [c.replace("Sub-structure generation", "Guided generation") for c in ch.plots[0].categories]
            for se in ch.plots[0].series: cd.add_series(se.name.replace("Sub-structure generation", "Guided generation"), list(se.values))
            ch.replace_data(cd)
# ---- 2. slide 5: shade the novel steps + legend ----------------------------------------------------
NOVEL = {"Step 3", "Step 4", "Step 7", "Step 8"}
for sh in S[4].shapes:
    if sh.name in NOVEL:
        sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor(0xFB, 0xEB, 0xC8); sh.line.color.rgb = RGBColor(0xB0, 0x6A, 0x14); sh.line.width = Pt(3.5)
    if sh.name == "Step 7": set_text_keep = sh.text_frame.paragraphs[0].runs[0]; set_text_keep.text = "GUIDED SUBGRAPH ANOMALY GENERATION"
def legend(slide, x, y):
    for k, (fill, ln, lab) in enumerate([("FBEBC8", "B06A14", "Novel component"), ("FFFFFF", "0D2257", "Standard step")]):
        bx = slide.shapes.add_shape(5, Inches(x + k * 4.6), Inches(y), Inches(0.55), Inches(0.32)); bx.fill.solid(); bx.fill.fore_color.rgb = RGBColor.from_string(fill)
        bx.line.color.rgb = RGBColor.from_string(ln); bx.line.width = Pt(2.5 if k == 0 else 1.75); bx.shadow.inherit = False
        tb = slide.shapes.add_textbox(Inches(x + k * 4.6 + 0.7), Inches(y - 0.05), Inches(3.6), Inches(0.42)); r = tb.text_frame.paragraphs[0].add_run(); r.text = lab
        r.font.size = Pt(16); r.font.name = "Times New Roman"; r.font.color.rgb = RGBColor(0x22, 0x2A, 0x33)
legend(S[4], 0.85, 10.62)
# ---- 3. new slides ---------------------------------------------------------------------------------
blank = S[6].slide_layout
def clone_frame(src, dst, skip):
    for el in src.shapes._spTree:
        tag = el.tag.split("}")[1]
        if tag not in ("sp", "grpSp", "pic", "graphicFrame", "cxnSp"): continue
        nm = el.xpath(".//p:cNvPr/@name")
        if nm and nm[0] in skip: continue
        new = copy.deepcopy(el)
        for a in new.iter():
            for k in list(a.attrib):
                if k.endswith("}embed") or k.endswith("}link"):
                    rel = src.part.rels[a.attrib[k]]; a.attrib[k] = dst.part.relate_to(rel.target_part, rel.reltype)
        dst.shapes._spTree.append(new)
def new_slide(title, skip=("Overall comparison", "Text")):
    s = prs.slides.add_slide(blank); clone_frame(S[6], s, skip)
    for x in walk(s.shapes):
        if x.has_text_frame and x.text_frame.text.strip() == "RESULTS: OVERALL COMPARISON": set_text(x, title)
    return s
def find(slide, name): return [x for x in slide.shapes if x.name == name]
def cell_set(tc, text, size, fill=None, bold=None):
    r = tc.xpath(".//a:r")[0]; r.xpath("./a:t")[0].text = text; rp = r.xpath("./a:rPr")[0]; rp.set("sz", str(size))
    if bold is not None: rp.set("b", "1" if bold else "0")
    if fill: tc.xpath("./a:tcPr/a:solidFill/a:srgbClr")[0].set("val", fill)
# slide A: comparison table
A = new_slide("RESULTS: BASELINE COMPARISON", skip=("Text",))
gf = [x for x in A.shapes if x.has_table][0]; tbl = gf.table._tbl; trs = tbl.tr_lst; hdr, plain = trs[0], trs[3]
for tr in trs[1:]: tbl.remove(tr)
ROWS = [("Proposed detector", "temporal_subgraph_detector"), ("Static score", "static_baseline"), ("Persistence score", "persistence_baseline"), ("One-class SVM", "ocsvm"),
        ("Isolation forest", "iforest"), ("Autoencoder", "autoencoder"), ("Whole-brain GIN+GRU", "whole_brain_gin_gru"), ("Few-shot head (guided)", "few_shot_structure_aware_anomalies")]
H = ["Detector", "Accuracy", "Sensitivity", "F1", "MCC", "AUROC"]
for tc, h in zip(hdr.tc_lst, H): cell_set(tc, h, 1800)
hdr.set("h", str(int(0.56 * 914400)))
for i, (nm, k) in enumerate(ROWS):
    m = J["syn"][k]; tr = copy.deepcopy(plain); tr.set("h", str(int(0.56 * 914400))); tbl.append(tr)
    vals = [nm, f"{m['accuracy']:.3f}", f"{m['sensitivity']:.3f}", f"{m['f1']:.3f}", f"{m['mcc']:.3f}", f"{m['auroc']:.3f}"]
    for j, (tc, v) in enumerate(zip(tr.tc_lst, vals)):
        best = i == 0
        cell_set(tc, v, 1700, fill=("DCEFE3" if best else ("E6F0FA" if i % 2 == 0 else "FFFFFF")), bold=best or None)
gf.top = Inches(1.55); gf.height = Inches(0.56 * 9)
for c, w in zip(gf.table.columns, [6.4, 2.4, 2.4, 2.4, 2.4, 2.4]): c.width = Inches(w)
def card(slide, x, y, w, h, val, txt, src):
    c, v, t = [copy.deepcopy(e._element) for e in (find(src, "Stat card")[0], find(src, "Stat value")[0], find(src, "Stat text")[0])]
    for e in (c, v, t): slide.shapes._spTree.append(e)
    c, v, t = [slide.shapes[-3], slide.shapes[-2], slide.shapes[-1]]
    c.left, c.top, c.width, c.height = Inches(x), Inches(y), Inches(w), Inches(h)
    v.left, v.top, v.width, v.height = Inches(x + 0.1), Inches(y + 0.1), Inches(w - 0.2), Inches(1.0); set_text(v, val)
    t.left, t.top, t.width, t.height = Inches(x + 0.2), Inches(y + 1.05), Inches(w - 0.4), Inches(h - 1.15); set_text(t, txt)
card(A, 0.8, 7.0, 6.0, 2.55, "96.0%", "accuracy of the proposed detector, the best of the seven detectors on 100 synthetic scans (AUROC 0.981).", S[7])
card(A, 7.0, 7.0, 6.0, 2.55, "+17 points", "over the whole-brain GIN+GRU (96.0% vs 79.0%): tracking sub-networks helps.", S[7])
card(A, 13.2, 7.0, 6.0, 2.55, "0.45–0.56", "AUROC of every detector on the 131 real patients: all are near chance.", S[7])
tx = copy.deepcopy(find(S[6], "Text")[0]._element); A.shapes._spTree.append(tx); tx = A.shapes[-1]
tx.left, tx.top, tx.width, tx.height = Inches(0.8), Inches(9.7), Inches(18.4), Inches(1.0)
set_text(tx, "Setup: 100 synthetic scans (50 normal-like, 50 with an injected sub-network change), thresholds from control scans only. Baselines: one-class SVM (Schölkopf et al., 2001), isolation forest (Liu et al., 2008), autoencoder (Sakurada & Yairi, 2014); MCC (Chicco & Jurman, 2020). The scans favour sub-network methods, so this is not evidence of clinical accuracy.")
for p in tx.text_frame.paragraphs:
    for r in p.runs: r.font.size = Pt(15)
# slide B: ROC and detection by strength (native, editable charts)
B = new_slide("RESULTS: ROC AND STRENGTH", skip=("Overall comparison", "Text"))
SHORT = {"temporal_subgraph_detector": "Proposed", "whole_brain_gin_gru": "Whole-brain GIN+GRU", "static_baseline": "Static score", "persistence_baseline": "Persistence score", "ocsvm": "One-class SVM", "iforest": "Isolation forest", "autoencoder": "Autoencoder"}
COL = {"temporal_subgraph_detector": "0D2257", "whole_brain_gin_gru": "6A3D9A", "static_baseline": "2A9D8F", "persistence_baseline": "7F9FD0", "ocsvm": "B5651D", "iforest": "D9A066", "autoencoder": "8C564B"}
y = np.array(cur["y_syn"]); st = np.array(cur["strength"])
def style(ch, title):
    ch.has_title = True; ch.chart_title.text_frame.text = title; r = ch.chart_title.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(18); r.font.bold = True; r.font.name = "Times New Roman"
    ch.font.size = Pt(14); ch.font.name = "Times New Roman"; ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(13)
xd = XyChartData()
for k in SHORT:
    f, t_, _ = roc_curve(y, np.array(cur["syn"][k])); se = xd.add_series(f"{SHORT[k]} ({J['syn'][k]['auroc']:.2f})")
    for a, b in zip(f, t_): se.add_data_point(float(a), float(b))
se = xd.add_series("Chance")
se.add_data_point(0, 0); se.add_data_point(1, 1)
g = B.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(0.8), Inches(1.5), Inches(9.2), Inches(8.0), xd); ch = g.chart; style(ch, "ROC curves, 100 synthetic scans (AUROC in brackets)")
for s_, k in zip(ch.plots[0].series, list(SHORT) + ["chance"]):
    s_.smooth = False; s_.marker.style = XL_MARKER_STYLE.NONE
    s_.format.line.width = Pt(4.5 if k == "temporal_subgraph_detector" else 2.25)
    s_.format.line.color.rgb = RGBColor.from_string(COL.get(k, "808080"))
    if k == "chance": s_.format.line.dash_style = 4
for ax, nm in ((ch.category_axis, "False-positive rate"), (ch.value_axis, "True-positive rate")):
    ax.minimum_scale, ax.maximum_scale = 0, 1; ax.has_title = True; ax.axis_title.text_frame.text = nm; ax.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14); ax.has_major_gridlines = False
cd = CategoryChartData(); cd.categories = ["Low", "Moderate", "High"]
for k in SHORT:
    fl = np.array(cur["syn_flag"][k]); cd.add_series(SHORT[k], [round(100 * float(fl[st == s].mean()), 1) for s in ("low", "moderate", "high")])
g2 = B.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(10.2), Inches(1.5), Inches(9.0), Inches(8.0), cd); c2 = g2.chart; style(c2, "Changed scans flagged (%) by strength of change")
for s_, k in zip(c2.plots[0].series, SHORT):
    s_.smooth = False; s_.format.line.width = Pt(4.5 if k == "temporal_subgraph_detector" else 2.25); s_.format.line.color.rgb = RGBColor.from_string(COL[k])
    s_.marker.style = XL_MARKER_STYLE.CIRCLE; s_.marker.size = 8; s_.marker.format.fill.solid(); s_.marker.format.fill.fore_color.rgb = RGBColor.from_string(COL[k]); s_.marker.format.line.color.rgb = RGBColor.from_string(COL[k])
c2.value_axis.maximum_scale = 100; c2.value_axis.minimum_scale = 0; c2.value_axis.has_major_gridlines = False; c2.value_axis.has_title = True; c2.value_axis.axis_title.text_frame.text = "Changed scans flagged (%)"
c2.value_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14); c2.category_axis.has_title = True; c2.category_axis.axis_title.text_frame.text = "Strength of the injected change"; c2.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
for nm in ("Takeaway", "Takeaway text"):
    e = copy.deepcopy(find(S[9], nm)[0]._element); B.shapes._spTree.append(e)
tk = B.shapes[-1]; tk.left, tk.top, tk.width, tk.height = Inches(1.0), Inches(9.78), Inches(18.0), Inches(0.62)
B.shapes[-2].left, B.shapes[-2].top, B.shapes[-2].width, B.shapes[-2].height = Inches(0.8), Inches(9.75), Inches(18.4), Inches(0.7)
set_text(tk, "On weak changes the proposed detector flags 76%, the autoencoder 47% and the whole-brain GIN+GRU 18%; all detectors are near chance on real patients.")
# move the two new slides after slide 10
ids = prs.slides._sldIdLst; lst = list(ids); a, b = lst[-2], lst[-1]
for e in (a, b): ids.remove(e)
ids.insert(10, a); ids.insert(11, b)
# ---- 4. discussion / conclusion ------------------------------------------------------------------
def add_par(shape, text, like=-1):
    ps = shape.text_frame.paragraphs; new = copy.deepcopy(ps[like]._p); ps[-1]._p.addnext(new)
    p = shape.text_frame.paragraphs[-1]; p.runs[0].text = text
    for x in p.runs[1:]: x._r.getparent().remove(x._r)
D = list(prs.slides)[12]; C = list(prs.slides)[13]
for sh in D.shapes:
    if sh.has_text_frame and "A diagnosis label" in sh.text_frame.text: add_par(sh, "Proposed detector is best of seven on synthetic scans (96.0%).")
    if sh.has_text_frame and "Detector is at chance" in sh.text_frame.text: add_par(sh, "On the 131 real patients all detectors, including the baselines, score AUROC 0.45–0.56.")
for sh in C.shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith("Built and tested"):
        add_par(sh, "On 100 synthetic scans the proposed detector is best of seven (96.0% accuracy).", like=3)
# ---- 5. slide numbers (bottom-right corner group) -------------------------------------------------
for i, s_ in enumerate(prs.slides, 1):
    for sh in s_.shapes:
        if sh.shape_type == 6 and sh.left > Inches(18.9) and sh.top > Inches(10.4) and sh.width < Inches(1):
            for x in walk([sh]):
                if x.has_text_frame and x.text_frame.text.strip().isdigit(): set_text(x, str(i))
prs.save(OUT); print("saved", OUT, len(prs.slides._sldIdLst), "slides")
