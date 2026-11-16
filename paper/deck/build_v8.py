"""v7 -> v8: shorter results section (2 graph slides + 2 app-screenshot slides), editable. Run from repo root after build_v7.py."""
import copy, json, numpy as np
from pptx import Presentation
from pptx.util import Pt, Inches, Emu
from pptx.dml.color import RGBColor
from pptx.chart.data import XyChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
from sklearn.metrics import roc_curve
SRC, OUT = "paper/deck/GRAPH_ANOMALY_EndSem_v7.pptx", "paper/deck/GRAPH_ANOMALY_EndSem_v8.pptx"
SH = "paper/deck/screens/"
J = json.load(open("paper/results/extra_baselines.json")); cur = J["curves"]
prs = Presentation(SRC); S = list(prs.slides)   # 0-5 intro..contributions, 6 overall, 7 10-shot, 8 generated, 9 edit-detect, 10 baseline, 11 ROC+strength, 12 discussion, 13 conclusion
def walk(shs):
    for sh in shs:
        yield sh
        if sh.shape_type == 6: yield from walk(sh.shapes)
def set_text(sh, new):
    tf = sh.text_frame; p = tf.paragraphs[0]; r = p.runs[0]; r.text = new
    for x in p.runs[1:]: x._r.getparent().remove(x._r)
    for q in tf.paragraphs[1:]: q._p.getparent().remove(q._p)
def find(slide, name): return [x for x in slide.shapes if x.name == name]
HEAD = {"Group 2", "Group 4", "Group 52", "Group 54", "Group 56", "Freeform 59"}
blank = S[6].slide_layout
def new_slide(title):
    s = prs.slides.add_slide(blank)
    for el in S[10].shapes._spTree:
        nm = el.xpath(".//p:cNvPr/@name")
        if el.tag.split("}")[1] in ("grpSp", "sp") and nm and nm[0] in HEAD:
            new = copy.deepcopy(el)
            for a in new.iter():
                for k in list(a.attrib):
                    if k.endswith("}embed") or k.endswith("}link"):
                        rel = S[10].part.rels[a.attrib[k]]; a.attrib[k] = s.part.relate_to(rel.target_part, rel.reltype)
            s.shapes._spTree.append(new)
    for x in walk(s.shapes):
        if x.has_text_frame and x.text_frame.text.strip() == "RESULTS: BASELINE COMPARISON": set_text(x, title)
    return s
def card(slide, x, y, w, h, val, txt):
    for e in (find(S[7], "Stat card")[0], find(S[7], "Stat value")[0], find(S[7], "Stat text")[0]): slide.shapes._spTree.append(copy.deepcopy(e._element))
    c, v, t = slide.shapes[-3], slide.shapes[-2], slide.shapes[-1]
    c.left, c.top, c.width, c.height = Inches(x), Inches(y), Inches(w), Inches(h)
    v.left, v.top, v.width, v.height = Inches(x + 0.1), Inches(y + 0.1), Inches(w - 0.2), Inches(0.95); set_text(v, val)
    t.left, t.top, t.width, t.height = Inches(x + 0.25), Inches(y + 1.0), Inches(w - 0.5), Inches(h - 1.05); set_text(t, txt)
    for r in t.text_frame.paragraphs[0].runs: r.font.size = Pt(17)
def pic(slide, path, x, y, w=None, h=None, cap=None):
    p = slide.shapes.add_picture(SH + path, Inches(x), Inches(y), width=Inches(w) if w else None, height=Inches(h) if h else None)
    p.line.color.rgb = RGBColor(0xAF, 0xC9, 0xE8); p.line.width = Pt(1.25)
    if cap:
        tb = slide.shapes.add_textbox(Inches(x), p.top + p.height + Emu(60000), p.width, Inches(0.5)); r = tb.text_frame.paragraphs[0].add_run(); r.text = cap
        r.font.size = Pt(15); r.font.italic = True; r.font.name = "Times New Roman"; r.font.color.rgb = RGBColor(0x44, 0x4F, 0x5C); tb.text_frame.word_wrap = True
    return p
# ---- R3: scan 8 in the app --------------------------------------------------------------------------
R3 = new_slide("RESULTS: SCAN 8 IN THE APP")
pic(R3, "s1_upload.png", 0.8, 1.5, w=9.6, cap="1. Patient details and scan_008.csv uploaded in the NeuroCare screening page.")
pic(R3, "s2_report.png", 0.8, 5.65, w=9.6, cap="2. Screening report: autism indicators detected, high severity.")
card(R3, 10.8, 1.5, 8.4, 2.75, "Scan 8", "scan_008.csv is a synthetic test scan with an injected sub-network change of high strength. It is not a patient.")
card(R3, 10.8, 4.5, 8.4, 2.75, "193.9", "patient score, far above the alert level of 19.8 (95th percentile of validation controls), so the scan is flagged.")
card(R3, 10.8, 7.5, 8.4, 2.75, "High", "severity band, the same as the injected strength. It is an exploratory estimate and is not validated.")
# ---- R4: brain connections + comparison tab ---------------------------------------------------------
R4 = new_slide("RESULTS: BRAIN CONNECTIONS")
pic(R4, "s3_brain_top.png", 0.8, 1.5, w=10.2, cap="3D view (top) of the connections that changed most in scan 8; orange links became stronger.")
pic(R4, "s4_compare.png", 11.2, 1.5, w=8.0, cap="Model comparison tab: the same 100 scans, ten approaches.")
for nm in ("Takeaway", "Takeaway text"): R4.shapes._spTree.append(copy.deepcopy(find(S[9], nm)[0]._element))
B, T = R4.shapes[-2], R4.shapes[-1]
B.left, B.top, B.width, B.height = Inches(0.8), Inches(9.9), Inches(18.4), Inches(0.7); T.left, T.top, T.width, T.height = Inches(1.0), Inches(9.95), Inches(18.0), Inches(0.6)
set_text(T, "Scan 8 is a synthetic test scan. On the 131 real ABIDE patients every approach is near chance (AUROC 0.45-0.56).")
# ---- R1: few-shot + generated anomalies on one slide --------------------------------------------------
R1 = S[8]
for x in walk(R1.shapes):
    if x.has_text_frame and x.text_frame.text.strip() == "RESULTS: GENERATED ANOMALIES": set_text(x, "RESULTS: FEW-SHOT, ANOMALIES")
gen = [x for x in R1.shapes if x.has_table][0]
fs_src = [x for x in S[6].shapes if x.has_table][0]
R1.shapes._spTree.append(copy.deepcopy(fs_src._element)); fs = R1.shapes[-1]
tb = fs.table._tbl                      # few-shot table: drop Precision and Recall columns
for i in (3, 2):
    tb.tblGrid.remove(tb.tblGrid.gridCol_lst[i])
    for tr in tb.tr_lst: tr.remove(tr.tc_lst[i])
def fit(frame, x, y, w, widths, rowh, sz):
    frame.left, frame.top, frame.width = Inches(x), Inches(y), Inches(w)
    for c, wd in zip(frame.table.columns, widths): c.width = Inches(wd)
    for r in frame.table.rows:
        r.height = Inches(rowh)
        for c in r.cells:
            for p in c.text_frame.paragraphs:
                for rn in p.runs: rn.font.size = Pt(sz)
    frame.height = Inches(rowh * len(frame.table.rows))
fit(fs, 0.8, 1.9, 9.0, [4.0, 1.7, 1.65, 1.65], 0.62, 15)
fit(gen, 10.2, 1.9, 9.0, [3.0, 1.2, 1.2, 1.2, 1.2, 1.2], 0.62, 15)
for r in fs.table.rows:
    for c in r.cells:
        for p in c.text_frame.paragraphs:
            for rn in p.runs: rn.text = rn.text.replace("Sub-structure", "Guided")
lab = [("Few-shot results (1, 5, 10 ASD scans, 131 real test patients)", 0.8), ("Generated-anomaly detection (70 held-out controls)", 10.2)]
for t_, x_ in lab:
    tbx = R1.shapes.add_textbox(Inches(x_), Inches(1.42), Inches(9.0), Inches(0.4)); r = tbx.text_frame.paragraphs[0].add_run(); r.text = t_; r.font.size = Pt(16); r.font.bold = True; r.font.name = "Times New Roman"; r.font.color.rgb = RGBColor(0x0D, 0x22, 0x57)
# move chart / cards / footnote up and tighten
for sh in R1.shapes:
    if sh.has_chart: sh.left, sh.top, sh.width, sh.height = Inches(0.8), Inches(4.75), Inches(11.4), Inches(4.6)
sc = [x for x in R1.shapes if x.name in ("Stat card", "Stat value", "Stat text")]
for k, grp in enumerate([sc[i:i + 3] for i in range(0, 9, 3)]):
    dy = Inches(4.75 - 4.45)
    for sh in grp: sh.top = sh.top + dy
ft = find(R1, "Text")[0]; ft.left, ft.top, ft.width, ft.height = Inches(0.8), Inches(9.75), Inches(18.4), Inches(0.9)
set_text(ft, "Few-shot: average over 1, 5 and 10 ASD scans (five random draws); generated anomalies raise F1, while accuracy and AUROC stay near chance. Generated anomalies: 70 held-out controls x 20 regions = 1,400 edited transitions per method, threshold 7.43 from validation controls; this measures how detectable the edits are (the same model guides and scores them), not ASD diagnosis.")
for rn in ft.text_frame.paragraphs[0].runs: rn.font.size = Pt(14)
# ---- R2: baseline slide: table left + ROC right ------------------------------------------------------
R2 = S[10]
bt = [x for x in R2.shapes if x.has_table][0]
fit(bt, 0.8, 1.55, 10.4, [4.1, 1.26, 1.26, 1.26, 1.26, 1.26], 0.56, 15)
xd = XyChartData()
SHORT = {"temporal_subgraph_detector": "Proposed", "whole_brain_gin_gru": "Whole-brain GIN+GRU", "static_baseline": "Static score", "persistence_baseline": "Persistence score", "ocsvm": "One-class SVM", "iforest": "Isolation forest", "autoencoder": "Autoencoder"}
COL = {"temporal_subgraph_detector": "0D2257", "whole_brain_gin_gru": "6A3D9A", "static_baseline": "2A9D8F", "persistence_baseline": "7F9FD0", "ocsvm": "B5651D", "iforest": "D9A066", "autoencoder": "8C564B"}
y = np.array(cur["y_syn"])
for k in SHORT:
    f, t_, _ = roc_curve(y, np.array(cur["syn"][k])); se = xd.add_series(f"{SHORT[k]} ({J['syn'][k]['auroc']:.2f})")
    for a, b in zip(f, t_): se.add_data_point(float(a), float(b))
se = xd.add_series("Chance"); se.add_data_point(0, 0); se.add_data_point(1, 1)
g = R2.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(11.4), Inches(1.45), Inches(7.9), Inches(5.4), xd); ch = g.chart
ch.has_title = True; ch.chart_title.text_frame.text = "ROC curves, 100 synthetic scans"; r = ch.chart_title.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(16); r.font.bold = True; r.font.name = "Times New Roman"
ch.font.size = Pt(12); ch.font.name = "Times New Roman"; ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(11)
for s_, k in zip(ch.plots[0].series, list(SHORT) + ["chance"]):
    s_.smooth = False; s_.marker.style = XL_MARKER_STYLE.NONE; s_.format.line.width = Pt(4 if k == "temporal_subgraph_detector" else 2); s_.format.line.color.rgb = RGBColor.from_string(COL.get(k, "808080"))
    if k == "chance": s_.format.line.dash_style = 4
for ax, nm in ((ch.category_axis, "False-positive rate"), (ch.value_axis, "True-positive rate")):
    ax.minimum_scale, ax.maximum_scale = 0, 1; ax.has_title = True; ax.axis_title.text_frame.text = nm; ax.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12); ax.has_major_gridlines = False
for sh in R2.shapes:
    if sh.name in ("Stat card", "Stat value", "Stat text"): sh.top = sh.top + Inches(0.1)
cards = [x for x in R2.shapes if x.name == "Stat text"]
set_text(cards[1], "of weak injected changes flagged by the proposed detector, against 18% for the whole-brain GIN+GRU (+17 points accuracy overall).")
# ---- delete slides, reorder, renumber -----------------------------------------------------------------
ids = prs.slides._sldIdLst; allids = list(ids)
for idx in (6, 7, 9, 11):
    sid = allids[idx]; prs.part.drop_rel(sid.rId); ids.remove(sid)
lst = list(ids)                       # now: 0-5, R1, R2, Disc, Concl, R3, R4
r3, r4 = lst[-2], lst[-1]
for e in (r3, r4): ids.remove(e)
ids.insert(8, r3); ids.insert(9, r4)
for i, s_ in enumerate(prs.slides, 1):
    for sh in s_.shapes:
        if sh.shape_type == 6 and sh.left > Inches(18.9) and sh.top > Inches(10.4) and sh.width < Inches(1):
            for x in walk([sh]):
                if x.has_text_frame and x.text_frame.text.strip().isdigit(): set_text(x, str(i))
prs.save(OUT); print("saved", OUT, len(prs.slides._sldIdLst), "slides")
