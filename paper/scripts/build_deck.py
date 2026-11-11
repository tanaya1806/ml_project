"""End-semester deck: edits the user's existing seminar deck (structure prepared in base.pptx) -> final pptx.
usage: python3 build_deck.py base.pptx out.pptx   (run from repo root)"""
import copy, json, re, sys, os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.oxml.ns import qn
from lxml import etree

ROOT = os.getcwd(); FIG = f"{ROOT}/paper/figures"; AS = f"{ROOT}/paper/docx_build/assets"
NAVY, TXT, CARD, EDGE, ACC, GREEN, GOLD, WHITE = "0D2257", "222A33", "E6F0FA", "AFC9E8", "9DC2EC", "2E6B4F", "A8752E", "FFFFFF"
FONT = "Times New Roman"
rgb = lambda h: RGBColor.from_string(h)
prs = Presentation(sys.argv[1]); S = list(prs.slides)

# ------------------------------------------------------------------ helpers
def inch(e): return e / 914400
def find_text_shapes(shapes):
    for sh in shapes:
        if sh.shape_type == 6:
            yield from find_text_shapes(sh.shapes)
        elif sh.has_text_frame:
            yield sh
def set_runs(tf, text):
    if not tf.paragraphs[0].runs: return
    """put text into the first run of the first paragraph, drop everything else (keeps formatting)"""
    p = tf.paragraphs[0]
    for extra in tf.paragraphs[1:]: extra._p.getparent().remove(extra._p)
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]: r._r.getparent().remove(r._r)
def set_title(slide, new):
    for sh in find_text_shapes(slide.shapes):
        if inch(sh.top) < 0.3 and sh.text_frame.text.strip() and inch(sh.height) > 1.2 and inch(sh.width) > 15:
            set_runs(sh.text_frame, new); return
    raise RuntimeError("title not found")
def set_number(slide, n):
    for g in slide.shapes:
        if g.shape_type == 6 and inch(g.left) > 19 and inch(g.top) > 10.5:
            for sh in find_text_shapes([g]):
                if sh.text_frame.text.strip().isdigit() and sh.text_frame.paragraphs[0].runs: set_runs(sh.text_frame, str(n)); return
def strip_content(slide):
    for sh in list(slide.shapes):
        if inch(sh.top) < 1.4 or (inch(sh.left) >= 18.4 and inch(sh.top) >= 9.7): continue
        sh._element.getparent().remove(sh._element)
def run_fmt(r, size, bold=False, color=TXT, italic=False):
    r.font.name = FONT; r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic; r.font.color.rgb = rgb(color)
def add_tb(slide, x, y, w, h, paras, size=22, color=TXT, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, bullet=False, after=6, name=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tb.name = name or "Text"
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align; p.space_after = Pt(after)
        parts = para if isinstance(para, list) else [(para, bold)]
        for t, b in parts:
            r = p.add_run(); r.text = t; run_fmt(r, size, b, color)
        if bullet:
            pPr = p._p.get_or_add_pPr(); pPr.set("marL", str(int(Inches(0.32)))); pPr.set("indent", str(-int(Inches(0.32))))
            bu = etree.SubElement(pPr, qn("a:buFont")); bu.set("typeface", "Arial"); bc = etree.SubElement(pPr, qn("a:buChar")); bc.set("char", "•")
    return tb
def add_card(slide, x, y, w, h, fill=CARD, line=EDGE, name="Card"):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); sh.name = name
    sh.adjustments[0] = 0.04; sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill); sh.line.color.rgb = rgb(line); sh.line.width = Pt(1.5); sh.shadow.inherit = False
    return sh
def add_pic(slide, path, x, y, w=None, h=None, name="Figure"):
    pic = slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None); pic.name = name; return pic
def cell_border(cell, color=EDGE, w=12700):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        for old in tcPr.findall(qn(tag)): tcPr.remove(old)
        ln = etree.SubElement(tcPr, qn(tag)); ln.set("w", str(w)); sf = etree.SubElement(ln, qn("a:solidFill")); c = etree.SubElement(sf, qn("a:srgbClr")); c.set("val", color)
    # borders must precede fill in tcPr
    fills = [e for e in tcPr if e.tag in (qn("a:solidFill"), qn("a:noFill"))]
    for f in fills: tcPr.remove(f); tcPr.append(f)
def add_table(slide, x, y, widths, rows, row_h=0.6, size=18, header=True, fills=None, bolds=None, name="Table", align_first_left=True):
    n_r, n_c = len(rows), len(widths)
    gf = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(sum(widths)), Inches(row_h * n_r)); gf.name = name; tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for a in ("firstRow", "bandRow"): tblPr.set(a, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None: sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"   # "No style, no grid"
    for j, w in enumerate(widths): tbl.columns[j].width = Inches(w)
    for i in range(n_r):
        tbl.rows[i].height = Inches(row_h)
        for j in range(n_c):
            c = tbl.cell(i, j); c.margin_left = c.margin_right = Inches(0.08); c.margin_top = c.margin_bottom = Inches(0.03); c.vertical_anchor = MSO_ANCHOR.MIDDLE
            hd = header and i == 0
            fill = NAVY if hd else ((fills or {}).get((i, j)) or (CARD if i % 2 == 1 else WHITE))
            c.fill.solid(); c.fill.fore_color.rgb = rgb(fill)
            tf = c.text_frame; tf.word_wrap = True; p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if (j == 0 and align_first_left and not hd and False) else PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(rows[i][j]); run_fmt(r, size, hd or bool((bolds or {}).get((i, j))), WHITE if hd else TXT)
            cell_border(c)
    return gf
def set_bullets(tb, items, size=None):
    tf = tb.text_frame; proto = copy.deepcopy(tf.paragraphs[0]._p)
    for p in list(tf.paragraphs): p._p.getparent().remove(p._p)
    for it in items:
        np_ = copy.deepcopy(proto); tf._txBody.append(np_)
        rs = np_.findall(qn("a:r"))
        for r in rs[1:]: np_.remove(r)
        rs[0].find(qn("a:t")).text = it
        if size: rs[0].find(qn("a:rPr")).set("sz", str(int(size * 100)))

# ------------------------------------------------------------------ data
fs = json.load(open(f"{ROOT}/paper/results/fewshot_with_accuracy.json"))["agg_mean_std"]
m1 = json.load(open(f"{ROOT}/outputs/experiments/phase1/metrics.json")); ab = json.load(open(f"{ROOT}/paper/results/ablation.json"))
fmt = lambda v, d=3: f"{v:.{d}f}"
SET = [("few_shot_only", "Few-shot data only (normal detection)"), ("few_shot_plus_random", "Random anomaly generation + detection"), ("few_shot_plus_structure_aware", "Sub-structure anomaly generation + detection")]

# ------------------------------------------------------------------ S3 objectives (index 2)
s = S[2]
titles = ["Build time-resolved brain graphs", "Track brain-region subgraphs", "Learn normal brain dynamics", "Detect anomalies", "Generate synthetic anomalies", "Evaluate few-shot and severity"]
descs = ["Convert 871 ABIDE I scans (360 regions) into sliding-window brain graphs.", "Follow 20 selected regions and their neighbours across time windows.", "Train a GIN + GRU on control scans only to predict the next window.",
         "Score prediction errors and set thresholds from validation controls.", "Gradient-guided edge edits on control subgraphs (7,200 records, 0 rule violations).", "Test 1, 5, 10 ASD scans (accuracy, F1, AUROC); severity and demo app."]
tbs = {sh.name: sh for sh in s.shapes if sh.shape_type == 17}
ttl_names = ["TextBox 12", "TextBox 19", "TextBox 26", "TextBox 33", "TextBox 40", "TextBox 47"]; dsc_names = ["TextBox 13", "TextBox 20", "TextBox 27", "TextBox 34", "TextBox 41", "TextBox 48"]
cards = [(0.81, 1.71), (10.19, 1.75), (0.81, 4.79), (10.3, 4.79), (0.81, 7.86), (10.3, 7.86)]
badge = [g for g in s.shapes if g.name == "Group 49"][0]
for i in range(6):
    t, d = tbs[ttl_names[i]], tbs[dsc_names[i]]
    set_runs(t.text_frame, titles[i]); set_runs(d.text_frame, descs[i])
    cx, cy = cards[i]; two = len(titles[i]) > 34
    t.top = Inches(cy + 0.16); t.height = Inches(1.0 if two else 0.6); d.top = Inches(cy + (1.2 if two else 0.8)); d.height = Inches(1.15)
    t.text_frame.word_wrap = True
    b = badge if i == 5 else None
    if b is None:
        b = copy.deepcopy(badge._element); s.shapes._spTree.append(b)
        from pptx.shapes.group import GroupShape
        b = GroupShape(b, s.shapes)
    b.left = Inches(cx + 8.91 - 3.52 - 0.2); b.top = Inches(cy + 2.72 - 0.48 - 0.12)
    for sh in find_text_shapes([b]):
        if sh.text_frame.text.strip(): set_runs(sh.text_frame, "✓ COMPLETED")
set_title(s, "OBJECTIVES")

# ------------------------------------------------------------------ S4 literature: remove development wording
tbl = [sh for sh in S[3].shapes if sh.has_table][0].table
for row in tbl.rows:
    for c in row.cells:
        for p_ in c.text_frame.paragraphs:
            for r in p_.runs:
                t = r.text
                for a_, b_ in [(" throughout M1.", " throughout."), (" (M1).", "."), (" (Phase 1).", "."), (" (Phase 2).", "."), (" (M1)", "")]: t = t.replace(a_, b_)
                r.text = t
for sh in find_text_shapes(S[3].shapes):
    for p_ in sh.text_frame.paragraphs:
        for r in p_.runs:
            if r.text.strip() == "Remaining": r.text = "Our"
            if r.text.strip() == "gap": r.text = "contribution"

# ------------------------------------------------------------------ S5 methodology
s = S[4]; strip_content(s); set_title(s, "PROPOSED METHODOLOGY")
add_pic(s, f"{FIG}/fig_flow_compact.png", 1.75, 1.5, w=16.5, name="End-to-end pipeline flowchart")
# ------------------------------------------------------------------ S6 contributions
s = S[5]; strip_content(s); set_title(s, "MAJOR CONTRIBUTIONS")
add_tb(s, 0.8, 1.55, 18.4, 0.7, ["Same pipeline order as the methodology; each box states what is new at that step."], size=24, color=NAVY, align=PP_ALIGN.CENTER)
add_pic(s, f"{FIG}/fig_flow_contrib.png", 0.8, 2.55, w=18.4, name="Contributions flowchart")

# ------------------------------------------------------------------ S7 results: detector
s = S[6]; strip_content(s); set_title(s, "RESULTS: DETECTOR")
add_pic(s, f"{AS}/fig_detector.png", 0.8, 1.65, w=12.4, name="Detector score distributions and ROC")
add_tb(s, 0.8, 6.7, 12.4, 0.5, ["Left: person scores with the control-calibrated threshold. Right: ROC curve with 95% bootstrap band (131 test patients)."], size=16, color="55606E")
add_tb(s, 13.5, 1.6, 5.7, 0.55, ["Detector on 131 test patients"], size=22, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
acc = (m1["true_positives"] + m1["true_negatives"]) / m1["num_test_subjects"]
rows = [["Metric", "Value"], ["Accuracy", f"{100*acc:.1f}%"], ["Precision", fmt(m1["precision"])], ["Recall", fmt(m1["recall"])], ["F1 score", fmt(m1["f1"])], ["AUROC", f"{m1['auroc']:.3f}"], ["AUROC 95% CI", "0.353 – 0.552"], ["Flagged", "4 of 131"]]
add_table(s, 13.5, 2.25, [2.9, 2.8], rows, row_h=0.56, size=19, name="Detector metrics")
add_tb(s, 0.8, 7.35, 18.4, 0.55, ["Does modelling time help? Three scoring methods on the same frozen encoder (131 test patients)"], size=22, bold=True, color=NAVY)
names = [("Static (distance to control mean)", "static"), ("Persistence (change from previous window)", "persist"), ("GRU prediction error (ours)", "gru")]
rows = [["Scoring method", "Accuracy", "Precision", "Recall", "F1", "AUROC [95% CI]"]]
for lab, k in names:
    d = ab["methods"][k]; tn = 70 - d["fp"]
    rows.append([lab, f"{100*(d['tp']+tn)/131:.1f}%", fmt(d["precision"]), fmt(d["recall"]), fmt(d["f1"]), f"{d['auroc']:.3f} [{d['auroc_ci95'][0]:.3f}, {d['auroc_ci95'][1]:.3f}]"])
add_table(s, 0.8, 7.95, [5.9, 2.0, 2.0, 2.0, 2.0, 4.5], rows, row_h=0.6, size=18, name="Scoring methods")

# ------------------------------------------------------------------ S8 results: few-shot comparison table
s = S[7]; strip_content(s); set_title(s, "RESULTS: FEW-SHOT COMPARISON")
M = ["accuracy", "precision", "recall", "f1", "auroc"]; head = ["Shots", "Approach", "Accuracy", "Precision", "Recall", "F1 score", "AUROC"]
rows = [head, ["–", "Baseline detector (no few-shot data)", f"{100*acc:.1f}%", fmt(m1["precision"]), fmt(m1["recall"]), fmt(m1["f1"]), fmt(m1["auroc"])]]
bolds, fills = {}, {}
for sh_i, shot in enumerate(["1-shot", "5-shot", "10-shot"]):
    best = {m: max(fs[shot][c][m][0] for c, _ in SET) for m in M}
    for c, lab in SET:
        r = [shot.split("-")[0] if c == SET[0][0] else "", lab]
        for m in M:
            mu, sd = fs[shot][c][m]; r.append((f"{100*mu:.1f}% ± {100*sd:.1f}" if m == "accuracy" else f"{mu:.3f} ± {sd:.2f}"))
            if abs(mu - best[m]) < 1e-12: bolds[(len(rows), len(r) - 1)] = True; fills[(len(rows), len(r) - 1)] = "DCEFE3"
        rows.append(r)
tb = add_table(s, 0.8, 1.6, [1.3, 6.5, 2.35, 2.05, 2.05, 2.05, 2.1], rows, row_h=0.6, size=17, bolds=bolds, fills=fills, name="Few-shot comparison")
t = tb.table
for g in range(3):  # merge the shots column per group
    a = 2 + 3 * g; t.cell(a, 0).merge(t.cell(a + 2, 0)); t.cell(a, 0).text_frame.paragraphs[0].runs[0].font.bold = True
add_tb(s, 0.8, 8.35, 18.4, 2.2, [
    [("Reading the table: ", True), ("mean ± SD over five support-set draws on 131 test patients (61 ASD, 70 control); green = best of the three approaches within each shot count.", False)],
    [("Generated anomalies raise recall and F1 (1-shot F1 0.006 → about 0.39–0.40), but AUROC stays near 0.50 and accuracy near 50%: ", False), ("random and sub-structure generation perform alike.", True)],
    [("Chance references: AUROC 0.5; accuracy 53.4% (always answer control); F1 of a random flagger at the same flag rate ≈ 0.40–0.42.", False)]],
    size=20, bullet=True, after=5)

# ------------------------------------------------------------------ S9 charts
s = S[8]; strip_content(s); set_title(s, "RESULTS: COMPARISON CHARTS")
cols = [NAVY, GOLD, GREEN]
for k, (metric, ttl, ymax) in enumerate([("accuracy", "Accuracy", 1.0), ("f1", "F1 score", 0.6), ("auroc", "AUROC", 1.0)]):
    cd = CategoryChartData(); cd.categories = ["1-shot", "5-shot", "10-shot"]
    for c, lab in SET: cd.add_series(lab.split(" (")[0].replace(" + detection", ""), [round(fs[f"{n}-shot"][c][metric][0], 3) for n in (1, 5, 10)])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.8 + 6.2 * k), Inches(1.65), Inches(6.0), Inches(6.2), cd); gf.name = f"{ttl} chart"; ch = gf.chart
    ch.has_title = True; ch.chart_title.text_frame.text = ttl; r0 = ch.chart_title.text_frame.paragraphs[0].runs[0]; run_fmt(r0, 22, True, NAVY)
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(14); ch.legend.font.name = FONT
    va = ch.value_axis; va.minimum_scale = 0; va.maximum_scale = ymax; va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = rgb("D9E3F0"); va.tick_labels.font.size = Pt(14); va.tick_labels.font.name = FONT
    va.format.line.fill.background(); ch.category_axis.tick_labels.font.size = Pt(16); ch.category_axis.tick_labels.font.name = FONT
    pl = ch.plots[0]; pl.gap_width = 60; pl.overlap = -5; pl.has_data_labels = True; dl = pl.data_labels; dl.number_format = "0.00"; dl.number_format_is_linked = False; dl.font.size = Pt(13); dl.font.name = FONT; dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for ser, col in zip(pl.series, cols): ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(col)
add_tb(s, 0.8, 8.1, 18.4, 2.4, [
    "Few-shot data only vs. random anomaly generation vs. sub-structure (gradient-guided) anomaly generation, mean of five draws.",
    "Anomaly generation raises F1 through a higher flag rate; AUROC and accuracy do not improve (chance: AUROC 0.5, accuracy 53.4%).",
    "Sub-structure generation is detected 45.8% of the time against 4.8% for random edits (generation test), yet gives no extra gain in the few-shot comparison."],
    size=20, bullet=True, after=5)

# ------------------------------------------------------------------ S10, S11 existing image slides: retitle
set_title(S[9], "RESULTS: PIPELINE OUTPUTS"); set_title(S[10], "RESULTS: SYNTHETIC ANOMALIES")

# ------------------------------------------------------------------ S12 severity + demo
s = S[11]; strip_content(s); set_title(s, "RESULTS: SEVERITY AND DEMO")
add_tb(s, 0.8, 1.6, 9.6, 0.6, ["Severity estimation (exploratory)"], size=26, bold=True, color=NAVY)
sv = json.load(open(f"{ROOT}/outputs/experiments/severity/severity_results.json"))
rows = [["Evaluation", "n", "Accuracy", "Macro-F1", "Majority guess"],
        ["Held-out ASD patients", str(sv["holdout"]["n_eval"]), f"{sv['holdout']['accuracy']:.3f}", f"{sv['holdout']['macro_f1']:.3f}", f"{sv['holdout']['majority_accuracy']:.3f}"],
        ["5-fold CV × 10", str(sv["n_asd_with_css"]), f"{sv['cv_5fold_x10']['accuracy_mean']:.3f}", f"{sv['cv_5fold_x10']['macro_f1_mean']:.3f}", f"{sv['cv_5fold_x10']['majority_accuracy_mean']:.3f}"]]
add_table(s, 0.8, 2.3, [3.3, 0.9, 1.6, 1.6, 2.2], rows, row_h=0.7, size=17, name="Severity results")
add_tb(s, 0.8, 4.65, 9.6, 5.9, [
    "Severity bands come from the ADOS-2 calibrated severity score: low (≤5), moderate (6–7), high (≥8).",
    "Estimate is below a majority guess (43.3% vs 55.0%): the scores carry no usable severity signal.",
    "Predicted vs true severity score: Spearman 0.13 (p = 0.32).",
    "Demo web app: upload a raw ROI time series, get the detector flag and a severity band.",
    "On 20 random held-out test files the output was right for 11 and the detector flagged 1."],
    size=21, bullet=True, after=8)
add_pic(s, f"{ROOT}/paper/deck/app_demo.png", 10.9, 1.6, w=8.3, name="Demo application screenshot")
add_tb(s, 10.9, 9.0, 8.3, 0.5, ["Demo web application (research prototype, runs locally)"], size=16, color="55606E", align=PP_ALIGN.CENTER)

# ------------------------------------------------------------------ S13 discussion
s = S[12]; tbs = {sh.name: sh for sh in s.shapes if sh.shape_type == 17}
set_bullets(tbs["TextBox 12"], [
    "Stable training with no representation collapse (embedding std 1.03–1.16).",
    "Generator works as designed: 7,200 records, 0 rule violations, severity-monotonic.",
    "Sub-structure edits detected 45.8% of the time vs 4.8% for random edits (9.6×).",
    "Leakage-safe protocol: patient-level split, thresholds from validation controls only.",
    "Complete pipeline plus a demo application and 20 sample scans."], 24)
set_bullets(tbs["TextBox 18"], [
    "Detector is at chance on ABIDE I: AUROC 0.453, accuracy 53.4%.",
    "A single step-level threshold flagged all 131 test patients; fixed with a patient-level threshold.",
    "Few-shot F1 gains come from flagging more patients; AUROC stays about 0.50.",
    "Severity estimate is below a majority guess (43.3% vs 55.0%).",
    "Only 20 of 360 regions, one training run, no site-level test."], 24)
set_bullets(tbs["TextBox 24"], [
    "A diagnosis label is a weak sign of unusual connectivity dynamics; ASD is highly varied.",
    "Site effects and short, noisy windows may hide a real signal.",
    "Detectability of generated anomalies is self-referential: the same model guides and scores the edits.",
    "Guided and random generation behave alike in the few-shot test.",
    "Research prototype: no clinical claim."], 24)
set_title(s, "DISCUSSION")
for g in s.shapes:
    if g.shape_type == 6 and inch(g.height) > 7.5 and inch(g.top) > 2: g.height = Inches(6.9)

# ------------------------------------------------------------------ S14 conclusion
s = S[13]; strip_content(s); set_title(s, "CONCLUSION")
add_card(s, 0.81, 1.7, 18.4, 7.4, name="Conclusion card")
add_tb(s, 1.3, 2.0, 17.4, 8.0, [
    "Built and tested an end-to-end pipeline: time-resolved brain graphs, brain-region subgraph tracking, temporal anomaly scoring, synthetic anomaly generation, few-shot adaptation and a severity output.",
    "All six objectives were completed, including a working demo web application.",
    "The anomaly generator works as designed: 0 rule violations, and 45.8% of sub-structure edits are detected against 4.8% for random edits.",
    "On ABIDE I the detector is at chance (AUROC 0.453, accuracy 53.4%); few-shot F1 gains come from a higher flag rate (AUROC about 0.50).",
    "Severity estimation is below a majority guess (43.3% vs 55.0%).",
    "Research prototype only: no clinical claim, and all results are reported as found."], size=30, bullet=True, after=22)

# ------------------------------------------------------------------ S15, S16 references
refs = json.load(open(f"{ROOT}/paper/deck/refs_all.json"))
for idx, (lo, hi, ttl) in zip((14, 15), [(0, 25, "REFERENCES (1/2)"), (25, 50, "REFERENCES (2/2)")]):
    s = S[idx]
    for g in list(s.shapes):                       # drop the bottom note pill (groups at y>9.5 except slide number / corner art)
        if g.shape_type == 6 and inch(g.top) > 9.5 and inch(g.left) < 5: g._element.getparent().remove(g._element)
    tbs = {sh.name: sh for sh in s.shapes if sh.shape_type == 17}; tb = tbs["TextBox 18"]
    tf = tb.text_frame; proto = copy.deepcopy(tf.paragraphs[0]._p)
    for p in list(tf.paragraphs): p._p.getparent().remove(p._p)
    for k in range(lo, hi):
        np_ = copy.deepcopy(proto); tf._txBody.append(np_); rs = np_.findall(qn("a:r"))
        for r in rs[1:]: np_.remove(r)
        rs[0].find(qn("a:t")).text = f"[{k+1}] {refs[k]}"; rs[0].find(qn("a:rPr")).set("sz", "1300")
        pPr = np_.find(qn("a:pPr"))
        if pPr is None: pPr = etree.Element(qn("a:pPr")); np_.insert(0, pPr)
        for ch in list(pPr): pPr.remove(ch)
        pPr.set("marL", str(int(Inches(0.45)))); pPr.set("indent", str(-int(Inches(0.45)))); pPr.set("algn", "l")
        ln = etree.SubElement(pPr, qn("a:lnSpc")); sp = etree.SubElement(ln, qn("a:spcPct")); sp.set("val", "100000")
        sa = etree.SubElement(pPr, qn("a:spcAft")); sp2 = etree.SubElement(sa, qn("a:spcPts")); sp2.set("val", "500")
        etree.SubElement(pPr, qn("a:buNone"))
    bp = tf._txBody.find(qn("a:bodyPr")); bp.set("numCol", "2"); bp.set("spcCol", str(int(Inches(0.5)))); bp.set("anchor", "t")
    tb.height = Inches(8.4); tb.top = Inches(2.0)
    for g in s.shapes:
        if g.shape_type == 6 and abs(inch(g.left) - 0.81) < 0.05 and inch(g.height) > 7: g.height = Inches(9.0); g.top = Inches(1.7)
    set_title(s, ttl)

# ------------------------------------------------------------------ slide numbers
for i, sl in enumerate(S, 1): set_number(sl, i)
prs.save(sys.argv[2]); print("saved", sys.argv[2], len(S), "slides")
