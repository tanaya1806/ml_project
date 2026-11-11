"""End-semester deck, fully editable (native shapes/charts/tables, no pictures), one common template.
usage (repo root): python3 paper/scripts/build_deck2.py base2.pptx out.pptx"""
import copy, csv, json, os, sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION, XL_MARKER_STYLE
from pptx.enum.dml import MSO_LINE
from pptx.oxml.ns import qn
from lxml import etree

ROOT = os.getcwd()
NAVY, TXT, CARD, EDGE, GREEN, GOLD, WHITE, GREY = "0D2257", "222A33", "E6F0FA", "AFC9E8", "2E6B4F", "A8752E", "FFFFFF", "7A8594"
FONT = "Times New Roman"; rgb = lambda h: RGBColor.from_string(h)
prs = Presentation(sys.argv[1]); S = list(prs.slides)
D = json.load(open(f"{ROOT}/paper/deck/deck_data.json"))
inch = lambda e: e / 914400

# ---------------------------------------------------------------- generic helpers
def walk_text(shapes):
    for sh in shapes:
        if sh.shape_type == 6: yield from walk_text(sh.shapes)
        elif sh.has_text_frame: yield sh
def set_runs(tf, text):
    if not tf.paragraphs[0].runs: return
    p = tf.paragraphs[0]
    for extra in tf.paragraphs[1:]: extra._p.getparent().remove(extra._p)
    p.runs[0].text = text
    for r in p.runs[1:]: r._r.getparent().remove(r._r)
def set_title(slide, new):
    for sh in walk_text(slide.shapes):
        if inch(sh.top) < 0.3 and sh.text_frame.text.strip() and inch(sh.height) > 1.2 and inch(sh.width) > 15: set_runs(sh.text_frame, new); return
    raise RuntimeError("no title")
def set_number(slide, n):
    for g in slide.shapes:
        if g.shape_type == 6 and inch(g.left) > 19 and inch(g.top) > 10.5:
            for sh in walk_text([g]):
                if sh.text_frame.text.strip().isdigit(): set_runs(sh.text_frame, str(n)); return
def strip_content(slide):
    for sh in list(slide.shapes):
        if inch(sh.top) < 1.4 or (inch(sh.left) >= 18.4 and inch(sh.top) >= 9.7): continue
        sh._element.getparent().remove(sh._element)
def fmt_run(r, size, bold=False, color=TXT, italic=False):
    r.font.name = FONT; r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic; r.font.color.rgb = rgb(color)
def fill_tf(tf, paras, size, color=TXT, bold=False, align=PP_ALIGN.LEFT, after=6, bullet=False):
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align; p.space_after = Pt(after)
        for t, b in (para if isinstance(para, list) else [(para, bold)]):
            r = p.add_run(); r.text = t; fmt_run(r, size, b, color)
        if bullet:
            pPr = p._p.get_or_add_pPr(); pPr.set("marL", str(int(Inches(0.32)))); pPr.set("indent", str(-int(Inches(0.32))))
            etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial"); etree.SubElement(pPr, qn("a:buChar")).set("char", "•")
def add_tb(slide, x, y, w, h, paras, size=22, color=TXT, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, bullet=False, after=6, name="Text"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tb.name = name; tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03); fill_tf(tf, paras, size, color, bold, align, after, bullet); return tb
def add_card(slide, x, y, w, h, fill=CARD, line=EDGE, name="Card"):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); sh.name = name; sh.adjustments[0] = 0.04
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill); sh.line.color.rgb = rgb(line); sh.line.width = Pt(1.5); sh.shadow.inherit = False; return sh
def cell_border(cell, color=EDGE, w=12700):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        for old in tcPr.findall(qn(tag)): tcPr.remove(old)
        ln = etree.SubElement(tcPr, qn(tag)); ln.set("w", str(w)); etree.SubElement(etree.SubElement(ln, qn("a:solidFill")), qn("a:srgbClr")).set("val", color)
    for f in [e for e in tcPr if e.tag in (qn("a:solidFill"), qn("a:noFill"))]: tcPr.remove(f); tcPr.append(f)
def add_table(slide, x, y, widths, rows, row_h=0.6, size=18, fills=None, bolds=None, name="Table"):
    gf = slide.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y), Inches(sum(widths)), Inches(row_h * len(rows))); gf.name = name; tbl = gf.table
    tp = tbl._tbl.tblPr
    for a in ("firstRow", "bandRow"): tp.set(a, "0")
    sid = tp.find(qn("a:tableStyleId"))
    if sid is not None: sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    for j, w in enumerate(widths): tbl.columns[j].width = Inches(w)
    for i in range(len(rows)):
        tbl.rows[i].height = Inches(row_h)
        for j in range(len(widths)):
            c = tbl.cell(i, j); c.margin_left = c.margin_right = Inches(0.08); c.margin_top = c.margin_bottom = Inches(0.03); c.vertical_anchor = MSO_ANCHOR.MIDDLE
            hd = i == 0; c.fill.solid(); c.fill.fore_color.rgb = rgb(NAVY if hd else ((fills or {}).get((i, j)) or (CARD if i % 2 == 1 else WHITE)))
            p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; c.text_frame.word_wrap = True
            r = p.add_run(); r.text = str(rows[i][j]); fmt_run(r, size, hd or bool((bolds or {}).get((i, j))), WHITE if hd else TXT); cell_border(c)
    return gf
def set_bullets(tb, items, size):
    tf = tb.text_frame; proto = copy.deepcopy(tf.paragraphs[0]._p)
    for p in list(tf.paragraphs): p._p.getparent().remove(p._p)
    for it in items:
        np_ = copy.deepcopy(proto); tf._txBody.append(np_); rs = np_.findall(qn("a:r"))
        for r in rs[1:]: np_.remove(r)
        rs[0].find(qn("a:t")).text = it; rs[0].find(qn("a:rPr")).set("sz", str(int(size * 100)))
# ---- native flowchart parts
def flow_box(slide, x, y, w, h, head, detail, hs=20, ds=16, fill=WHITE, name="Flow box"):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); sh.name = name; sh.adjustments[0] = 0.08
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill); sh.line.color.rgb = rgb(NAVY); sh.line.width = Pt(1.75); sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP; tf.margin_left = tf.margin_right = Inches(0.15); tf.margin_top = Inches(0.12); tf.margin_bottom = Inches(0.08)
    fill_tf(tf, [[(head, True)]], hs, NAVY, align=PP_ALIGN.CENTER, after=3)
    p = tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = detail; fmt_run(r, ds, False, TXT)
    return sh
def arrow(slide, a, ia, b, ib):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, 0, 0, 0, 0); c.begin_connect(a, ia); c.end_connect(b, ib)
    c.line.color.rgb = rgb(NAVY); c.line.width = Pt(2.5); ln = c.line._get_or_add_ln(); etree.SubElement(ln, qn("a:tailEnd")).set("type", "triangle"); c.name = "Flow arrow"; return c
def serpentine(slide, boxes, x0, y0, bw, gap, row_h, vgap):
    """3x3 snake: left-to-right, right-to-left, left-to-right; returns shapes"""
    sh = []; y = y0
    for r in range(3):
        xs = [x0 + c * (bw + gap) for c in range(3)]; xs = xs if r % 2 == 0 else xs[::-1]
        for c in range(3):
            i = 3 * r + c; hd, dt = boxes[i]; sh.append(flow_box(slide, xs[c], y, bw, row_h[r], hd, dt, name=f"Step {i+1}"))
        y += row_h[r] + vgap
    for i in range(8):
        r = i // 3
        if i % 3 != 2: arrow(slide, sh[i], 3 if r % 2 == 0 else 1, sh[i + 1], 1 if r % 2 == 0 else 3)
        else: arrow(slide, sh[i], 2, sh[i + 1], 0)
    return sh
def mini_graph(slide, g, x, y, w, h, name="Brain subgraph", node_d=0.11):
    grp = slide.shapes.add_group_shape(); grp.name = name; P = []
    xs = [n[0] for n in g["nodes"]]; ys = [n[1] for n in g["nodes"]]; mnx, mxx, mny, mxy = min(xs), max(xs), min(ys), max(ys)
    for n in g["nodes"]: P.append((x + (n[0] - mnx) / (mxx - mnx) * (w - 0.2) + 0.1, y + (n[1] - mny) / (mxy - mny) * (h - 0.2) + 0.1))
    for u, v, neg, bold in g["edges"]:
        c = grp.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(P[u][0]), Inches(P[u][1]), Inches(P[v][0]), Inches(P[v][1]))
        c.line.color.rgb = rgb("000000" if bold else "8B97A8"); c.line.width = Pt(2.6 if bold else 0.7)
        if neg and not bold: c.line.dash_style = MSO_LINE.DASH
    for (px, py), n in zip(P, g["nodes"]):
        d = node_d * (1.8 if n[2] else 1)
        o = grp.shapes.add_shape(MSO_SHAPE.OVAL, Inches(px - d / 2), Inches(py - d / 2), Inches(d), Inches(d)); o.fill.solid(); o.fill.fore_color.rgb = rgb(WHITE if n[2] else NAVY)
        o.line.color.rgb = rgb(NAVY); o.line.width = Pt(1.5 if n[2] else 0.25); o.shadow.inherit = False
    return grp
def result_card(slide, x, y, w, h):
    KK = h / 1.92; sc = D["scan"]; grp = slide.shapes.add_group_shape(); grp.name = "Demo result card"
    bg = grp.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); bg.adjustments[0] = 0.05; bg.fill.solid(); bg.fill.fore_color.rgb = rgb("F4F8FF"); bg.line.color.rgb = rgb(EDGE); bg.shadow.inherit = False
    def tbox(tx, ty, tw, th, text, size, bold=False, color=TXT, align=PP_ALIGN.LEFT, fill=None):
        s = grp.shapes.add_textbox(Inches(tx), Inches(ty), Inches(tw), Inches(th)); tf = s.text_frame; tf.word_wrap = True; tf.margin_left = tf.margin_right = Inches(0.04); tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.alignment = align; r = p.add_run(); r.text = text; fmt_run(r, size, bold, color)
        if fill: s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
        return s
    tbox(x + 0.1, y + KK * 0.05, w - 0.2, 0.2, "Result (demo application)", 11, True, NAVY)
    tbox(x + 0.1, y + KK * 0.3, w - 0.2, 0.26, "Detector output: flagged", 12, True, "B4551A", PP_ALIGN.CENTER, "FFF0E0")
    gx, gy, gw = x + 0.15, y + KK * 0.72, w - 0.3; mx = max(sc["subject_score"], sc["tau_subject"]) * 1.25
    b0 = grp.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(gx), Inches(gy), Inches(gw), Inches(0.12)); b0.fill.solid(); b0.fill.fore_color.rgb = rgb("D6E6FA"); b0.line.fill.background(); b0.shadow.inherit = False
    b1 = grp.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(gx), Inches(gy), Inches(gw * sc["subject_score"] / mx), Inches(0.12)); b1.fill.solid(); b1.fill.fore_color.rgb = rgb("2F7DE1"); b1.line.fill.background(); b1.shadow.inherit = False
    tk = grp.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(gx + gw * sc["tau_subject"] / mx), Inches(gy - 0.06), Inches(0.03), Inches(0.24)); tk.fill.solid(); tk.fill.fore_color.rgb = rgb("B4551A"); tk.line.fill.background(); tk.shadow.inherit = False
    tbox(x + 0.1, y + KK * 0.95, w - 0.2, 0.2, f"score {sc['subject_score']:.2f} | threshold {sc['tau_subject']:.2f}", 10, False, TXT, PP_ALIGN.CENTER)
    tbox(x + 0.1, y + KK * 1.2, w - 0.2, 0.2, "Severity band (exploratory)", 10, True, NAVY, PP_ALIGN.CENTER)
    for k, (lab, pr) in enumerate([("low", .07), ("mod.", .79), ("high", .14)]):
        bx = x + 0.2 + k * (w - 0.4) / 3; bw_ = (w - 0.4) / 3 - 0.1; hh = 0.3 * pr / .79
        r = grp.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(bx), Inches(y + KK * 1.72 - hh), Inches(bw_), Inches(max(hh, 0.03))); r.fill.solid(); r.fill.fore_color.rgb = rgb("2F7DE1" if lab == "mod." else "9DC2EC"); r.line.fill.background(); r.shadow.inherit = False
        tbox(bx - 0.05, y + KK * 1.74, bw_ + 0.1, 0.16, f"{lab} {int(pr*100)}%", 9, False, TXT, PP_ALIGN.CENTER)
    return grp
def style_chart(ch, title, legend=True, legsz=14):
    ch.has_title = True; ch.chart_title.text_frame.text = title; run = ch.chart_title.text_frame.paragraphs[0].runs[0]; fmt_run(run, 20, True, NAVY)
    ch.has_legend = legend
    if legend: ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(legsz); ch.legend.font.name = FONT
    for ax in (ch.value_axis, ch.category_axis):
        ax.tick_labels.font.size = Pt(14); ax.tick_labels.font.name = FONT
    ch.value_axis.has_major_gridlines = True; ch.value_axis.major_gridlines.format.line.color.rgb = rgb("D9E3F0"); ch.value_axis.format.line.fill.background()
def axis_title(ax, text, size=14):
    ax.has_title = True; ax.axis_title.text_frame.text = text; fmt_run(ax.axis_title.text_frame.paragraphs[0].runs[0], size, False, TXT)
def line_series_fmt(ser, color, width=2.5, dash=False, marker=True):
    ser.format.line.color.rgb = rgb(color); ser.format.line.width = Pt(width); ser.smooth = False
    if dash: ser.format.line.dash_style = MSO_LINE.DASH
    if marker: ser.marker.style = XL_MARKER_STYLE.CIRCLE; ser.marker.size = 7; ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = rgb(color); ser.marker.format.line.color.rgb = rgb(color)
    else: ser.marker.style = XL_MARKER_STYLE.NONE

fs = json.load(open(f"{ROOT}/paper/results/fewshot_with_accuracy.json"))["agg_mean_std"]
m1 = json.load(open(f"{ROOT}/outputs/experiments/phase1/metrics.json")); ab = json.load(open(f"{ROOT}/paper/results/ablation.json"))
sv = json.load(open(f"{ROOT}/outputs/experiments/severity/severity_results.json"))
acc = (m1["true_positives"] + m1["true_negatives"]) / m1["num_test_subjects"]; f3 = lambda v: f"{v:.3f}"
SET = [("few_shot_only", "Few-shot data only (normal detection)"), ("few_shot_plus_random", "Random anomaly generation + detection"), ("few_shot_plus_structure_aware", "Sub-structure anomaly generation + detection")]
SHORT = {"few_shot_only": "Few-shot data only", "few_shot_plus_random": "Random anomaly generation", "few_shot_plus_structure_aware": "Sub-structure anomaly generation"}

# ================================================================ 3 objectives (edit in place)
s = S[2]; tbs = {sh.name: sh for sh in s.shapes if sh.shape_type == 17}
titles = ["Build time-resolved brain graphs", "Track brain-region subgraphs", "Learn normal brain dynamics", "Detect anomalies", "Generate synthetic anomalies", "Evaluate few-shot and severity"]
descs = ["Convert 871 ABIDE I scans (360 regions) into sliding-window brain graphs.", "Follow 20 selected regions and their neighbours across time windows.", "Train a GIN + GRU on control scans only to predict the next window.",
         "Score prediction errors and set thresholds from validation controls.", "Gradient-guided edge edits on control subgraphs (7,200 records, 0 rule violations).", "Test 1, 5, 10 ASD scans (accuracy, F1, AUROC); severity and demo app."]
cards = [(0.81, 1.71), (10.19, 1.75), (0.81, 4.79), (10.3, 4.79), (0.81, 7.86), (10.3, 7.86)]
badge = [g for g in s.shapes if g.name == "Group 49"][0]
for i in range(6):
    t, d = tbs[["TextBox 12", "TextBox 19", "TextBox 26", "TextBox 33", "TextBox 40", "TextBox 47"][i]], tbs[["TextBox 13", "TextBox 20", "TextBox 27", "TextBox 34", "TextBox 41", "TextBox 48"][i]]
    set_runs(t.text_frame, titles[i]); set_runs(d.text_frame, descs[i]); cx, cy = cards[i]
    t.top = Inches(cy + 0.16); t.height = Inches(0.6); d.top = Inches(cy + 0.8); d.height = Inches(1.15); t.text_frame.word_wrap = True
    if i < 5:
        from pptx.shapes.group import GroupShape
        el = copy.deepcopy(badge._element); s.shapes._spTree.append(el); b = GroupShape(el, s.shapes)
    else: b = badge
    b.left = Inches(cx + 8.91 - 3.52 - 0.2); b.top = Inches(cy + 2.72 - 0.48 - 0.12)
    for sh in walk_text([b]):
        if sh.text_frame.text.strip(): set_runs(sh.text_frame, "✓ COMPLETED")
set_title(s, "OBJECTIVES")

# ================================================================ 4 literature wording
tbl = [sh for sh in S[3].shapes if sh.has_table][0].table
for row in tbl.rows:
    for c in row.cells:
        for p_ in c.text_frame.paragraphs:
            for r in p_.runs:
                t = r.text
                for a_, b_ in [(" throughout M1.", " throughout."), (" (M1).", "."), (" (Phase 1).", "."), (" (Phase 2).", "."), (" (M1)", "")]: t = t.replace(a_, b_)
                r.text = t
for sh in walk_text(S[3].shapes):
    for p_ in sh.text_frame.paragraphs:
        for r in p_.runs:
            if r.text.strip() == "Remaining": r.text = "Our"
            if r.text.strip() == "gap": r.text = "contribution"

# ================================================================ 5 methodology flowchart (native)
s = S[4]; strip_content(s); set_title(s, "PROPOSED METHODOLOGY")
METH = [("DATA PREPARATION AND PATIENT SPLIT", "ABIDE I, 871 patient scans"),
        ("SLIDING-WINDOW BRAIN GRAPH BUILDING", "30-TR windows, stride 10; Pearson correlation; signed graph, top 20% edges"),
        ("BRAIN-REGION SUBGRAPH TRACKING", "20 selected brain regions (ROIs), 1-hop neighbourhood; node set fixed at the first window"),
        ("GIN ENCODING AND GRU NEXT-WINDOW PREDICTION", "2-layer weighted GIN, mean pool, 128-D embedding; GRU (hidden 128), MSE + variance loss, controls only"),
        ("TEMPORAL SUBGRAPH ANOMALY SCORING FOR ASD DETECTION", "patient score = max over regions and transitions"),
        ("CONTROL-BASED DETECTION THRESHOLD CALIBRATION", "95th percentile of validation-control scores (transition and patient level)"),
        ("SYNTHETIC SUBGRAPH ANOMALY GENERATION", "Gradient-guided PGD edge-weight edits, 5 steps, up to 15% of edge pairs, 3 severities"),
        ("FEW-SHOT ASD ADAPTATION", "1 / 5 / 10 ASD patient scans + 30 generated anomalies; logistic regression on 20-D region scores"),
        ("ANOMALY DETECTION AND SEVERITY OUTPUT", "Flag if score exceeds threshold; ADOS-2 severity band")]
RH = [2.95, 2.05, 3.4]; X0, BW, GAPX, VG = 0.8, 5.8, 0.5, 0.4
bx = serpentine(s, METH, X0, 1.45, BW, GAPX, RH, VG)
for sh in s.shapes:
    if sh.name.startswith('Step') and sh.has_text_frame:
        for p_ in sh.text_frame.paragraphs[1:]:
            for r_ in p_.runs: r_.font.size = Pt(18)
# row 1, box 1: two native brain subgraphs
y1 = 1.45 + 1.15
mini_graph(s, D["ctl"], X0 + 0.25, y1, 2.55, 1.35, "Control patient subgraph"); mini_graph(s, D["asd"], X0 + 3.0, y1, 2.55, 1.35, "ASD patient subgraph")
add_tb(s, X0 + 0.25, y1 + 1.35, 2.55, 0.3, ["control patient"], 13, GREY, align=PP_ALIGN.CENTER, name="Caption"); add_tb(s, X0 + 3.0, y1 + 1.35, 2.55, 0.3, ["ASD patient"], 13, GREY, align=PP_ALIGN.CENTER, name="Caption")
# row 3, box 9 (right end of row 3 = last box is at column 0 of snake? row 3 goes left->right so last box is rightmost)
rx = X0 + 2 * (BW + GAPX); ry = 1.45 + RH[0] + RH[1] + 2 * VG
for sh in s.shapes:
    if sh.name == 'Step 9':
        for p_ in sh.text_frame.paragraphs[1:]:
            for r_ in p_.runs: r_.font.size = Pt(16)
result_card(s, rx + 0.15, ry + 1.6, 2.55, 1.7)
mini_graph(s, D["synth_after"], rx + 2.85, ry + 1.6, 2.7, 1.45, "Synthetic anomaly subgraph")
add_tb(s, rx + 2.85, ry + 3.05, 2.7, 0.3, ["synthetic anomaly (bold = edited)"], 12, GREY, align=PP_ALIGN.CENTER, name="Caption")
# ================================================================ 6 contributions flowchart (native)
s = S[5]; strip_content(s); set_title(s, "MAJOR CONTRIBUTIONS")
CONTRIB = [("TIME-RESOLVED BRAIN GRAPHS", "One signed brain network per 30-TR window instead of a single static network"),
           ("BRAIN-REGION SUBGRAPH TRACKING", "20 regions with fixed 1-hop node sets followed over time, so an anomaly is localised in region and time"),
           ("CONTROL-ONLY TEMPORAL MODEL", "GIN + GRU predict the next window from control scans only; no labelled anomalies needed for training"),
           ("TEMPORAL SUBGRAPH ANOMALY SCORING", "Prediction error per region and step; patient score = maximum over regions and steps"),
           ("TWO-LEVEL THRESHOLD CALIBRATION", "Step and patient thresholds from validation controls; fixed a failure that flagged all 131 test patients"),
           ("GRADIENT-GUIDED ANOMALY GENERATION", "Weight-only edits of real control subgraphs; 45.8% detected against 4.8% for random edits"),
           ("GENERATION VALIDITY CHECKS", "7,200 records, 0 rule violations, severity-monotonic, edit diversity close to 1.0"),
           ("HONEST FEW-SHOT EVALUATION", "Chance-matched F1 shows the gain comes from the flag rate, not from ranking (AUROC about 0.5)"),
           ("SEVERITY ESTIMATE AND DEMO APPLICATION", "ADOS-based severity bands (no better than a majority guess) and a web app for raw time-series upload")]
add_tb(s, 0.8, 1.45, 18.4, 0.6, ["Same pipeline order as the methodology; each box states what is new at that step."], 24, NAVY, align=PP_ALIGN.CENTER)
serpentine(s, CONTRIB, 0.8, 2.3, 5.8, 0.5, [2.3, 2.3, 2.3], 0.8)
for sh in s.shapes:   # larger text for the contribution boxes
    if sh.name.startswith("Step") and sh.has_text_frame:
        for p in sh.text_frame.paragraphs[1:]:
            for r in p.runs: r.font.size = Pt(20)
        for r in sh.text_frame.paragraphs[0].runs: r.font.size = Pt(22)

# ================================================================ 7 results: detector (native charts)
s = S[6]; strip_content(s); set_title(s, "RESULTS: DETECTOR")
H = D["hist"]; labs = [f"{H['bins'][i]:.0f}–{H['bins'][i+1]:.0f}" for i in range(len(H["bins"]) - 1)]
cd = CategoryChartData(); cd.categories = labs
for k, lab in (("val_control", "validation controls"), ("test_control", "test controls"), ("test_asd", "test ASD")): cd.add_series(lab, [round(100 * v, 1) for v in H[k]])
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.8), Inches(1.55), Inches(9.0), Inches(4.8), cd).chart; s.shapes[-1].name = "Score distribution chart"; style_chart(ch, "Patient score distribution (% of group)", legsz=13)
for ser, col in zip(ch.plots[0].series, ["8B97A8", "1B7F79", "B5651D"]): ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(col)
ch.plots[0].gap_width = 40; axis_title(ch.category_axis, "patient score (threshold 19.81)", 14); axis_title(ch.value_axis, "% of group", 14)
xd = XyChartData()
for nm, ys in (("GRU error", D["roc"]["gru"]), ("95% lower", D["roc"]["lo"]), ("95% upper", D["roc"]["hi"]), ("chance", D["roc"]["grid"])):
    ser = xd.add_series(nm)
    for xv, yv in zip(D["roc"]["grid"], ys): ser.add_data_point(round(xv, 3), round(yv, 3))
ch = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(10.0), Inches(1.55), Inches(9.2), Inches(4.8), xd).chart; s.shapes[-1].name = "ROC chart"; style_chart(ch, f"ROC curve (AUROC {m1['auroc']:.3f})", legsz=13)
for ser, (col, w, dash) in zip(ch.plots[0].series, [(NAVY, 3.0, False), ("9DC2EC", 1.5, True), ("9DC2EC", 1.5, True), ("7A8594", 1.5, True)]): line_series_fmt(ser, col, w, dash, marker=False)
ch.value_axis.minimum_scale = 0; ch.value_axis.maximum_scale = 1; ch.category_axis.minimum_scale = 0; ch.category_axis.maximum_scale = 1; ch.category_axis.has_major_gridlines = False
axis_title(ch.category_axis, "false-positive rate", 14); axis_title(ch.value_axis, "true-positive rate", 14)
add_tb(s, 0.8, 6.55, 18.4, 0.55, ["Does modelling time help? Three scoring methods on the same frozen encoder (131 test patients)"], 22, NAVY, bold=True)
rows = [["Scoring method", "Accuracy", "Precision", "Recall", "F1", "AUROC [95% CI]"]]
for lab, k in [("Static (distance to control mean)", "static"), ("Persistence (change from previous window)", "persist"), ("GRU prediction error (ours, the detector)", "gru")]:
    d = ab["methods"][k]; tn = 70 - d["fp"]
    rows.append([lab, f"{100*(d['tp']+tn)/131:.1f}%", f3(d["precision"]), f3(d["recall"]), f3(d["f1"]), f"{d['auroc']:.3f} [{d['auroc_ci95'][0]:.3f}, {d['auroc_ci95'][1]:.3f}]"])
add_table(s, 0.8, 7.2, [6.4, 2.0, 2.0, 2.0, 2.0, 4.0], rows, row_h=0.62, size=19, name="Scoring methods")
add_tb(s, 0.8, 9.8, 18.4, 0.6, ["The detector flagged 4 of 131 patients; all three scores are at chance (AUROC 0.45–0.49, overlapping intervals)."], 18, GREY)

# ================================================================ 8 results: few-shot table
s = S[7]; strip_content(s); set_title(s, "RESULTS: FEW-SHOT COMPARISON")
M = ["accuracy", "precision", "recall", "f1", "auroc"]; head = ["Shots", "Approach", "Accuracy", "Precision", "Recall", "F1 score", "AUROC"]
rows = [head, ["–", "Baseline detector (no few-shot data)", f"{100*acc:.1f}%", f3(m1["precision"]), f3(m1["recall"]), f3(m1["f1"]), f3(m1["auroc"])]]; bolds, fills = {}, {}
for shot in ["1-shot", "5-shot", "10-shot"]:
    best = {m: max(fs[shot][c][m][0] for c, _ in SET) for m in M}
    for c, lab in SET:
        r = [shot.split("-")[0] if c == SET[0][0] else "", lab]
        for m in M:
            mu, sd = fs[shot][c][m]; r.append(f"{100*mu:.1f}% ± {100*sd:.1f}" if m == "accuracy" else f"{mu:.3f} ± {sd:.2f}")
            if abs(mu - best[m]) < 1e-12: bolds[(len(rows), len(r) - 1)] = True; fills[(len(rows), len(r) - 1)] = "DCEFE3"
        rows.append(r)
tb = add_table(s, 0.8, 1.55, [1.3, 6.5, 2.35, 2.05, 2.05, 2.05, 2.1], rows, row_h=0.6, size=17, bolds=bolds, fills=fills, name="Few-shot comparison")
for g in range(3):
    a = 2 + 3 * g; tb.table.cell(a, 0).merge(tb.table.cell(a + 2, 0)); tb.table.cell(a, 0).text_frame.paragraphs[0].runs[0].font.bold = True
add_tb(s, 0.8, 8.3, 18.4, 2.2, [
    [("Reading the table: ", True), ("mean ± SD over five support-set draws on 131 test patients (61 ASD, 70 control); green = best of the three approaches within each shot count.", False)],
    "Generated anomalies raise recall and F1 (1-shot F1 0.006 → about 0.39–0.40), but AUROC stays near 0.50 and accuracy near 50%: random and sub-structure generation perform alike.",
    "Chance references: AUROC 0.5; accuracy 53.4% (always answer control); F1 of a random flagger at the same flag rate ≈ 0.40–0.42."], 20, bullet=True, after=5)

# ================================================================ 9 results: comparison charts
s = S[8]; strip_content(s); set_title(s, "RESULTS: COMPARISON CHARTS")
for k, (metric, ttl, ymax) in enumerate([("accuracy", "Accuracy", 1.0), ("f1", "F1 score", 0.6), ("auroc", "AUROC", 1.0)]):
    cd = CategoryChartData(); cd.categories = ["1-shot", "5-shot", "10-shot"]
    for c, _ in SET: cd.add_series(SHORT[c], [round(fs[f"{n}-shot"][c][metric][0], 3) for n in (1, 5, 10)])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.8 + 6.2 * k), Inches(1.6), Inches(6.0), Inches(6.2), cd); gf.name = f"{ttl} chart"; ch = gf.chart; style_chart(ch, ttl, legsz=13)
    ch.value_axis.minimum_scale = 0; ch.value_axis.maximum_scale = ymax
    pl = ch.plots[0]; pl.gap_width = 60; pl.overlap = -5; pl.has_data_labels = True; dl = pl.data_labels; dl.number_format = "0.00"; dl.number_format_is_linked = False; dl.font.size = Pt(13); dl.font.name = FONT; dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for ser, col in zip(pl.series, [NAVY, GOLD, GREEN]): ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(col)
add_tb(s, 0.8, 8.1, 18.4, 2.4, ["Few-shot data only vs. random anomaly generation vs. sub-structure (gradient-guided) anomaly generation, mean of five draws.",
    "Anomaly generation raises F1 through a higher flag rate; AUROC and accuracy do not improve (chance: AUROC 0.5, accuracy 53.4%).",
    "Sub-structure generation is detected 45.8% of the time against 4.8% for random edits (generation test), yet gives no extra gain in the few-shot comparison."], 20, bullet=True, after=5)

# ================================================================ 10 pipeline outputs for one scan (native charts)
s = S[9]; strip_content(s); set_title(s, "RESULTS: OUTPUT FOR ONE SCAN"); sc = D["scan"]
n = len(sc["top_seed_scores"]); cd = CategoryChartData(); cd.categories = [str(i + 1) for i in range(n)]
cd.add_series("transition score", [round(v, 2) for v in sc["top_seed_scores"]]); cd.add_series("step threshold", [round(sc["tau_transition"], 2)] * n)
ch = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.8), Inches(1.6), Inches(9.0), Inches(5.6), cd).chart; s.shapes[-1].name = "Transition scores chart"; style_chart(ch, f"Transition scores, region {sc['top_seed']}")
line_series_fmt(ch.plots[0].series[0], NAVY, 3); line_series_fmt(ch.plots[0].series[1], "B4551A", 2, True, marker=False); axis_title(ch.category_axis, "window transition"); axis_title(ch.value_axis, "anomaly score")
cd = CategoryChartData(); cd.categories = [str(i + 1) for i in range(20)]; cd.add_series("highest score per region", [round(v, 2) for v in sc["seed_max"]]); cd.add_series("patient threshold", [round(sc["tau_subject"], 2)] * 20)
ch = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(10.0), Inches(1.6), Inches(9.2), Inches(5.6), cd).chart; s.shapes[-1].name = "Per-region scores chart"; style_chart(ch, "Highest score for each of the 20 tracked regions")
line_series_fmt(ch.plots[0].series[0], GREEN, 3); line_series_fmt(ch.plots[0].series[1], "B4551A", 2, True, marker=False); axis_title(ch.category_axis, "tracked region (1–20)"); axis_title(ch.value_axis, "highest score")
add_tb(s, 0.8, 7.5, 18.4, 3.0, [f"Example: one held-out ASD test scan ({sc['n_timepoints']} time points, {sc['n_windows']} windows); the scan the demo application opens.",
    f"Patient score {sc['subject_score']:.2f} against threshold {sc['tau_subject']:.2f}: the scan is flagged; {sc['n_flagged_transitions']} of {sc['n_transitions']} transitions exceed the step threshold.",
    "A typical scan scores below the threshold: on the 131 test patients only 4 were flagged."], 21, bullet=True, after=7)

# ================================================================ 11 synthetic anomalies (native)
s = S[10]; strip_content(s); set_title(s, "RESULTS: SYNTHETIC ANOMALIES")
cd = CategoryChartData(); cd.categories = ["low (0.25)", "medium (0.55)", "high (0.9)"]; cd.add_series("sub-structure (guided)", [round(v, 2) for v in D["severity_curve"]["structure_aware"]]); cd.add_series("random", [round(v, 2) for v in D["severity_curve"]["random"]]); cd.add_series("step threshold", [round(D["tau_transition"], 2)] * 3)
ch = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.8), Inches(1.55), Inches(6.2), Inches(4.4), cd).chart; s.shapes[-1].name = "Severity chart"; style_chart(ch, "Score after editing", legsz=12)
line_series_fmt(ch.plots[0].series[0], "6A3D9A", 3); line_series_fmt(ch.plots[0].series[1], "8B97A8", 3); line_series_fmt(ch.plots[0].series[2], "B4551A", 2, True, marker=False); axis_title(ch.category_axis, "edit severity", 13)
dt = D["detect"]; keys = [("real_normal_val_control", "real control"), ("real_asd_test", "real ASD"), ("random/high", "random, high"), ("structure_aware/low", "guided, low"), ("structure_aware/medium", "guided, medium"), ("structure_aware/high", "guided, high")]
cd = CategoryChartData(); cd.categories = [l for _, l in keys]; cd.add_series("steps above threshold (%)", [round(100 * dt[k], 1) for k, _ in keys])
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(7.2), Inches(1.55), Inches(6.4), Inches(4.4), cd).chart; s.shapes[-1].name = "Detection rate chart"; style_chart(ch, "Steps above threshold (%)", legend=False)
pl = ch.plots[0]; pl.gap_width = 50; pl.has_data_labels = True; pl.data_labels.number_format = '0.0'; pl.data_labels.number_format_is_linked = False; pl.data_labels.font.size = Pt(13); pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
for i, col in enumerate(["1B7F79", "B5651D", "8B97A8", "6A3D9A", "6A3D9A", "6A3D9A"]):
    pt = pl.series[0].points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(col)
ch.category_axis.tick_labels.font.size = Pt(12)
mini_graph(s, D["synth_before"], 13.8, 1.7, 2.7, 3.7, "Real control subgraph"); mini_graph(s, D["synth_after"], 16.55, 1.7, 2.7, 3.7, "Synthetic anomaly subgraph")
add_tb(s, 13.8, 5.4, 2.7, 0.5, ["real control"], 14, GREY, align=PP_ALIGN.CENTER, name="Caption"); add_tb(s, 16.55, 5.4, 2.7, 0.5, ["after guided edit (bold)"], 14, GREY, align=PP_ALIGN.CENTER, name="Caption")
si = D["synth_info"]
rows = [["Patient (control)", "Method", "Severity", "Edge pairs edited", "Score before", "Score after"]]
for r in D["records"]: rows.append([r["subject_id"], "sub-structure" if r["method"] == "structure_aware" else "random", r["severity_name"], f"{r['num_edges_changed']} / {r['num_undirected_edges']}", f"{r['anomaly_score_before']:.2f}", f"{r['anomaly_score_after']:.2f}"])
add_table(s, 0.8, 6.3, [4.0, 3.2, 2.6, 3.4, 2.6, 2.6], rows, row_h=0.5, size=17, name="Generated records")
add_tb(s, 0.8, 9.9, 18.4, 0.6, [f"Six real generated records (each traceable to an exact subject, region and window). Drawing shows 34 nodes of one subgraph; {si['shown_edited']} of the {si['edited_pairs']} edited edges are visible (score {si['score_before']:.2f} → {si['score_after']:.2f})."], 16, GREY)

# ================================================================ 12 severity (native)
s = S[11]; strip_content(s); set_title(s, "RESULTS: SEVERITY ESTIMATION")
rows = [["Evaluation", "n", "Accuracy", "Macro-F1", "Majority guess"],
        ["Held-out ASD patients", str(sv["holdout"]["n_eval"]), f3(sv["holdout"]["accuracy"]), f3(sv["holdout"]["macro_f1"]), f3(sv["holdout"]["majority_accuracy"])],
        ["5-fold CV × 10", str(sv["n_asd_with_css"]), f3(sv["cv_5fold_x10"]["accuracy_mean"]), f3(sv["cv_5fold_x10"]["macro_f1_mean"]), f3(sv["cv_5fold_x10"]["majority_accuracy_mean"])]]
add_table(s, 0.8, 1.7, [4.2, 1.0, 1.9, 1.9, 2.6], rows, row_h=0.75, size=19, name="Severity results")
add_tb(s, 0.8, 4.4, 11.6, 6.0, ["Severity bands come from the ADOS-2 calibrated severity score: low (≤5), moderate (6–7), high (≥8).",
    "ADOS scores exist for 193 of 403 ASD patients; the estimator is trained on training patients only.",
    "The estimate is below a majority guess (43.3% vs 55.0%): the scores carry no usable severity signal.",
    "Predicted vs true severity score: Spearman 0.13 (p = 0.32).", "Severity is only estimated for scans that the detector flags."], 23, bullet=True, after=10)
cd = CategoryChartData(); cd.categories = ["Held-out ASD", "5-fold CV"]; cd.add_series("severity estimate", [round(sv["holdout"]["accuracy"], 3), round(sv["cv_5fold_x10"]["accuracy_mean"], 3)]); cd.add_series("majority guess", [round(sv["holdout"]["majority_accuracy"], 3), round(sv["cv_5fold_x10"]["majority_accuracy_mean"], 3)])
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(13.0), Inches(1.6), Inches(6.2), Inches(7.0), cd).chart; s.shapes[-1].name = "Severity accuracy chart"; style_chart(ch, "Severity accuracy", legsz=14)
ch.value_axis.minimum_scale = 0; ch.value_axis.maximum_scale = 0.8; pl = ch.plots[0]; pl.gap_width = 70; pl.has_data_labels = True; pl.data_labels.number_format = "0.00"; pl.data_labels.number_format_is_linked = False; pl.data_labels.font.size = Pt(14); pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
for ser, col in zip(pl.series, [NAVY, GOLD]): ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(col)

# ================================================================ 13 demo application (native table of the 20 sample files)
s = S[12]; strip_content(s); set_title(s, "RESULTS: DEMO APPLICATION")
kr = list(csv.DictReader(open(f"{ROOT}/testing_data/answer_key_and_results.csv")))
rows = [["File", "True label", "ADOS band", "Score", "Detector output", "Detector right?"]]; fills = {}
for r in kr:
    ok = r["detector_correct"] == "True"; rows.append([r["file"].replace(".csv", ""), r["true_label"], r["true_severity_band"] or "–", r["subject_score"], r["detector_output"], "yes" if ok else "no"]); fills[(len(rows) - 1, 5)] = "DCEFE3" if ok else "FBE3DC"
add_table(s, 0.8, 1.55, [3.2, 2.6, 2.8, 2.2, 3.6, 3.0], rows, row_h=0.355, size=13, fills=fills, name="Demo sample results")
n_ok = sum(r["detector_correct"] == "True" for r in kr)
add_tb(s, 0.8, 9.2, 17.4, 1.3, [f"Demo web application: upload a raw ROI time series and get the detector flag and a severity band. On 20 random held-out test files (10 control, 10 ASD) the output was right for {n_ok} and the detector flagged {sum(r['detector_output']=='flagged' for r in kr)}.", "Files were drawn at random, never chosen by model output; the table shows every result, including the wrong ones."], 19, bullet=True, after=5)

# ================================================================ 14 discussion (existing slide, edit text)
s = S[13]; tbs = {sh.name: sh for sh in s.shapes if sh.shape_type == 17}
set_bullets(tbs["TextBox 12"], ["Stable training with no representation collapse (embedding std 1.03–1.16).", "Generator works as designed: 7,200 records, 0 rule violations, severity-monotonic.", "Sub-structure edits detected 45.8% of the time vs 4.8% for random edits (9.6×).", "Leakage-safe protocol: patient-level split, thresholds from validation controls only.", "Complete pipeline plus a demo application and 20 sample scans."], 24)
set_bullets(tbs["TextBox 18"], ["Detector is at chance on ABIDE I: AUROC 0.453, accuracy 53.4%.", "A single step-level threshold flagged all 131 test patients; fixed with a patient-level threshold.", "Few-shot F1 gains come from flagging more patients; AUROC stays about 0.50.", "Severity estimate is below a majority guess (43.3% vs 55.0%).", "Only 20 of 360 regions, one training run, no site-level test."], 24)
set_bullets(tbs["TextBox 24"], ["A diagnosis label is a weak sign of unusual connectivity dynamics; ASD is highly varied.", "Site effects and short, noisy windows may hide a real signal.", "Detectability of generated anomalies is self-referential: the same model guides and scores the edits.", "Guided and random generation behave alike in the few-shot test.", "Research prototype: no clinical claim."], 24)
set_title(s, "DISCUSSION")
for g in s.shapes:
    if g.shape_type == 6 and inch(g.height) > 7.5 and inch(g.top) > 2: g.height = Inches(6.9)

# ================================================================ 15 conclusion
s = S[14]; strip_content(s); set_title(s, "CONCLUSION"); add_card(s, 0.81, 1.7, 18.4, 7.4, name="Conclusion card")
add_tb(s, 1.3, 2.0, 17.4, 6.8, ["Built and tested an end-to-end pipeline: time-resolved brain graphs, brain-region subgraph tracking, temporal anomaly scoring, synthetic anomaly generation, few-shot adaptation and a severity output.",
    "All six objectives were completed, including a working demo web application.", "The anomaly generator works as designed: 0 rule violations, and 45.8% of sub-structure edits are detected against 4.8% for random edits.",
    "On ABIDE I the detector is at chance (AUROC 0.453, accuracy 53.4%); few-shot F1 gains come from a higher flag rate (AUROC about 0.50).", "Severity estimation is below a majority guess (43.3% vs 55.0%).",
    "Research prototype only: no clinical claim, and all results are reported as found."], 30, bullet=True, after=22)

# ================================================================ 16-17 references (native text, same template)
refs = json.load(open(f"{ROOT}/paper/deck/refs_all.json"))
for idx, (lo, hi, ttl) in zip((15, 16), [(0, 25, "REFERENCES (1/2)"), (25, 50, "REFERENCES (2/2)")]):
    s = S[idx]; strip_content(s); set_title(s, ttl); add_card(s, 0.81, 1.7, 18.4, 9.0, name="References card")
    tb = add_tb(s, 1.1, 1.85, 17.8, 8.7, [f"[{k+1}] {refs[k]}" for k in range(lo, hi)], 13, after=5, name="Reference list")
    for p in tb.text_frame.paragraphs:
        pPr = p._p.get_or_add_pPr(); pPr.set("marL", str(int(Inches(0.45)))); pPr.set("indent", str(-int(Inches(0.45))))
    bp = tb.text_frame._txBody.find(qn("a:bodyPr")); bp.set("numCol", "2"); bp.set("spcCol", str(int(Inches(0.5))))

for i, sl in enumerate(S, 1): set_number(sl, i)
prs.save(sys.argv[2]); print("saved", len(S), "slides")
