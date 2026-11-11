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
def serpentine(slide, boxes, x0, y0, bw, gap, row_h, vgap, hs=None):
    """3x3 snake; hs = optional per-box heights (boxes are centred vertically in their row)"""
    sh = []; y = y0
    for r in range(3):
        xs = [x0 + c * (bw + gap) for c in range(3)]; xs = xs if r % 2 == 0 else xs[::-1]
        for c in range(3):
            i = 3 * r + c; hd, dt = boxes[i]; h = (hs or {}).get(i, row_h[r])
            sh.append(flow_box(slide, xs[c], y + (row_h[r] - h) / 2, bw, h, hd, dt, name=f"Step {i+1}"))
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
bx = serpentine(s, METH, X0, 1.45, BW, GAPX, RH, VG, hs={1: 2.05, 2: 2.05, 6: 2.0, 7: 2.0})
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

# ================================================================ 2 introduction (shortened)
s = S[1]; tbs = {sh.name: sh for sh in s.shapes if sh.shape_type == 17}
set_bullets(tbs["TextBox 11"], ["Autism (ASD) is linked to differences in brain connectivity seen in resting-state fMRI.",
    "Brain networks change over time, so one static network per scan can miss short-lived changes.",
    "Labelled ASD scans are few, so we learn what is normal from control scans and flag what deviates.",
    "We also create synthetic anomalies to see if they help when only a few ASD scans exist."], 30)
set_runs(tbs["TextBox 17"].text_frame, "Can unusual changes in small brain subgraphs be found by learning only normal dynamics, and can synthetic anomalies help with few ASD examples?")
for k, (big, small) in enumerate([("871", "ABIDE I scans (403 ASD, 468 control)"), ("20", "scanning sites"), ("360", "brain regions per scan")]):
    x = 0.82 + k * 3.95; add_card(s, x, 7.3, 3.7, 2.7, name="Dataset card")
    add_tb(s, x + 0.1, 7.45, 3.5, 1.2, [big], 54, NAVY, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, name="Dataset value")
    add_tb(s, x + 0.15, 8.7, 3.4, 1.2, [small], 20, align=PP_ALIGN.CENTER, name="Dataset text")
add_tb(s, 0.82, 6.65, 11.4, 0.6, ["Dataset used"], 24, NAVY, bold=True, name="Dataset heading")

# ================================================================ 6 major contributions (what is unique in this work)
s = S[5]; strip_content(s); set_title(s, "MAJOR CONTRIBUTIONS")
Q = [(0.8, 1.5, "1  Two-level anomaly detection", "Subgraph-step scores localise where and when; a patient-level threshold, calibrated on the patient score itself, decides who is flagged. A step-level cut-off alone flagged all 131 test patients."),
     (10.1, 1.5, "2  Guided sub-structure anomaly generation", "Only existing connection weights are edited, choosing the connections the model is most sensitive to. 45.8% are detected vs 4.8% for random edits."),
     (0.8, 6.1, "3  Control-only temporal subgraph model", "Fixed-membership subgraphs are followed over time and a GIN + GRU predicts the next window from control scans only: no labelled anomalies are needed."),
     (10.1, 6.1, "4  Leakage-safe pipeline and demo application", "Patient-level splits, thresholds from validation controls only, and a web app that turns a raw time series into a flag and a severity band.")]
for x, y, ttl, txt in Q:
    add_card(s, x, y, 9.1, 4.4, name="Contribution card")
    add_tb(s, x + 0.25, y + 0.12, 8.6, 0.55, [ttl], 25, NAVY, bold=True, name="Contribution title")
    add_tb(s, x + 0.25, y + 0.72, 8.6, 1.5, [txt], 19, name="Contribution text")
def small_chart(slide, x, y, w, h, cats, vals, title, colors, ymax, fmt="0", name="Proof chart", lab=16):
    cd = CategoryChartData(); cd.categories = cats; cd.add_series(title, vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd); gf.name = name; ch = gf.chart
    ch.has_title = True; ch.chart_title.text_frame.text = title; fmt_run(ch.chart_title.text_frame.paragraphs[0].runs[0], 15, True, NAVY); ch.has_legend = False
    for ax in (ch.value_axis, ch.category_axis): ax.tick_labels.font.size = Pt(13); ax.tick_labels.font.name = FONT
    ch.value_axis.minimum_scale = 0; ch.value_axis.maximum_scale = ymax; ch.value_axis.major_gridlines.format.line.color.rgb = rgb("D9E3F0")
    pl = ch.plots[0]; pl.gap_width = 55; pl.has_data_labels = True; pl.data_labels.number_format = fmt; pl.data_labels.number_format_is_linked = False; pl.data_labels.font.size = Pt(lab); pl.data_labels.font.bold = True; pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    for i, col in enumerate(colors): pt = pl.series[0].points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(col)
# proof 1: patients flagged, step-level only vs two-level
small_chart(s, 0.95, 3.45, 8.8, 2.55, ["Step-level cut-off only", "Two-level (patient-level)"], [131, 4], "Test patients flagged (of 131)", ["B4551A", GREEN], 160, "0", "Flagged patients chart")
# proof 2: detection rates
small_chart(s, 10.25, 3.45, 8.8, 2.55, ["Real ASD steps", "Random edits", "Guided edits"], [round(100 * D['detect']["real_asd_test"], 1), round(100 * D['detect']["random/high"], 1), round(100 * D['detect']["structure_aware/high"], 1)], "Steps detected (%)", ["B5651D", "8B97A8", "6A3D9A"], 60, "0.0", "Detection rate chart")
small_chart(s, 10.25, 3.45, 8.8, 2.55, ["Real ASD steps", "Random edits", "Guided edits"], [round(100 * D["detect"]["real_asd_test"], 1), round(100 * D["detect"]["random/high"], 1), round(100 * D["detect"]["structure_aware/high"], 1)], "Steps detected (%)", ["B5651D", "8B97A8", "6A3D9A"], 60, "0.0", "Detection rate chart")
# proof 3: control subgraph + score curve
mini_graph(s, D["ctl"], 0.95, 7.65, 3.6, 2.35, "Tracked subgraph"); add_tb(s, 0.95, 9.98, 3.6, 0.35, ["tracked subgraph"], 14, GREY, align=PP_ALIGN.CENTER, name="Caption")
sc = D["scan"]; n_ = len(sc["top_seed_scores"]); cd = CategoryChartData(); cd.categories = [str(i + 1) for i in range(n_)]
cd.add_series("anomaly score", [round(v, 2) for v in sc["top_seed_scores"]]); cd.add_series("threshold", [round(sc["tau_transition"], 2)] * n_)
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(4.7), Inches(7.55), Inches(5.1), Inches(2.75), cd); gf.name = "Anomaly score chart"; ch = gf.chart
ch.has_title = True; ch.chart_title.text_frame.text = "Next-window prediction error"; fmt_run(ch.chart_title.text_frame.paragraphs[0].runs[0], 14, True, NAVY); ch.has_legend = False
for ax in (ch.value_axis, ch.category_axis): ax.tick_labels.font.size = Pt(11); ax.tick_labels.font.name = FONT
ch.value_axis.major_gridlines.format.line.color.rgb = rgb("D9E3F0"); line_series_fmt(ch.plots[0].series[0], NAVY, 3); line_series_fmt(ch.plots[0].series[1], "B4551A", 2, True, marker=False)
# proof 4: demo result card
result_card(s, 10.45, 7.6, 3.9, 2.7)
add_tb(s, 14.6, 7.7, 4.4, 2.5, [[("Demo web app", True)], "Upload a raw ROI time series → detector flag, score and severity band.", "Runs locally; research prototype."], 18, after=8, name="Demo text")

# ================================================================ 7 overall table (averaged over 1/5/10 shots, real values)
s = S[6]; strip_content(s); set_title(s, "RESULTS: OVERALL COMPARISON")
M = ["accuracy", "precision", "recall", "f1", "auroc"]; avg = {}
for c, _ in SET:
    avg[c] = {m: (sum(fs[f"{n}-shot"][c][m][0] for n in (1, 5, 10)) / 3, sum(fs[f"{n}-shot"][c][m][1] for n in (1, 5, 10)) / 3) for m in M}
best = {m: max(avg[c][m][0] for c, _ in SET) for m in M}
rows = [["Approach", "Accuracy", "Precision", "Recall", "F1 score", "AUROC"], ["Baseline detector (no few-shot data)", f"{100*acc:.1f}%", f3(m1["precision"]), f3(m1["recall"]), f3(m1["f1"]), f3(m1["auroc"])]]; bolds, fills = {}, {}
for c, lab in SET:
    r = [lab]
    for m in M:
        mu, sd = avg[c][m]; r.append(f"{100*mu:.1f}% ± {100*sd:.1f}" if m == "accuracy" else f"{mu:.3f} ± {sd:.2f}")
        if abs(mu - best[m]) < 1e-12: bolds[(len(rows), len(r) - 1)] = True; fills[(len(rows), len(r) - 1)] = "DCEFE3"
    rows.append(r)
add_table(s, 0.8, 1.8, [6.9, 2.3, 2.3, 2.3, 2.3, 2.3], rows, row_h=0.95, size=22, bolds=bolds, fills=fills, name="Overall comparison")
add_tb(s, 0.8, 7.0, 18.4, 3.4, [[("How to read: ", True), ("average over 1, 5 and 10 ASD scans (each from five random draws), tested on 131 held-out patients; green marks the best value in each column.", False)],
    "Adding generated anomalies raises recall and F1 over few-shot data alone.", "Accuracy and AUROC stay close to chance (about 50%) for every approach."], 24, bullet=True, after=12)

# ================================================================ 8 ten-shot comparison charts + three boxes
s = S[7]; strip_content(s); set_title(s, "RESULTS: 10-SHOT COMPARISON")
cats = ["Few-shot only", "Random generation", "Sub-structure generation"]; cfg = ["few_shot_only", "few_shot_plus_random", "few_shot_plus_structure_aware"]
for k, (metric, ttl, ymax, pct) in enumerate([("accuracy", "Accuracy (%)", 100, True), ("f1", "F1 score", 0.6, False), ("auroc", "AUROC", 1.0, False)]):
    cd = CategoryChartData(); cd.categories = cats; cd.add_series(ttl, [round((100 if pct else 1) * fs["10-shot"][c][metric][0], 1 if pct else 3) for c in cfg])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.8 + 6.2 * k), Inches(1.5), Inches(6.0), Inches(5.6), cd); gf.name = f"{ttl} chart"; ch = gf.chart; style_chart(ch, ttl, legend=False)
    ch.value_axis.minimum_scale = 0; ch.value_axis.maximum_scale = ymax; ch.category_axis.tick_labels.font.size = Pt(13)
    pl = ch.plots[0]; pl.gap_width = 45; pl.has_data_labels = True; pl.data_labels.number_format = "0.0" if pct else "0.00"; pl.data_labels.number_format_is_linked = False; pl.data_labels.font.size = Pt(16); pl.data_labels.font.bold = True; pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    for i, col in enumerate([NAVY, GOLD, GREEN]): pt = pl.series[0].points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(col)
BOX = [("9.6×", "Sub-structure edits are detected 9.6× more often than random edits (45.8% vs 4.8%)."),
       ("0.31 → 0.43", "10-shot F1 rises from 0.31 with few-shot data alone to 0.43 (random) and 0.40 (sub-structure)."),
       ("≈ 0.50", "AUROC stays near chance for all three approaches: the gain comes from flagging more patients.")]
for k, (big, small) in enumerate(BOX):
    x = 0.8 + k * 6.2; add_card(s, x, 7.4, 6.0, 3.0, name="Stat card")
    add_tb(s, x + 0.2, 7.5, 5.6, 1.1, [big], 48, NAVY, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, name="Stat value")
    add_tb(s, x + 0.25, 8.65, 5.5, 1.7, [small], 21, align=PP_ALIGN.CENTER, name="Stat text")

# ================================================================ 8b detecting generated anomalies (held-out test controls)
s = S[8]; strip_content(s); set_title(s, "RESULTS: GENERATED ANOMALIES")
sb = json.load(open(f"{ROOT}/paper/results/synth_benchmark.json")); G_ = sb["structure_aware"]; R_ = sb["random"]
MET = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 score"), ("auroc", "AUROC")]
rows = [["Approach", "Accuracy", "Precision", "Recall", "F1 score", "AUROC"],
        ["Random anomaly generation + detection"] + [(f"{100*R_[m]:.1f}%" if m == "accuracy" else f3(R_[m])) for m, _ in MET],
        ["Sub-structure anomaly generation + detection"] + [(f"{100*G_[m]:.1f}%" if m == "accuracy" else f3(G_[m])) for m, _ in MET]]
fills = {(2, j): "DCEFE3" for j in range(1, 6)}; bolds = {(2, j): True for j in range(1, 6)}
add_table(s, 0.8, 1.6, [7.4, 2.2, 2.2, 2.2, 2.2, 2.2], rows, row_h=0.85, size=22, fills=fills, bolds=bolds, name="Generated-anomaly detection")
cd = CategoryChartData(); cd.categories = [l for _, l in MET]
cd.add_series("Random generation", [round(R_[m], 3) for m, _ in MET]); cd.add_series("Sub-structure generation", [round(G_[m], 3) for m, _ in MET])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.8), Inches(4.45), Inches(11.4), Inches(5.0), cd); gf.name = "Generated-anomaly metrics chart"; ch = gf.chart; style_chart(ch, "Detection of edited transitions (held-out test scans)", legsz=16)
ch.value_axis.minimum_scale = 0; ch.value_axis.maximum_scale = 1.0; pl = ch.plots[0]; pl.gap_width = 55; pl.overlap = -5; pl.has_data_labels = True; pl.data_labels.number_format = "0.00"; pl.data_labels.number_format_is_linked = False; pl.data_labels.font.size = Pt(15); pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
for ser, col in zip(pl.series, ["8B97A8", "6A3D9A"]): ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(col)
for k, (big, small) in enumerate([(f"{G_['auroc']:.2f}", "AUROC for guided edits, against " + f"{R_['auroc']:.2f} for random edits"), (f"{100*G_['precision']:.0f}%", "of transitions flagged after guided edits are truly edited"), (f"{100*G_['accuracy']:.0f}%", "accuracy for guided edits, against " + f"{100*R_['accuracy']:.0f}% (chance) for random edits")]):
    y = 4.45 + k * 1.68; add_card(s, 12.5, y, 6.7, 1.55, name="Stat card"); add_tb(s, 12.6, y + 0.1, 2.4, 1.35, [big], 40, NAVY, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, name="Stat value"); add_tb(s, 15.0, y + 0.1, 4.1, 1.35, [small], 17, anchor=MSO_ANCHOR.MIDDLE, name="Stat text")
add_tb(s, 0.8, 9.55, 18.4, 1.0, [f"Setup: 70 held-out test controls × 20 regions = {G_['n_pairs']:,} edited transitions per method (equal edit budget); threshold {sb['tau_transition']:.2f} from validation controls. This measures how detectable the generated anomalies are (the same model guides and scores the edits); it is not ASD diagnosis."], 17, GREY)

# ================================================================ 9 synthetic anomalies (detection-rate chart + the two subgraphs)
s = S[9]; strip_content(s); set_title(s, "RESULTS: EDIT DETECTABILITY")
dt = D["detect"]; keys = [("real_normal_val_control", "real control"), ("real_asd_test", "real ASD"), ("random/high", "random, high"), ("structure_aware/low", "guided, low"), ("structure_aware/medium", "guided, medium"), ("structure_aware/high", "guided, high")]
cd = CategoryChartData(); cd.categories = [l for _, l in keys]; cd.add_series("steps above threshold (%)", [round(100 * dt[k], 1) for k, _ in keys])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.8), Inches(1.55), Inches(9.4), Inches(8.2), cd); gf.name = "Detection rate chart"; ch = gf.chart; style_chart(ch, "Share of steps above the detection threshold (%)", legend=False)
pl = ch.plots[0]; pl.gap_width = 45; pl.has_data_labels = True; pl.data_labels.number_format = "0.0"; pl.data_labels.number_format_is_linked = False; pl.data_labels.font.size = Pt(17); pl.data_labels.font.bold = True; pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
for i, col in enumerate(["1B7F79", "B5651D", "8B97A8", "6A3D9A", "6A3D9A", "6A3D9A"]): pt = pl.series[0].points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(col)
ch.category_axis.tick_labels.font.size = Pt(14)
mini_graph(s, D["synth_before"], 10.6, 1.9, 4.2, 6.6, "Real control subgraph"); mini_graph(s, D["synth_after"], 15.0, 1.9, 4.2, 6.6, "Synthetic anomaly subgraph")
add_tb(s, 10.6, 8.55, 4.2, 0.45, ["real control subgraph"], 17, GREY, align=PP_ALIGN.CENTER, name="Caption"); add_tb(s, 15.0, 8.55, 4.2, 0.45, ["after guided edit (edited edges bold)"], 17, GREY, align=PP_ALIGN.CENTER, name="Caption")
si = D["synth_info"]
add_card(s, 0.8, 9.85, 18.4, 0.7, name="Takeaway"); add_tb(s, 1.0, 9.9, 18.0, 0.6, [f"Guided edits: score {si['score_before']:.2f} → {si['score_after']:.2f} on this example; 45.8% of guided high-severity steps are detected vs 4.8% for random edits."], 20, NAVY, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, name="Takeaway text")

# ================================================================ 14 discussion (existing slide, edit text)
s = S[10]; tbs = {sh.name: sh for sh in s.shapes if sh.shape_type == 17}
set_bullets(tbs["TextBox 12"], ["Stable training with no representation collapse (embedding std 1.03–1.16).", "Generator works as designed: 7,200 records, 0 rule violations, severity-monotonic.", "Sub-structure edits detected 45.8% of the time vs 4.8% for random edits (9.6×).", "Leakage-safe protocol: patient-level split, thresholds from validation controls only.", "Complete pipeline plus a demo application and 20 sample scans."], 24)
set_bullets(tbs["TextBox 18"], ["Detector is at chance on ABIDE I: AUROC 0.453, accuracy 53.4%.", "A single step-level threshold flagged all 131 test patients; fixed with a patient-level threshold.", "Few-shot F1 gains come from flagging more patients; AUROC stays about 0.50.", "Severity estimate is below a majority guess (43.3% vs 55.0%).", "Only 20 of 360 regions, one training run, no site-level test."], 24)
set_bullets(tbs["TextBox 24"], ["A diagnosis label is a weak sign of unusual connectivity dynamics; ASD is highly varied.", "Site effects and short, noisy windows may hide a real signal.", "Detectability of generated anomalies is self-referential: the same model guides and scores the edits.", "Guided and random generation behave alike in the few-shot test.", "Research prototype: no clinical claim."], 24)
set_title(s, "DISCUSSION")
for g in s.shapes:
    if g.shape_type == 6 and inch(g.height) > 7.5 and inch(g.top) > 2: g.height = Inches(6.9)

# ================================================================ 15 conclusion
s = S[11]; strip_content(s); set_title(s, "CONCLUSION"); add_card(s, 0.81, 1.7, 18.4, 7.4, name="Conclusion card")
add_tb(s, 1.3, 2.0, 17.4, 6.8, ["Built and tested an end-to-end pipeline: time-resolved brain graphs, brain-region subgraph tracking, temporal anomaly scoring, synthetic anomaly generation, few-shot adaptation and a severity output.",
    "All six objectives were completed, including a working demo web application.", "The anomaly generator works as designed: 0 rule violations, and 45.8% of sub-structure edits are detected against 4.8% for random edits.",
    "On ABIDE I the detector is at chance (AUROC 0.453, accuracy 53.4%); few-shot F1 gains come from a higher flag rate (AUROC about 0.50).", "Severity estimation is below a majority guess (43.3% vs 55.0%).",
    "Research prototype only: no clinical claim, and all results are reported as found."], 30, bullet=True, after=22)

for i, sl in enumerate(S, 1): set_number(sl, i)
prs.save(sys.argv[2]); print("saved", len(S), "slides")
