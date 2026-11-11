"""Adds the measured generated-anomaly benchmark (paper/results/synth_benchmark.json) to the user's humanized paper.
Every added word is red (FF0000); no existing text is altered. usage: python3 add_synth_to_paper.py in.docx out.docx"""
import copy, json, sys
import docx
from docx.oxml.ns import qn
from docx.oxml import OxmlElement, parse_xml
RED = "FF0000"
d = docx.Document(sys.argv[1]); P = d.paragraphs
sb = json.load(open("results/synth_benchmark.json")); G, R = sb["structure_aware"], sb["random"]
def find(prefix): return next(p for p in d.paragraphs if p.text.strip().startswith(prefix))
def red_rpr(src_run):
    rpr = copy.deepcopy(src_run._r.find(qn("w:rPr"))) if src_run._r.find(qn("w:rPr")) is not None else OxmlElement("w:rPr")
    for c in rpr.findall(qn("w:color")): rpr.remove(c)
    col = OxmlElement("w:color"); col.set(qn("w:val"), RED); rpr.insert(0 if rpr.find(qn("w:rFonts")) is None else 1, col); return rpr
def clone(src, text):
    """new paragraph with src's paragraph + first-run formatting, text in red"""
    el = copy.deepcopy(src._p)
    for r in el.findall(qn("w:r")): el.remove(r)
    r = OxmlElement("w:r"); r.append(red_rpr([x for x in src.runs if x.text.strip()][0] if any(x.text.strip() for x in src.runs) else src.runs[0]))
    t = OxmlElement("w:t"); t.text = text; t.set(qn("xml:space"), "preserve"); r.append(t); el.append(r); return el
def append_red(par, text):
    last = [x for x in par.runs if x.text.strip()][-1]; r = OxmlElement("w:r"); r.append(red_rpr(last))
    t = OxmlElement("w:t"); t.text = " " + text; t.set(qn("xml:space"), "preserve"); r.append(t); par._p.append(r)

heading_src = find("I. Detection and Severity"); body_src = find("Individuals are considered marked")
cap_src = find("TABLE IV."); demo_para = find("We developed a web app")
pct = lambda v: f"{100*v:.1f}%"

# Abstract, Discussion, Conclusion: one added red sentence each
append_red(find("Abstract:"), f"In an extra test on held-out control scans, sub-structure edits were detected far better than random edits (AUROC {G['auroc']:.3f} against {R['auroc']:.3f}).")
append_red(find("Although the training of the pipeline"), f"In the extra test on held-out control scans, guided sub-structure edits were detected much better than random edits (AUROC {G['auroc']:.3f} against {R['auroc']:.3f}); this still reflects that the generator and the scorer share one model, and it does not mean better ASD detection.")
append_red(find("We constructed and evaluated a pipeline"), f"Guided sub-structure edits are detected much more reliably than random edits of the same size (AUROC {G['auroc']:.3f} against {R['auroc']:.3f} on held-out control scans).")

# Method: new subsection J after subsection I
mh = clone(heading_src, "J. Benchmark on generated anomalies")
mb = clone(body_src, f"To measure how detectable the generated anomalies are, we ran an extra test on the 70 held-out test control scans. For every scan and each of the 20 tracked regions we picked one window transition and made an edited version at severity 0.9, once with guided edits and once with random edits of the same size. The unedited score of the transition is the negative example and the score after the edit is the positive example ({G['n_pairs']:,} pairs per method). A transition counts as detected when its score is above the step threshold of {sb['tau_transition']:.2f} taken from the validation controls. We report accuracy, precision, recall, F1 score and AUROC. These scans were not used for training, calibration, support sets or generation.")
body_src._p.addnext(mh); mh.addnext(mb)

# Results: new subsection G after the demo application paragraph
rh = clone(heading_src, "G. Detection of generated anomalies")
rb = clone(body_src, f"Table V shows the result. Guided sub-structure edits are detected much better than random edits: accuracy {pct(G['accuracy'])} against {pct(R['accuracy'])}, precision {G['precision']:.3f} against {R['precision']:.3f}, recall {G['recall']:.3f} against {R['recall']:.3f}, F1 score {G['f1']:.3f} against {R['f1']:.3f} and AUROC {G['auroc']:.3f} against {R['auroc']:.3f}. Random edits of the same size are not noticed, while guided edits raise the mean transition score from {G['mean_before']:.2f} to {G['mean_after']:.2f}. This shows that choosing which connections to edit, using the model's sensitivity, makes the anomalies far easier to detect. Because the same model guides and scores the edits, this test shows how detectable the generated anomalies are; it is not a test of ASD diagnosis.")
rc = clone(cap_src, f"TABLE V. Detection of generated anomalies on held-out test controls ({G['n_pairs']:,} edited transitions per method).")
W = [1060, 840, 830, 700, 560, 760]
def cell(txt, w, hdr=False, shade=None):
    sh = f'<w:shd w:val="clear" w:color="auto" w:fill="{shade}"/>' if shade else ""
    b = "<w:b/>" if hdr else ""
    return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{sh}<w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>'
            f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>{b}<w:color w:val="{RED}"/><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr><w:t xml:space="preserve">{txt}</w:t></w:r></w:p></w:tc>')
def row(vals, hdr=False, shade=None, bold=False): return "<w:tr><w:trPr><w:cantSplit/></w:trPr>" + "".join(cell(v, w, hdr or bold, shade) for v, w in zip(vals, W)) + "</w:tr>"
hdr = ["Approach", "Accuracy", "Precision", "Recall", "F1", "AUROC"]
rows = [hdr, ["Random", pct(R["accuracy"]), f"{R['precision']:.3f}", f"{R['recall']:.3f}", f"{R['f1']:.3f}", f"{R['auroc']:.3f}"],
        ["Sub-structure", pct(G["accuracy"]), f"{G['precision']:.3f}", f"{G['recall']:.3f}", f"{G['f1']:.3f}", f"{G['auroc']:.3f}"]]
bd = "".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="000000"/>' for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
tbl = parse_xml(f'<w:tbl xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:tblPr><w:tblW w:w="{sum(W)}" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>{bd}</w:tblBorders><w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="50" w:type="dxa"/><w:right w:w="50" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>' + "".join(f'<w:gridCol w:w="{w}"/>' for w in W) + "</w:tblGrid>"
                + row(rows[0], hdr=True) + row(rows[1]) + row(rows[2], shade="E2F0D9", bold=True) + "</w:tbl>")
spacer = clone(body_src, " ")
demo_para._p.addnext(rh); rh.addnext(rb); rb.addnext(rc); rc.addnext(tbl); tbl.addnext(spacer)
d.save(sys.argv[2]); print("saved")
