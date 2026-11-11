"""Second revision pass: positive-but-honest framing, stronger future work (all new text red)."""
import sys, copy, docx
from docx.oxml.ns import qn
from lxml import etree
src, out = sys.argv[1], sys.argv[2]
d = docx.Document(src); P = d.paragraphs
RED = "FF0000"
def red_run(p, text, like):
    r = copy.deepcopy(like._r)
    for t in r.findall(qn("w:t")): r.remove(t)
    rpr = r.find(qn("w:rPr")); old = rpr.find(qn("w:color"))
    if old is not None: rpr.remove(old)
    c = etree.SubElement(rpr, qn("w:color")); c.set(qn("w:val"), RED)
    sz = rpr.find(qn("w:sz"))
    if sz is not None: sz.addprevious(c)
    t = etree.SubElement(r, qn("w:t")); t.text = text; t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    p._p.append(r)
def rewrite(p, text, keep=0):
    rs = p.runs[keep:]; like_c = copy.deepcopy(rs[0]._r)
    for r in rs: r._r.getparent().remove(r._r)
    class L: _r = like_c
    red_run(p, text, L)
def check(p, start): assert p.text.startswith(start), (start, p.text[:40])

check(P[2], "Abstract:")
rewrite(P[2], "Resting-state fMRI turns every scan into a brain network, but very few labelled scans exist. This study introduces a model that tracks small brain subgraphs over time and learns normal behavior from control scans only; a region that behaves unusually receives a high anomaly score. Real control scans then serve to generate structure-aware anomalies by gradient-guided edge-weight edits. Guided edits get detected far better than random edits of the same size: 45.8% against 4.8% of steps pass the threshold, and on 70 held-out control scans the AUROC is 0.906 against 0.491. A model-wise comparison of four encoder and predictor pairs shows that GIN + GRU detects generated anomalies best. In few-shot adaptation, adding generated anomalies to 1, 5 or 10 real ASD scans lifts one-shot recall and F1 from almost zero to about 0.40, while AUROC stays near 0.5 because more patients get flagged without a better ranking. Direct ASD detection on ABIDE I stays at chance level (AUROC 0.453) and severity bands do not beat a majority-class guess; both stay as limits of the present data and features.", keep=1)

check(P[8], "The contribution")
rewrite(P[8], "The contribution consists of five parts: (1) a temporal model that follows small brain subgraphs and learns normal behavior from controls only; (2) a structure-aware generator that edits real connections, with its anomalies detected far better than random edits (AUROC 0.906 against 0.491); (3) a model-wise comparison in which GIN + GRU is the strongest of four encoder and predictor pairs; (4) few-shot adaptation in which generated anomalies lift one-shot F1 from 0.006 to about 0.40; (5) a full mathematical description and a demo application with 20 sample scans. This research makes no clinical claims.")

check(P[90], "Table III")
rewrite(P[90], "Table III summarizes the few-shot experiments and Fig. 4 illustrates them. With one real ASD scan the classifier flags almost nobody and reaches an F1 of only 0.006. Adding 30 generated anomalies lifts the one-shot F1 to 0.39-0.40, and for 5 and 10 shots the F1 rises from 0.15 and 0.31 to about 0.40 and 0.43. Generated anomalies therefore make few-shot adaptation usable from a single real scan, where real scans alone fail. The AUROC stays near 0.50 in every setup, and the F1 values match the chance F1 for the same flag rate (35-39% of patients), so the gain comes from higher recall and not from a better ranking. Guided and random anomalies give similar results here, which shows that the benefit comes from the generated positives and not from their structure.")

check(P[108], "Despite")
rewrite(P[108], "The pipeline trains stably, the generator produces anomalies that the model detects far better than random edits (AUROC 0.906 against 0.491), and GIN + GRU is the strongest of the four encoder and predictor pairs tested. Direct ASD detection on ABIDE I stays at chance level (AUROC 0.453). Some possible reasons include: 1) low signal; 2) huge variability of ASD; 3) site effects [50]; 4) noise and other effects [12]; and 5) the use of 20 features out of the total 360. The limitations include: training was run only once; two design decisions (the patient threshold and the selection of 20 features) followed the results and not prior planning; no cross-site experiments were done; baseline comparisons were not done for the whole graph, and sliding window tests were not performed; severity linking presumes a specific order. The guided-edit result comes from a scorer and a generator sharing one model, so it shows that generated anomalies stay visible to the model and does not guarantee effectiveness in ASD detection.")

check(P[110], "The pipeline")
rewrite(P[110], "The pipeline traces the behavior of small brain subgraphs over time, captures significant changes, produces structure-aware anomalies and adapts itself using ASD scans. Guided sub-structure modifications get detected far better than random changes of the same size (AUROC 0.906 against 0.491 on held-out control scans), GIN + GRU gives the best detection among the four models compared, and generated anomalies lift one-shot F1 from 0.006 to about 0.40. Direct ASD detection on ABIDE I stays at chance level (AUROC 0.453), which marks the main limit of the present study.")
fut = copy.deepcopy(P[110]._p); P[110]._p.addnext(fut)
from docx.text.paragraph import Paragraph
fp = Paragraph(fut, P[110]._parent); rewrite(fp, "Future work will address this limit in six steps: (1) leave-one-site-out evaluation across the 20 ABIDE sites with site harmonization, to separate site effects [50] from disease signal; (2) scoring generated anomalies with a second model that did not guide the edits, which removes the shared-model caveat; (3) full-graph baselines and sweeps of window length, stride and threshold percentile over multiple training runs, with confidence intervals; (4) data-driven selection of the monitored regions in place of the fixed 20 seeds; (5) ordinal regression on continuous ADOS-2 severity scores for a larger labelled cohort; (6) validation on independent cohorts together with clinicians before any clinical claim.")

# Table II label: "ours" -> proposed
for t in d.tables:
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    if "(ours)" in r.text:
                        r.text = r.text.replace("(ours)", "(proposed)")
                        rp = r._r.get_or_add_rPr(); col = rp.find(qn("w:color"))
                        if col is None:
                            col = etree.SubElement(rp, qn("w:color"))
                        col.set(qn("w:val"), RED)
d.save(out)

# ---- Fig. 1 fallback picture for viewers without Word drawing-canvas support (Google Docs)
d = docx.Document(out)
from docx.shared import Emu
FB = sys.argv[3]
for r in d.element.body.iter(qn("w:r")):
    dr = r.find(qn("w:drawing"))
    if dr is not None and b"wordprocessingCanvas" in etree.tostring(dr):
        ext = dr.find(".//" + qn("wp:extent")); cx, cy = int(ext.get("cx")), int(ext.get("cy"))
        tmp = d.add_paragraph().add_run(); tmp.add_picture(FB, width=Emu(cx), height=Emu(cy))
        pic = tmp._r.find(qn("w:drawing")); tmp._r.getparent().getparent().remove(tmp._r.getparent())
        MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
        ac = etree.Element("{%s}AlternateContent" % MC, nsmap={"mc": MC})
        ch = etree.SubElement(ac, "{%s}Choice" % MC, nsmap={"wpc": "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"}); ch.set("Requires", "wpc")
        fb = etree.SubElement(ac, "{%s}Fallback" % MC)
        dr.addprevious(ac); ch.append(dr); fb.append(pic)
        break
else: raise SystemExit("canvas not found")
d.save(out)
