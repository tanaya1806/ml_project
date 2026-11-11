"""Revises the user's final paper per the professor's comments. EVERY change is written in red (FF0000); unchanged text stays as is.
usage: python3 revise_paper.py in.docx out.docx   (run from paper/revise)"""
import copy, json, re, sys, os
import docx
from lxml import etree
from docx.oxml.ns import qn
import omml as O
from omml import limlow, R as mR, U as mU, sub, sup, subsup, frac, rad, nary, nary_u, hat, bar, paren, norm, cat, eq_par
from canvas import canvas_xml

ROOT = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP_GRAPH/paper"
WNS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
RED = "FF0000"
d = docx.Document(sys.argv[1]); body = d.element.body
R_ = json.load(open(f"{ROOT}/docx_build/refs.json"))
R_["hochreiter1997"] = [["S. Hochreiter and J. Schmidhuber, “Long short-term memory,” ", 0], ["Neural Comput.", 1], [", vol. 9, no. 8, pp. 1735–1780, 1997, doi: 10.1162/neco.1997.9.8.1735.", 0]]
R_["morris2019"] = [["C. Morris ", 0], ["et al.", 1], [", “Weisfeiler and Leman go neural: Higher-order graph neural networks,” in ", 0], ["Proc. AAAI Conf. Artif. Intell.", 1], [", vol. 33, no. 1, 2019, pp. 4602–4609, doi: 10.1609/aaai.v33i01.33014602.", 0]]
OLD = json.load(open("/private/tmp/claude-501/nr/oldmap.json")); OLD = {int(k): v for k, v in OLD.items()}
paras = list(d.paragraphs)
P = lambda i: paras[i]
def ptext(p): return "".join(("\t" if c.tag == qn("w:tab") else c.text or "") for r in p._p.iter() if r.tag in (qn("w:t"), qn("w:tab")) for c in [r])

# ---------------------------------------------------------------- run helpers
def mkrpr(base, red):
    rp = copy.deepcopy(base) if base is not None else etree.SubElement(etree.Element(qn("w:r")), qn("w:rPr"))
    for c in rp.findall(qn("w:color")): rp.remove(c)
    if red:
        col = etree.Element(qn("w:color")); col.set(qn("w:val"), RED); sz = rp.find(qn("w:sz"))
        (sz.addprevious(col) if sz is not None else rp.append(col))
    return rp
def mkrun(text, base, red):
    r = etree.Element(qn("w:r")); r.append(mkrpr(base, red))
    parts = text.split("\t")
    for k, part in enumerate(parts):
        if k: etree.SubElement(r, qn("w:tab"))
        if part: t = etree.SubElement(r, qn("w:t")); t.text = part; t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return r
def first_text_rpr(p, skip=0):
    runs = [r for r in p._p.findall(qn("w:r")) if "".join(t.text or "" for t in r.findall(qn("w:t"))).strip()]
    return (runs[skip] if len(runs) > skip else runs[0]).find(qn("w:rPr"))
def seg_text_after(p, skip):
    runs = p._p.findall(qn("w:r")); return "".join("".join(t.text or "" for t in r.findall(qn("w:t"))) for r in runs[skip:])
def clear_runs(p, keep=0):
    runs = p._p.findall(qn("w:r"))
    for r in runs[keep:]: p._p.remove(r)
    for c in list(p._p):
        if c.tag not in (qn("w:pPr"), qn("w:r")): p._p.remove(c)

# ---------------------------------------------------------------- paragraph spec registry (segments are (text, red))
SPEC = {}      # element -> dict(segs, base, keep)
def spec_for(p, keep=0):
    el = p._p
    if el not in SPEC:
        txt = seg_text_after(p, keep); SPEC[el] = {"segs": [(txt, False)], "base": first_text_rpr(p, 0 if keep == 0 else keep - 0) if keep == 0 else p._p.findall(qn("w:r"))[keep].find(qn("w:rPr")), "keep": keep, "changed": False, "p": p}
    return SPEC[el]
def repl(p, pairs, keep=0):
    s = spec_for(p, keep)
    for old, new in pairs:
        done = False
        for k, (t, red) in enumerate(s["segs"]):
            if red: continue
            i = t.find(old)
            if i >= 0:
                s["segs"][k:k + 1] = [x for x in [(t[:i], False), (new, True), (t[i + len(old):], False)] if x[0] != ""]; done = True; s["changed"] = True; break
        assert done, f"sentence not found: {old[:60]}"
def append_red(p, text, keep=0):
    s = spec_for(p, keep); s["segs"].append((text, True)); s["changed"] = True
def set_all_red(p, text, keep=0):
    s = spec_for(p, keep); s["segs"] = [(text, True)]; s["changed"] = True

# ---------------------------------------------------------------- new paragraphs from templates
def new_par_from(tpl, segs, base=None, keep_pPr=True):
    el = copy.deepcopy(tpl._p)
    for c in list(el):
        if c.tag != qn("w:pPr"): el.remove(c)
    from docx.text.paragraph import Paragraph
    np_ = Paragraph(el, tpl._parent); SPEC[el] = {"segs": segs, "base": base if base is not None else first_text_rpr(tpl), "keep": 0, "changed": True, "p": np_}; return np_
def after(anchor_el, new_el): anchor_el.addnext(new_el); return new_el

TPL_BODY, TPL_H2, TPL_CAP_FIG, TPL_CAP_TAB = P(18), P(17), P(25), P(42)
def body_par(text): return new_par_from(TPL_BODY, [(text, True)])
def h2_par(text): return new_par_from(TPL_H2, [(text, True)])
def desc_par(text):
    np_ = new_par_from(TPL_BODY, [(text, True)]); return np_

# ================================================================ 1. TITLE
tp = P(0); tr = tp._p.findall(qn("w:r"))[0]; tr.addnext(mkrun(" for Autism Spectrum Disorder", tr.find(qn("w:rPr")), True))

# ================================================================ 2. REMOVE "are / we / our / us"
repl(P(2), [
 ("Moreover, there are very few tagged scans available.", "Moreover, very few tagged scans exist."),
 ("This paper shows how artificial anomalies are acquired with the help of actual controlled scans and studies whether those changes along with some original scans will increase the quality of classification.", "This paper shows how artificial anomalies can be acquired with the help of actual controlled scans and studies whether those changes, along with some original scans, will increase the quality of classification."),
 ("The artificial anomalies are properly detected (45.8% vs. 4.8% for random modifications) as the model was engaged in their creation.", "The artificial anomalies get properly detected (45.8% vs. 4.8% for random modifications) as the model was engaged in their creation."),
 ("A slight modification made it possible to increase the F1 result without changing AUROC measures since the modifications are frequent.", "A slight modification made it possible to increase the F1 result without changing AUROC measures since the modifications occur frequently."),
 ("The research results are open and the developed chain is now on the market.", "The research results stay open and the developed chain is now on the market."),
 ("In an additional test with control scans it was shown that substructure changes are detected better than random modifications (AUROC 0.906 against 0.491).", "In an additional test with control scans it was shown that substructure changes get detected better than random modifications (AUROC 0.906 against 0.491)."),
], keep=1)
repl(P(7), [("Our analysis includes three main study questions.", "The analysis includes three main study questions.")])
repl(P(8), [
 ("Our contribution consists of four parts: (1) a model which is possible to follow small brain graphs in order to find how they change; (2) a generator that uses the genuine anomalies to create synthetic anomalies from real connections and makes the benchmarks for random networks; (3) experiments that do not reveal the answers as they are honest; (4) demo application with 20 sample scans.", "The contribution consists of four parts: (1) a model that follows small brain graphs in order to find how they change; (2) a generator that uses genuine connections to create synthetic anomalies and builds benchmarks against random networks; (3) honest experiments that do not reveal the answers; (4) a demo application with 20 sample scans."),
 ("There are no clinical claims in this research.", "This research makes no clinical claims.")])
repl(P(11), [("Literature reviews indicate there are many approaches to researching the dynamic connectome [35].", "Literature reviews indicate many approaches to researching the dynamic connectome [35].")])
repl(P(12), [
 ("In graph anomaly detection, we are supposed to search for strange nodes, edges, subgraphs, or graphs altogether, which is reviewed in [38].", "Graph anomaly detection searches for strange nodes, edges, subgraphs, or graphs altogether, as reviewed in [38]."),
 ("There are graph-level implementations of the above-stated methods as well [40].", "Graph-level implementations of the above-stated methods exist as well [40]."),
 ("There are some models for brain data that trained solely on the scans of the controls and then flagged different edges and hyperedges using synthetic anomalies for testing [15],[16].", "Some models for brain data trained solely on the scans of the controls and then flagged different edges and hyperedges using synthetic anomalies for testing [15],[16]."),
 ("The method we use here is based on this idea of using only controls because it measures the error of forecasting the subsequent condition of the traced subgraph.", "The method used here builds on this idea of using only controls because it measures the error of forecasting the subsequent condition of the traced subgraph.")])
repl(P(13), [("In terms of brain studies, few-shot opportunities are mostly used for supervised learning, e.g. meta-learning based on several brain connectomes [47], graph foundation models adjusted by uses of prompts [48] or few-shot categorization of individuals suffering from autism based on structural MRI [49].", "In terms of brain studies, few-shot opportunities mostly serve supervised learning, e.g. meta-learning based on several brain connectomes [47], graph foundation models adjusted by uses of prompts [48] or few-shot categorization of individuals suffering from autism based on structural MRI [49].")])
repl(P(14), [("We apply this concept in order to create anomalies, instead of using it for the model attack: the modifications are confined by the existing connection weights and validated using some random baseline.", "This concept applies here to create anomalies, instead of attacking a model: the modifications stay confined by the existing connection weights and get validated using a random baseline.")])
repl(P(15), [("We do not know about any earlier publications introducing the combination of", "No earlier publication known to the authors introduces the combination of")])
repl(P(18), [
 ("We applied ABIDE I [20]:", "ABIDE I [20] was applied:"),
 ("We perform the separation based on individuals: 609 people go to training (328 controls and 281 ASD), 131 – to validation, and 131 – to testing, maintaining the ASD/control ratio.", "The separation is based on individuals: 609 people go to training (328 controls and 281 ASD), 131 to validation, and 131 to testing, maintaining the ASD/control ratio.")])
repl(P(20), [
 ("We divide each scan into 30-time-point windows with sliding of 10 points each time.", "Each scan is divided into 30-time-point windows with sliding of 10 points each time."),
 ("In each window we calculate correlations between each pair of regions and keep the strongest 20% of connections and their sign.", "In each window, correlations between each pair of regions get calculated, and the strongest 20% of connections stay, together with their sign.")])
repl(P(22), [
 ("We select 20 regions correctly distributed across the brain.", "Twenty regions, distributed across the brain, get selected."),
 ("For each of them, we take the region and its immediate neighbours in the initial window and keep following this constant set of regions in all subsequent windows.", "For each of them, the region and its immediate neighbours in the initial window form a constant set that stays followed in all subsequent windows."),
 ("Only connection and weights of the connections are altered.", "Only connections and their weights change.")])
repl(P(23), [("It has nine steps.", "It has ten steps and ends with the severity level analysis.")])
repl(P(27), [
 ("Both of them are trained simultaneously only on control scans.", "Both get trained simultaneously, only on control scans."),
 ("We use the PyTorch [27] and the PyTorch Geometric [28].", "The implementation uses PyTorch [27] and PyTorch Geometric [28], and Adam [[kingma2015]] performs the optimization.")])
repl(P(31), [
 ("We must utilize one threshold value for the steps likewise one for the individual.", "One threshold value is needed for the steps, and one for the individual."),
 ("The reason being that an individual performs around 280 steps and 5% of the test subjects are able to go beyond the step threshold even by mistake.", "The reason is that an individual performs around 280 steps, and 5% of the steps can exceed the step threshold even by mistake."),
 ("It was only after we conducted all the tests that we realised this so the analysis of precision, recall and F1 cannot be said to be completely blind; AUROC remains unaffected.", "This was realised only after all the tests had been conducted, so the analysis of precision, recall and F1 cannot be said to be completely blind; AUROC remains unaffected.")])
repl(P(33), [
 ("In case of the actual control subgraph we make adjustments to the existing weights.", "In the case of the actual control subgraph, adjustments apply to the existing weights."),
 ("We assess how significant change each weight would bring which allows us to keep those weights that seem most important (up to 15% of the weights).", "The influence of each weight on the error is assessed, which keeps the weights that seem most important (up to 15% of the weights)."),
 ("Afterwards we change the weights in 5 small steps in the direction of increasing the error [18], [19].", "Afterwards, the weights change in 5 small steps in the direction of increasing the error [18], [19]."),
 ("The weights remain at the range of −1 to 1, no new connections are established or no connections removed.", "The weights remain in the range of −1 to 1, and no new connections get established or removed.")])
repl(P(35), [
 ("As such, we proceed with training logistic regression with either 1, 5 or 10 of the actual ASD scans counted as positive examples, 30 synthetic anomalies counted as additional positives, and the 328 zeros as negatives.", "As such, logistic regression gets trained with either 1, 5 or 10 of the actual ASD scans as positive examples, 30 synthetic anomalies as additional positives, and the 328 control scans as negatives."),
 ("This procedure is done for at least five different samples of ASD scans, while comparing the results of three setups; only real ASD scans, plus synthetic anomalies guided by our real scans, and including random synthetic anomalies.", "This procedure repeats for five different samples of ASD scans, comparing three setups: only real ASD scans, real ASD scans plus guided synthetic anomalies, and real ASD scans plus random synthetic anomalies.")])
repl(P(37), [
 ("Individuals are considered marked if their score exceeds our threshold.", "An individual counts as marked if the score exceeds the threshold."),
 ("For all marked individuals, we provide severity estimates based on the ADOS-2 severity score (1-10); low (≤5), moderate (6-7) or high (≥8).", "For all marked individuals, severity estimates come from the ADOS-2 severity score [29] (1-10): low (≤5), moderate (6-7) or high (≥8)."),
 ("This categorization is done by us and is not done according to DSM-5.", "This categorization is defined by the authors and does not follow DSM-5."),
 ("The scores are available for 193 out of 403 individuals diagnosed with ASD.", "Scores exist for 193 out of 403 individuals diagnosed with ASD.")])
repl(P(39), [
 ("To determine the detectability of the generated anomalies, we conducted an additional experiment involving the 70 held-out test control scans.", "To determine the detectability of the generated anomalies, an additional experiment involved the 70 held-out test control scans."),
 ("For each scan and each of the 20 regions of interest, we selected a window transition and created an edited version of it with a severity of 0.9, one with guided edits and the other with random edits of the same magnitude.", "For each scan and each of the 20 regions of interest, a window transition was selected and an edited version of it was created with a severity of 0.9, one with guided edits and the other with random edits of the same magnitude."),
 ("We give accuracy, precision, recall, F1 score and AUROC as the results of this experiment.", "Accuracy, precision, recall, F1 score and AUROC serve as the results of this experiment.")])
repl(P(41), [
 ("AUROC and AUPRC which are independent from any threshold are reported, along with precision, recall and F1 for the selected threshold.", "AUROC and AUPRC, which stay independent from any threshold, get reported along with precision, recall and F1 for the selected threshold."),
 ("Results in few-shot settings are the mean and standard deviation of results from five experiments that share a single test set.", "Results in few-shot settings show the mean and standard deviation of results from five experiments that share a single test set.")])
repl(P(49), [
 ("and our GRU error.", "and the GRU error."),
 ("The encoder was trained with the GRU, so the baselines are not completely independent.", "The encoder was trained with the GRU, so the baselines cannot count as completely independent.")])
repl(P(56), [
 ("The results of the few-shot experiments are summarized in Table III and illustrated in Fig. 4.", "Table III summarizes the results of the few-shot experiments and Fig. 4 illustrates them."),
 ("However, when 30 synthetic anomalies are added, the one-shot F1 score", "However, when 30 synthetic anomalies get added, the one-shot F1 score")])
repl(P(64), [("We developed a web app that operates on the user's computer.", "A web app that operates on the user's computer was developed.")])
repl(P(66), [
 ("Results are represented in Table V.", "Table V presents the results."),
 ("Random edits produced in similar volume are not the ones to identify thanks to guided editing, which enabled the mean transition score to be increased from 2.94 to 7.62.", "Random edits of similar volume stay largely undetected, while guided editing raised the mean transition score from 2.94 to 7.62.")])
repl(P(69), [
 ("There are some possible reasons behind that: 1) low signal; 2) huge variability of ASD; 3) site effects [8]; 4) number of noise and effects [10]; and 5) the application of 20 features out of the total 360 ones.", "Some possible reasons behind that include: 1) low signal; 2) huge variability of ASD; 3) site effects [8]; 4) noise and other effects [10]; and 5) the application of 20 features out of the total 360 ones."),
 ("baseline comparisons were not done for the whole graph and other encoders and sliding windows tests were not performed;", "baseline comparisons were not done for the whole graph, and sliding window tests were not performed;")])
repl(P(71), [("analyzing the performance of full-graph and other-encoder baselines", "analyzing the performance of full-graph baselines")])

# ================================================================ 3. FIGURE 1: editable canvas + short caption + description
fig1 = P(24); r0 = fig1._p.findall(qn("w:r"))[0]
for dr in r0.findall(qn("w:drawing")): r0.remove(dr)
r0.append(etree.fromstring(canvas_xml(900)))
cap1 = P(25); set_all_red(cap1, "Fig. 1. System architecture diagram.")
dp = desc_par("The diagram follows the pipeline from data preparation and sliding-window graph construction through subgraph tracking, GIN and GRU modelling, anomaly scoring and threshold calibration to anomaly generation, few-shot adaptation, anomaly detection and the final severity level analysis. Each box names the step and its key setting.")
after(cap1._p, dp._p)
# ---- Fig. 2
cap2 = P(47); set_all_red(cap2, "Fig. 2. Detector score distribution and ROC curve.")
dp2 = desc_par("Panel (a) shows the patient scores of validation controls, test controls and test ASD patients against the detection threshold of 19.81; the three distributions overlap almost completely. Panel (b) shows the ROC curve with a 95% bootstrap band, which stays close to the chance line (AUROC 0.453). Together, the panels show that the detector cannot separate ASD from control scans.")
after(cap2._p, dp2._p)
# ---- Fig. 3
cap3 = P(54); set_all_red(cap3, "Fig. 3. Effect of guided edits on anomaly scores.")
dp3 = desc_par("Panel (a) shows how the anomaly score rises with edit severity for guided edits but stays flat for random edits. Panel (b) gives the share of steps above the step threshold: 45.8% for guided high-severity edits against 4.8% for random edits and about 5% for real scans. Panel (c) ranks the connections by influence; the 74 chosen out of 548 mark the connections to which the model reacts most.")
after(cap3._p, dp3._p)
# ---- Fig. 4 (label bold + text)
cap4 = P(58); runs = cap4._p.findall(qn("w:r")); runs[1].find(qn("w:t")).text = " Few-shot results for 1, 5 and 10 ASD scans."
runs[0].find(qn("w:rPr")).find(qn("w:sz")).addprevious(etree.fromstring(f'<w:color xmlns:w="{WNS}" w:val="{RED}"/>')); runs[1].find(qn("w:rPr")).find(qn("w:sz")).addprevious(etree.fromstring(f'<w:color xmlns:w="{WNS}" w:val="{RED}"/>'))
dp4 = desc_par("Panels (a), (b) and (c) show F1, recall and AUROC for 1, 5 and 10 real ASD scans in three setups. Adding synthetic anomalies, guided or random, raises F1 and recall, while AUROC stays near the chance level of 0.5, which indicates that more patients get flagged without a better ranking.")
after(cap4._p, dp4._p)
# ---- table captions
def retitle_tab(p, label, text):
    runs = p._p.findall(qn("w:r")); t0 = runs[0].find(qn("w:t")); t0.text = label
    if len(runs) > 1 and runs[1].find(qn("w:t")) is not None: runs[1].find(qn("w:t")).text = text
    else: t0.text = label + text
    for r in runs[:2]:
        sz = r.find(qn("w:rPr")).find(qn("w:sz")); sz.addprevious(etree.fromstring(f'<w:color xmlns:w="{WNS}" w:val="{RED}"/>'))
retitle_tab(P(59), "TABLE III.", " Few-shot results. SA: guided, R: random.")
retitle_tab(P(67), "TABLE V. Detection of generated anomalies.", "")

# ================================================================ 4. NEW METHOD SUBSECTIONS (math + model comparison protocol)
def newpar(kind, text=None, eq=None, num=None):
    if kind == "h": return h2_par(text)._p
    if kind == "p": return body_par(text)._p
    return etree.fromstring(eq_par(eq, num))
EQ = []
xi, xj = sub(mR("x"), cat(mR("i"), mR("t"))), sub(mR("x"), cat(mR("j"), mR("t")))
xbi, xbj = sub(bar(mR("x")), mR("i")), sub(bar(mR("x")), mR("j"))
numer = cat(nary("∑", mR("t"), "", cat(paren(cat(xi, mR("−"), xbi)), paren(cat(xj, mR("−"), xbj)))))
denom = rad(cat(nary("∑", mR("t"), "", sup(paren(cat(xi, mR("−"), xbi)), mR("2"))), nary("∑", mR("t"), "", sup(paren(cat(xj, mR("−"), xbj)), mR("2")))))
eq1 = cat(sub(mR("r"), cat(mR("i"), mR("j"))), mR("="), frac(numer, denom))
_nb = nary_u("\u2211", cat(mR("j\u2208"), mR("N(i)")), cat(sub(mR("w"), cat(mR("i"), mR("j"))), subsup(mR("h"), mR("j"), paren(mR("l\u22121")))))
eq2 = cat(subsup(mR("a"), mR("i"), paren(mR("l"))), mR("="), paren(cat(mR("1+"), sub(mR("\u03b5"), mR("l")))), subsup(mR("h"), mR("i"), paren(mR("l\u22121"))), mR("+"), _nb)
eq2b = cat(subsup(mR("h"), mR("i"), paren(mR("l"))), mR("="), mU("ReLU"), paren(cat(sup(mU("MLP"), paren(mR("l"))), paren(subsup(mR("a"), mR("i"), paren(mR("l")))))))
eq3 = cat(mR("z="), sub(mR("W"), mR("o")), frac(mR("1"), mR("|V|")), nary_u("∑", mR("i∈V"), subsup(mR("h"), mR("i"), paren(mR("2")))), mR("+"), sub(mR("b"), mR("o")))
eq4 = cat(sub(mR("r"), mR("t")), mR("=σ"), paren(cat(sub(mR("W"), mR("r")), sub(mR("z"), mR("t")), mR("+"), sub(mR("U"), mR("r")), sub(mR("h"), mR("t−1")))))
eq5 = cat(sub(mR("u"), mR("t")), mR("=σ"), paren(cat(sub(mR("W"), mR("u")), sub(mR("z"), mR("t")), mR("+"), sub(mR("U"), mR("u")), sub(mR("h"), mR("t−1")))))
eq6 = cat(sub(mR("h\u0303"), mR("t")), mR("="), mU("tanh"), paren(cat(sub(mR("W"), mR("h")), sub(mR("z"), mR("t")), mR("+"), sub(mR("U"), mR("h")), paren(cat(sub(mR("r"), mR("t")), mR("\u2299"), sub(mR("h"), mR("t\u22121")))))))
eq7 = cat(sub(mR("h"), mR("t")), mR("="), paren(cat(mR("1−"), sub(mR("u"), mR("t")))), mR("⊙"), sub(mR("h"), mR("t−1")), mR("+"), sub(mR("u"), mR("t")), mR("⊙"), sub(mR("h̃"), mR("t")))
eq8 = cat(sub(hat(mR("z")), mR("t+1")), mR("="), sub(mR("W"), mR("p")), sub(mR("h"), mR("t")), mR("+"), sub(mR("b"), mR("p")))
eq9 = cat(sub(mR("L"), mU("pred")), mR("="), frac(mR("1"), mR("(T\u22121)D")), nary("\u2211", mR("t=1"), mR("T\u22121"), sup(norm(cat(sub(hat(mR("z")), mR("t+1")), mR("\u2212"), sub(mR("z"), mR("t+1")))), mR("2"))))
eq9b = cat(sub(mR("L"), mU("var")), mR("="), frac(mR("1"), mR("D")), nary("\u2211", mR("d=1"), mR("D"), cat(mU("max"), paren(cat(mR("0,1\u2212"), rad(cat(mU("Var"), paren(sub(mR("z"), mR("d"))), mR("+"), sup(mR("10"), mR("\u22124")))))))))
eq9c = cat(mR("L="), sub(mR("L"), mU("pred")), mR("+\u03bb"), sub(mR("L"), mU("var")))
eq10 = cat(sub(mR("A"), mR("t")), mR("="), sub(norm(cat(sub(mR("z"), mR("t+1")), mR("\u2212"), sub(hat(mR("z")), mR("t+1")))), mR("2")))
eq10b = cat(mR("S="), limlow(mU("max"), mR("v")), limlow(mU("max"), mR("t")), subsup(mR("A"), mR("t"), paren(mR("v"))))
eq11 = cat(sub(mR("g"), mR("p")), mR("="), nary_u("∑", mR("e∈p"), frac(cat(mR("∂"), sup(norm(cat(mR("f(G;w)−"), sub(hat(mR("z")), mR("t+1")))), mR("2"))), cat(mR("∂"), sub(mR("w"), mR("e"))))))
eq12 = cat(subsup(mR("w"), mR("p"), paren(mR("q+1"))), mR("="), sub(mU("clip"), mR("[−1,1]")), paren(cat(subsup(mR("w"), mR("p"), paren(mR("q"))), mR("+"), frac(cat(mR("α"), sub(mR("δ"), mU("max"))), mR("Q")), mU("sgn"), paren(subsup(mR("g"), mR("p"), paren(mR("q")))))))
eq13 = cat(mR("P(y=1|x)="), frac(mR("1"), cat(mR("1+exp"), paren(cat(mR("−"), paren(cat(sup(mR("β"), mR("T")), mR("x+b"))))))))
K = [("h", "K. Mathematical formulation"),
 ("p", "This subsection collects the models and algorithms used in the pipeline. In a window of 30 time points, the connectivity between regions i and j follows the Pearson correlation [2]:"), ("e", eq1, 1),
 ("p", "Only the strongest 20% of pairs by absolute correlation stay as signed edge weights, that is, an edge exists when |r_ij| reaches the 80th percentile of the window. A weighted GIN layer [24] then aggregates the neighbours N(i) of node i with the signed weights w_ij, where \u03b5_l is trainable and the input feature of every node is 1, and passes the result through a two-layer perceptron:"), ("e", eq2, 2), ("e", eq2b, 3),
 ("p", "After two layers, mean pooling over the nodes V of the subgraph and a linear projection give the 128-dimensional embedding:"), ("e", eq3, 4),
 ("p", "The GRU [25] reads the embeddings z_1, ..., z_t. Its reset gate, update gate, candidate state and hidden state follow:"), ("e", eq4, 5), ("e", eq5, 6), ("e", eq6, 7), ("e", eq7, 8),
 ("p", "A linear read-out gives the predicted next embedding:"), ("e", eq8, 9),
 ("p", "Training minimizes the mean squared prediction error plus a variance penalty in the style of VICReg [26], which stops all embeddings from collapsing to one vector (T is the number of windows, D = 128 the embedding size and \u03bb = 0.5). Adam [[kingma2015]] updates the parameters with a learning rate of 0.001:"), ("e", eq9, 10), ("e", eq9b, 11), ("e", eq9c, 12),
 ("p", "The anomaly score of step t and the patient score S, the maximum over the tracked regions v and all steps, follow:"), ("e", eq10, 13), ("e", eq10b, 14),
 ("p", "The step threshold and the patient threshold equal the 95th percentile of the validation-control step scores and patient scores, and a patient is flagged when S exceeds the patient threshold. For guided anomaly generation, the sensitivity of an undirected connection p (two directed edges e) is the gradient of the model error, where f is the frozen encoder:"), ("e", eq11, 15),
 ("p", "The m connections with the largest |g_p| move in Q = 5 projected sign-gradient steps [18], [19], where \u03b1 is the severity, \u03b4_max = 0.6 and the weights stay between \u22121 and 1:"), ("e", eq12, 16),
 ("p", "The few-shot head is a logistic regression with balanced class weights on the 20-dimensional vector x of per-region maximum scores, fitted by weighted cross-entropy with an L2 penalty (C = 1):"), ("e", eq13, 17),
 ("p", "The severity estimator is a standardized multinomial logistic regression (C = 0.1) on 40 features, the per-region maximum and mean scores. Evaluation uses precision = TP/(TP + FP), recall = TP/(TP + FN), F1 = 2PR/(P + R) and AUROC, the probability that a random positive scores above a random negative. Confidence intervals come from 2000 bootstrap resamples of the test patients, p-values from 5000 label permutations, and the relation between predicted and true severity from the Spearman rank correlation."),
 ("h", "L. Model-wise comparison protocol"),
 ("p", "To compare models, the graph encoder and the temporal predictor were exchanged while the data, the split, the loss in Eq. (12), the optimizer and the five training epochs stayed identical. The graph encoders were GIN [24], GCN [32] (neighbour weights summed without normalization, because signed weights make normalization unstable) and GraphConv [[morris2019]], and the temporal predictors were GRU [25] and LSTM [[hochreiter1997]]. Every model produced its own gradients for the guided edits and its own thresholds from the validation controls. Encoders such as GAT and GraphSAGE use no scalar edge weight in their standard form and were therefore not part of the comparison.")]
anchor = P(39)._p
for item in K:
    el = newpar(item[0], text=item[1]) if item[0] in "hp" else newpar("e", eq=item[1], num=item[2])
    anchor = after(anchor, el)

# ================================================================ 5. NEW RESULTS SUBSECTION H + Table VI
mc = json.load(open(f"{ROOT}/results/model_compare.json")) if os.path.exists(f"{ROOT}/results/model_compare.json") else {}
tabV = [el for el in body if el.tag == qn("w:tbl")][-1]
cur = tabV
hH = h2_par("H. Model-wise comparison"); cur = after(cur, hH._p)
order_ = [m for m in ["GIN + GRU (proposed)", "GCN + GRU", "GraphConv + GRU", "GIN + LSTM"] if m in mc]
if order_:
    best = max(order_, key=lambda m: mc[m]["structure_aware"]["auroc"])
    txt = ("Table VI compares encoder and predictor combinations trained under identical conditions. For every model the table lists the ASD detection AUROC and accuracy on the 131 test patients and, for the 1,400 generated edits per method, the accuracy and AUROC of guided edits and the AUROC of random edits. ")
    gm = mc["GIN + GRU (proposed)"]
    txt += f"The proposed GIN + GRU reaches an accuracy of {100*gm['structure_aware']['accuracy']:.1f}% and an AUROC of {gm['structure_aware']['auroc']:.3f} on guided edits. "
    others = [m for m in order_ if m != "GIN + GRU (proposed)"]
    cmp_ = "; ".join(f"{m} reaches {100*mc[m]['structure_aware']['accuracy']:.1f}% and {mc[m]['structure_aware']['auroc']:.3f}" for m in others)
    txt += f"The other models reach: {cmp_}. "
    rk = sorted(order_, key=lambda m: -mc[m]["structure_aware"]["auroc"]); assert rk[0] == "GIN + GRU (proposed)", rk
    txt += f"GIN + GRU gives the highest guided-edit AUROC, followed closely by {rk[1].replace(' (proposed)', '')}, while {rk[-1].replace(' (proposed)', '')} gives the lowest, which supports the choice of GIN as the graph encoder. "
    txt += "All models give ASD detection AUROC values near the chance level of 0.5, and random edits stay near 0.5 for every model."
    cur = after(cur, body_par(txt)._p)
    capVI = new_par_from(TPL_CAP_TAB, [("TABLE VI. Model-wise comparison.", True)], base=P(67)._p.findall(qn("w:r"))[0].find(qn("w:rPr")))
    cur = after(cur, capVI._p)
    W = [1060, 760, 600, 640, 760, 760]
    def cell(txt, w, bold=False):
        b = "<w:b/><w:bCs/>" if bold else ""
        return (f'<w:tc xmlns:w="{WNS}"><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>'
                f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>{b}<w:color w:val="{RED}"/><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr><w:t xml:space="preserve">{txt}</w:t></w:r></w:p></w:tc>')
    def row(vals, bold=False): return f'<w:tr xmlns:w="{WNS}"><w:trPr><w:cantSplit/></w:trPr>' + "".join(cell(v, w, bold).replace(f' xmlns:w="{WNS}"', "") for v, w in zip(vals, W)) + "</w:tr>"
    hdr = ["Model", "ASD AUROC", "ASD Acc.", "Guided Acc.", "Guided AUROC", "Random AUROC"]
    rows = [row(hdr, True)]
    for m in order_:
        r = mc[m]; rows.append(row([m.replace(" (proposed)", ""), f"{r['asd_auroc']:.3f}", f"{100*r['asd_accuracy']:.1f}%", f"{100*r['structure_aware']['accuracy']:.1f}%", f"{r['structure_aware']['auroc']:.3f}", f"{r['random']['auroc']:.3f}"], bold=(m == best)))
    bd = "".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="000000"/>' for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
    tbl = etree.fromstring(f'<w:tbl xmlns:w="{WNS}"><w:tblPr><w:tblW w:w="{sum(W)}" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>{bd}</w:tblBorders><w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="40" w:type="dxa"/><w:right w:w="40" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>' + "".join(f'<w:gridCol w:w="{w}"/>' for w in W) + "</w:tblGrid>" + "".join(rows) + "</w:tbl>")
    cur = after(cur, tbl)
    cur = after(cur, body_par(" ")._p)   # spacer

# ================================================================ 6. CITATION NUMBERING (sequence of first appearance) + write runs
TOK = re.compile(r"\[\[(\w+)\]\]|\[(\d+)\]")
intro = P(4)._p; refs_head = P(72)._p
order = []; pending = []
inside = False
for el in body:
    if el is intro: inside = True
    if el is refs_head: break
    if not inside or el.tag != qn("w:p"): continue
    if el in SPEC: segs = SPEC[el]["segs"]
    else:
        txt = "".join(t.text or "" for t in el.iter(qn("w:t"))); segs = [(txt, False)]
    for t, red in segs:
        for m in TOK.finditer(t):
            key = m.group(1) or OLD[int(m.group(2))]
            if key not in order: order.append(key)
num = {k: i + 1 for i, k in enumerate(order)}
print("cited:", len(order))
def expand(segs):
    out = []
    for t, red in segs:
        pos = 0
        for m in TOK.finditer(t):
            if m.start() > pos: out.append((t[pos:m.start()], red))
            if m.group(1): out.append((f"[{num[m.group(1)]}]", True))
            else:
                n = int(m.group(2)); nn = num[OLD[n]]; out.append((f"[{nn}]", red or nn != n))
            pos = m.end()
        if pos < len(t): out.append((t[pos:], red))
    merged = []
    for t, red in out:
        if merged and merged[-1][1] == red: merged[-1] = (merged[-1][0] + t, red)
        else: merged.append((t, red))
    return merged
for el in body:
    if el is refs_head: break
    if not inside and el is not intro and el not in SPEC: continue
    if el.tag != qn("w:p"): continue
    if el in SPEC: s = SPEC[el]; segs = s["segs"]; keep = s["keep"]; base = s["base"]; p = s["p"]
    else:
        txt = "".join(t.text or "" for t in el.iter(qn("w:t")))
        if not TOK.search(txt): continue
        from docx.text.paragraph import Paragraph
        p = Paragraph(el, None); segs = [(txt, False)]; keep = 0; base = first_text_rpr(p); s = {"changed": False}
    ex = expand(segs)
    if ex == segs and not s["changed"] and el not in SPEC: continue
    if not s["changed"] and all(not r for _, r in ex): continue
    if ex == segs and not s["changed"]: continue
    clear_runs(p, keep)
    for t, red in ex: p._p.append(mkrun(t, base, red))

# ================================================================ 7. REBUILD REFERENCE LIST in citation order
ref_paras = []
started = False
for el in list(body):
    if el is refs_head: started = True; continue
    if started and el.tag == qn("w:p"): ref_paras.append(el)
tpl = ref_paras[0]; lab_rpr = tpl.findall(qn("w:r"))[0].find(qn("w:rPr")); txt_rpr = [r for r in tpl.findall(qn("w:r")) if r.find(qn("w:t")) is not None][1].find(qn("w:rPr"))
oldnum_of = {v: k for k, v in OLD.items()}
for el in ref_paras: body.remove(el)
sect = body.find(qn("w:sectPr")); anchor = refs_head
for k in order:
    n = num[k]; el = etree.Element(qn("w:p")); el.append(copy.deepcopy(tpl.find(qn("w:pPr"))))
    changed = (oldnum_of.get(k) != n)
    el.append(mkrun(f"[{n}]", lab_rpr, changed)); tr_ = etree.SubElement(el, qn("w:r")); etree.SubElement(tr_, qn("w:tab"))
    for t, it in R_[k]:
        base = copy.deepcopy(txt_rpr)
        for i_ in base.findall(qn("w:i")) + base.findall(qn("w:iCs")): base.remove(i_)
        if it:
            i1 = etree.Element(qn("w:i")); i1.set(qn("w:val"), "1"); i2 = etree.Element(qn("w:iCs")); i2.set(qn("w:val"), "1"); sz = base.find(qn("w:sz")); sz.addprevious(i1); sz.addprevious(i2)
        el.append(mkrun(t.replace("\\_", "_"), base, k in ("kingma2015", "hochreiter1997", "morris2019")))
    anchor.addnext(el); anchor = el
d.save(sys.argv[2]); print("saved", sys.argv[2], "refs:", len(order))
json.dump({"order": order, "num": num}, open("/private/tmp/claude-501/nr/refnum.json", "w"))
