"""Rephrase are/we/our/us sentences and drop uncited references with renumbering by first appearance."""
import sys, re, docx
from docx.oxml.ns import qn
src, out = sys.argv[1], sys.argv[2]
d = docx.Document(src); body = d.element.body
R = [
("very few labelled scans are available", "very few labelled scans exist"),
("the anomalous regions are given a high anomaly score", "anomalous regions receive a high anomaly score"),
("suitable controls are used for the formation", "suitable controls serve for the formation"),
("These guided edits are detected with", "These guided edits get detected with"),
("many more patients are classified", "many more patients get classified"),
("The obtained data are processed in such a way", "The obtained data get processed in such a way"),
("the links between nodes are representative of correlation", "the links between nodes represent correlation"),
("brain networks are not fixed entities", "brain networks do not stay fixed"),
("Our analysis will answer", "The analysis will answer"),
("anomalies are created instead of model attacks", "anomalies get created instead of model attacks"),
("and are assessed using a random baseline", "and get assessed using a random baseline"),
("Individuals in the dataset are distinguished based on three groups", "Individuals in the dataset fall into three groups"),
("the correlations between all areas are computed", "the correlations between all areas get computed"),
("brain are selected in this regard", "brain get selected in this regard"),
("Only connections and their weights are altered", "Only connections and their weights change"),
("Both systems are trained concurrently", "Both systems get trained concurrently"),
("the weights that are already in place are modified", "the weights already in place get modified"),
("The most important weights that are retained are those whose contribution does not exceed 15%", "The most important weights retained have a contribution that does not exceed 15%"),
("how many links are changed", "how many links change"),
("The same 20 digits are evaluated", "The same 20 digits get evaluated"),
("negatives that are control scans", "negatives taken from control scans"),
("synthetic anomalies that are created through the guidance of real scans", "synthetic anomalies created through the guidance of real scans"),
("they are considered to be in the marked category", "they count as being in the marked category"),
("193 scores are gathered from", "193 scores come from"),
("Step threshold and the patient threshold are equal in the sense that both of them equal the 95th percent of validation-control step score and patient score.", "The step threshold and the patient threshold each equal the 95th percentile of the validation-control step scores and patient scores, respectively."),
("The derived values of the performance of the detector are precision as", "The derived performance values of the detector give precision as"),
("ROC curve are found in Fig. 2", "ROC curve appear in Fig. 2"),
("the ones that are categorized as ASD", "the ones categorized as ASD"),
("these percentages are as follows", "these percentages read as follows"),
("the connections are ranked based on their influence", "the connections get ranked based on their influence"),
("the F1 metrics are practically the same", "the F1 metrics stay practically the same"),
("As far as the 60 ASD people are concerned who failed to enter the training scheme", "For the 60 ASD people who failed to enter the training scheme"),
("changes are determined solely by the existing", "changes get determined solely by the existing"),
("this implies that the generator is not producing random noise, indicating that the anomalies in the produced records are similar to autism", "this implies that the generator is not producing random noise, not that the anomalies in the produced records resemble autism"),
("guided sub-structure changes are recognized far better", "guided sub-structure changes get recognized far better"),
("generator are good and, respectively,", "generator prove good and, respectively,"),
]
paras = [p for p in d.paragraphs if not re.match(r"^\[\d+\]", p.text)]
for a, b in R:
    hits = [p for p in paras if a in p.text]
    assert len(hits) == 1, (a, len(hits))
    p = hits[0]
    for r in p.runs:
        if a in r.text: r.text = r.text.replace(a, b); break
    else: raise SystemExit("split across runs: " + a)
# residual check
res = [s for p in paras for s in re.findall(r"[^.]*\b(?:are|we|our|us|We|Our|ours)\b[^.]*\.", p.text)]
print("residual:", res)
# references
allp = d.paragraphs; ri = [i for i, p in enumerate(allp) if p.text.strip() == "REFERENCES"][0]
refp = {int(re.match(r"\[(\d+)\]", p.text).group(1)): p for p in allp[ri + 1:] if re.match(r"^\[\d+\]", p.text)}
refset = {id(p._p) for p in refp.values()}
def in_refs(t):
    par = t.getparent()
    while par is not None and par.tag != qn("w:p"): par = par.getparent()
    return id(par) in refset
order = []
for t in body.iter(qn("w:t")):
    if in_refs(t) or not t.text: continue
    for m in re.finditer(r"\[(\d+)\]", t.text):
        n = int(m.group(1))
        if n not in order: order.append(n)
M = {o: i + 1 for i, o in enumerate(order)}
print("dropping", sorted(set(refp) - set(order)), "kept", len(order))
for t in body.iter(qn("w:t")):
    if in_refs(t) or not t.text: continue
    t.text = re.sub(r"\[(\d+)\]", lambda m: "[%d]" % M[int(m.group(1))], t.text)
for n, p in refp.items():
    if n not in M: p._p.getparent().remove(p._p); continue
    for t in p._p.iter(qn("w:t")):
        if t.text and re.match(r"\[\d+\]", t.text): t.text = re.sub(r"^\[\d+\]", "[%d]" % M[n], t.text, count=1); break
    else: raise SystemExit("label %d" % n)
items = sorted(((M[n], p) for n, p in refp.items() if n in M), key=lambda x: x[0])
anchor = items[0][1]._p.getprevious()
for _, p in items: anchor.addnext(p._p); anchor = p._p
d.save(out)
