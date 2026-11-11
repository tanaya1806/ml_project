"""Cut sentences, drop uncited references and renumber 1..N by first appearance, drop demo subsection."""
import sys, re, copy, docx
from docx.oxml.ns import qn
src, out = sys.argv[1], sys.argv[2]
d = docx.Document(src); body = d.element.body
def find(start):
    for p in d.paragraphs:
        if p.text.startswith(start): return p
    raise SystemExit("missing " + start)
def merge(p):
    rs = p.runs; rs[0].text = p.text
    for r in rs[1:]: r._r.getparent().remove(r._r)
def cut(p, *sentences):
    merge(p); t = p.runs[0].text
    for s in sentences:
        assert s in t, s; t = t.replace(s, "")
    p.runs[0].text = re.sub(r" {2,}", " ", t).strip() + " "
cut(find("It should be taken"), " What is more, the small number of and the HR (high resolution) scans done in patients does not allow some of the neuro/verbal tests to indicate serious brain disorders.")
cut(find("The brain activity measures"), " BrainNetCNN employs special filters for analysis of connectivity matrices [6]. Population graph models use each person as a node in the graph [7].")
cut(find("Research in the area"), " Literature reviews indicate many approaches to researching the dynamic connectome [13]. STAGIN utilizes a graph network to each window, hence categorizing the entire scan study [14]; that is similar to what was done in [15].")
cut(find("Graph anomaly detection"), " Classical scoring aims at seeing how well the model reconstructs a node [19]. The one-class methods deal with building a compact description of normal data from which the instances that deviate should be flagged [20]. Graph-level implementations of the above-stated methods exist as well [21].")
cut(find("Various methods in the field"), "Various methods in the field of meta-learning, such as MAML [28] and prototype-based techniques [29], show great degree of adaptability toward new tasks based on a few training instances.")
# demo subsection
h = find("F. Demo application"); nxt = h._p.getnext(); body.remove(nxt); body.remove(h._p)
for a, b in (("G. Guided sub-structure versus random edits", "F. Guided sub-structure versus random edits"), ("H. Model-wise comparison", "G. Model-wise comparison")):
    for r in find(a).runs:
        if a[:2] in r.text: r.text = r.text.replace(a[:2], b[:2], 1); break
c = find("The contribution consists"); merge(c)
c.runs[0].text = c.runs[0].text.replace("five parts", "five parts").replace("(5) a full mathematical description and a demo application with 20 sample scans.", "(5) a full mathematical description of every model used.")
# ---- references: drop uncited, renumber by first appearance
paras = list(d.paragraphs); ri = [i for i, p in enumerate(paras) if p.text.strip() == "REFERENCES"][0]
refp = {}
for p in paras[ri + 1:]:
    m = re.match(r"\[(\d+)\]", p.text)
    if m: refp[int(m.group(1))] = p
refset = {id(p._p) for p in refp.values()}
order = []
for t in body.iter(qn("w:t")):
    par = t.getparent().getparent()
    if par.tag != qn("w:p"): par = par.getparent()
    if id(par) in refset or not t.text: continue
    for m in re.finditer(r"\[(\d+)\]", t.text):
        n = int(m.group(1))
        if n not in order: order.append(n)
M = {o: i + 1 for i, o in enumerate(order)}
print("kept", len(order), "dropped", sorted(set(refp) - set(order)))
for t in body.iter(qn("w:t")):
    par = t.getparent().getparent()
    if par.tag != qn("w:p"): par = par.getparent()
    if id(par) in refset or not t.text: continue
    t.text = re.sub(r"\[(\d+)\]", lambda m: "[%d]" % M[int(m.group(1))], t.text)
for n, p in refp.items():
    if n not in M: p._p.getparent().remove(p._p)
    else:
        for t in p._p.iter(qn("w:t")):
            if t.text and re.match(r"\[\d+\]", t.text): t.text = re.sub(r"^\[\d+\]", "[%d]" % M[n], t.text, 1); break
        else: raise SystemExit("label run not found %d" % n)
# physically reorder reference paragraphs
last = None; items = sorted(((M[n], p) for n, p in refp.items() if n in M), key=lambda x: x[0])
first = items[0][1]._p; anchor = first.getprevious()
for _, p in items: anchor.addnext(p._p); anchor = p._p
d.save(out)
