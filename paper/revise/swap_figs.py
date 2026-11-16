"""Replace the Fig. 2 / Fig. 3 pictures in the paper with the 600 dpi versions, at full column width (3.4 in)."""
import sys, docx
from docx.oxml.ns import qn
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
d = docx.Document(src); body = d.element.body
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"; MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
new = ["paper/figures_hires/fig_detector_col.png", "paper/figures_hires/fig_generation_col.png"]
pics = []
for inl in body.iter(qn("wp:inline")):
    if any(p.tag == MC + "Fallback" for p in inl.iterancestors()): continue
    blip = inl.find(".//" + A + "blip")
    if blip is not None: pics.append((inl, blip))
assert len(pics) == 2, len(pics)
for (inl, blip), path in zip(pics, new):
    rid = blip.get(qn("r:embed")); part = d.part.related_parts[rid]
    part._blob = open(path, "rb").read()
    w, h = Image.open(path).size; cx = int(3.4 * 914400); cy = int(cx * h / w)
    inl.find(qn("wp:extent")).set("cx", str(cx)); inl.find(qn("wp:extent")).set("cy", str(cy))
    for e in inl.iter(A + "ext"): e.set("cx", str(cx)); e.set("cy", str(cy))
    print(path, "->", round(cx / 914400, 2), "x", round(cy / 914400, 2), "in")
d.save(out)
