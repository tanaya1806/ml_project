"""Rebuild the Fig. 1 canvas with the compact layout and refresh its picture fallback (for Google Docs)."""
import sys, io, subprocess, os, re, docx
from lxml import etree
from docx.oxml.ns import qn
from PIL import Image, ImageChops
sys.path.insert(0, os.path.dirname(__file__))
from canvas import canvas_xml
src, out, work = sys.argv[1:4]
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
def swap(d):
    ac = d.element.body.find(".//{%s}AlternateContent" % MC)
    ch = ac.find("{%s}Choice" % MC); old = ch.find(qn("w:drawing"))
    new = etree.fromstring(canvas_xml(900)); ch.replace(old, new)
    return ac, new
d = docx.Document(src); ac, new = swap(d)
ext = new.find(".//" + qn("wp:extent")); cx, cy = int(ext.get("cx")), int(ext.get("cy"))
fb = ac.find("{%s}Fallback" % MC); pic = fb.find(qn("w:drawing"))
for e in pic.iter(qn("wp:extent")): e.set("cx", str(cx)); e.set("cy", str(cy))
for e in pic.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}ext"): e.set("cx", str(cx)); e.set("cy", str(cy))
tmp = os.path.join(work, "fig1_tmp.docx"); d.save(tmp)
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", work, tmp], capture_output=True)
pdf = os.path.join(work, "fig1_tmp.pdf")
# locate the figure: between the sentence "Fig. 1 shows" and the caption "System architecture diagram."
bb = subprocess.run(["pdftotext", "-bbox", pdf, "-"], capture_output=True, text=True).stdout
pages = bb.split("<page ")[1:]
for pn, pg in enumerate(pages, 1):
    if "architecture" in pg and "diagram." in pg: break
words = re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>', pg)
cap = [w for w in words if w[4] == "architecture"][0]
cap_y = float(cap[1])
first = [w for w in words if w[4] == "DATA" and float(w[1]) < cap_y][0]; top_y = float(first[1]) - 12
cap_x = float(cap[0])
DPI = 400; k = DPI / 72
subprocess.run(["pdftoppm", "-r", str(DPI), "-f", str(pn), "-l", str(pn), "-png", pdf, os.path.join(work, "f1pg")], capture_output=True)
img = Image.open([os.path.join(work, f) for f in sorted(os.listdir(work)) if f.startswith("f1pg")][0]).convert("RGB")
colx0 = 310 * k   # right column starts right of the page centre (letter, 612pt wide)
box = (int(colx0), int(top_y * k) + 4, int(img.width - 40 * k), int(cap_y * k) - 4)
crop = img.crop(box); bg = Image.new("RGB", crop.size, "white")
bbox = ImageChops.difference(crop, bg).getbbox(); crop = crop.crop((max(bbox[0] - 6, 0), max(bbox[1] - 6, 0), bbox[2] + 6, bbox[3] + 6))
buf = io.BytesIO(); crop.save(buf, "PNG")
rid = pic.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip").get(qn("r:embed"))
d.part.related_parts[rid]._blob = buf.getvalue()
# keep the fallback's aspect ratio equal to the crop
ratio = crop.height / crop.width; ncy = int(cx * ratio)
for e in pic.iter(qn("wp:extent")): e.set("cy", str(ncy))
for e in pic.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}ext"): e.set("cy", str(ncy))
d.save(out); print("canvas", cx / 914400, cy / 914400, "fallback", crop.size)
