"""Builds an editable Word drawing canvas (native shapes + arrow connectors) for the system architecture diagram."""
from xml.sax.saxutils import escape
EMU = 914400
RED = "000000"
NS = ('xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
      'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"')
STEPS = [
 ("DATA PREPARATION", "ABIDE I, 871 scans, 360 regions; patient-level split 609 / 131 / 131"),
 ("SLIDING-WINDOW BRAIN GRAPHS", "30-point windows, stride 10; Pearson correlation; top 20% signed edges"),
 ("SUBGRAPH TRACKING", "20 brain regions with 1-hop neighbours; node set fixed at the first window"),
 ("GIN ENCODING AND GRU PREDICTION", "2-layer GIN gives a 128-D vector; GRU predicts the next window; controls only"),
 ("ANOMALY SCORING", "Prediction error per region and step; patient score = maximum"),
 ("THRESHOLD CALIBRATION", "95th percentile of validation-control scores (step and patient level)"),
 ("ANOMALY GENERATION", "Gradient-guided edge-weight edits, 5 steps, up to 15% of edge pairs"),
 ("FEW-SHOT ADAPTATION", "1, 5 or 10 ASD scans + 30 generated anomalies; logistic regression"),
 ("ANOMALY DETECTION", "Patient is flagged when the score exceeds the patient threshold"),
 ("SEVERITY LEVEL ANALYSIS", "Flagged patient gets an ADOS-2 band: low, moderate or high"),
]
def para(text, sz, bold, red=True):
    col = f'<w:color w:val="{RED}"/>' if red else ""
    b = "<w:b/><w:bCs/>" if bold else ""
    return (f'<w:p><w:pPr><w:spacing w:before="0" w:after="{20 if bold else 0}" w:line="228" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>'
            f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>{b}{col}<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/></w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>')
def box(idx, name, x, y, w, h, head, detail):
    return (f'<wps:wsp><wps:cNvPr id="{idx}" name="{name}"/><wps:cNvSpPr/><wps:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
            f'<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val 7000"/></a:avLst></a:prstGeom><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>'
            f'<a:ln w="9525"><a:solidFill><a:srgbClr val="000000"/></a:solidFill></a:ln></wps:spPr>'
            f'<wps:txbx><w:txbxContent>{para(head, 14, True)}{para(detail, 13, False)}</w:txbxContent></wps:txbx>'
            f'<wps:bodyPr rot="0" vert="horz" wrap="square" lIns="32000" tIns="22000" rIns="32000" bIns="22000" anchor="ctr" anchorCtr="0"><a:noAutofit/></wps:bodyPr></wps:wsp>')
def conn(idx, name, x, y, cx, cy, flipH=False):
    fl = ' flipH="1"' if flipH else ""
    return (f'<wps:wsp><wps:cNvPr id="{idx}" name="{name}"/><wps:cNvCnPr/><wps:spPr><a:xfrm{fl}><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="straightConnector1"><a:avLst/></a:prstGeom><a:ln w="12700"><a:solidFill><a:srgbClr val="000000"/></a:solidFill><a:tailEnd type="triangle" w="med" len="med"/></a:ln></wps:spPr><wps:bodyPr/></wps:wsp>')
def canvas_xml(docpr_id=900):
    bw, bh, gx, gy = int(1.5 * EMU), int(0.64 * EMU), int(0.30 * EMU), int(0.14 * EMU); x0 = int(0.04 * EMU)
    W = x0 * 2 + 2 * bw + gx; H = 5 * bh + 4 * gy + int(0.04 * EMU)
    rows = [(1, 2), (4, 3), (5, 6), (8, 7), (9, 10)]       # (left, right) step numbers per row; flow snakes 1>2, 3>4 (leftwards), 5>6, 7>8 (leftwards), 9>10
    pos = {}; shapes = []; sid = 1
    for r, (l, rt) in enumerate(rows):
        y = int(0.02 * EMU) + r * (bh + gy)
        pos[l] = (x0, y); pos[rt] = (x0 + bw + gx, y)
    for n in range(1, 11):
        x, y = pos[n]; h, d = STEPS[n - 1]; shapes.append(box(sid, f"Step {n}", x, y, bw, bh, h, d)); sid += 1
    # arrows
    for n in range(1, 10):
        (x1, y1), (x2, y2) = pos[n], pos[n + 1]
        if y1 == y2:                                       # same row: horizontal
            if x2 > x1: shapes.append(conn(sid, f"Arrow {n}-{n+1}", x1 + bw, y1 + bh // 2, gx, 0))
            else: shapes.append(conn(sid, f"Arrow {n}-{n+1}", x2 + bw, y1 + bh // 2, gx, 0, flipH=True))
        else:                                              # next row, same column: vertical
            shapes.append(conn(sid, f"Arrow {n}-{n+1}", x1 + bw // 2, y1 + bh, 0, gy))
        sid += 1
    cx, cy = W, H
    return (f'<w:drawing {NS}><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{docpr_id}" name="System architecture canvas"/><wp:cNvGraphicFramePr/><a:graphic><a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas">'
            f'<wpc:wpc><wpc:bg><a:noFill/></wpc:bg><wpc:whole><a:ln><a:noFill/></a:ln></wpc:whole>{"".join(shapes)}</wpc:wpc></a:graphicData></a:graphic></wp:inline></w:drawing>')
