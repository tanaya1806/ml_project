"""Re-apply IEEE-conference spacing/formatting to the user's edited docx WITHOUT changing any wording.
usage: python3 format_ieee.py in.docx out.docx"""
import re, sys, zipfile, shutil, os
src, dst = sys.argv[1:3]
z = zipfile.ZipFile(src); files = {n: z.read(n) for n in z.namelist()}
d = files["word/document.xml"].decode("utf8")

FONT = '<w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>'
COLW, GAP = 5040, 360                       # IEEE: 3.5 in columns, 0.25 in gutter
MARG = dict(top=1080, bottom=1440, left=900, right=900)   # 0.75 / 1.0 / 0.625 / 0.625 in

# ---- page geometry (both sections) ----
def sect(m):
    s = m.group(0)
    s = re.sub(r'<w:pgMar [^>]*/>', '<w:pgMar w:bottom="%d" w:top="%d" w:left="%d" w:right="%d" w:header="708" w:footer="708" w:gutter="0"/>' % (MARG["bottom"], MARG["top"], MARG["left"], MARG["right"]), s)
    if "<w:cols" in s:
        s = re.sub(r'<w:cols.*?</w:cols>', '<w:cols w:equalWidth="1" w:num="2" w:space="%d"/>' % GAP, s, flags=re.S)
    return s
d = re.sub(r'<w:sectPr.*?</w:sectPr>', sect, d, flags=re.S)

# ---- helpers ----
def fix_runs(p, sz):
    """normalise font (and size when sz given) on every run, keep bold/italic/sub/superscript"""
    def run(m):
        head, rest = m.group(1), m.group(2)
        mm = re.match(r'(<w:rPr>(.*?)</w:rPr>|<w:rPr/>)', rest, flags=re.S)
        inner = (mm.group(2) or "") if mm else ""
        tail = rest[mm.end():] if mm else rest
        if sz is not None:
            inner = re.sub(r'<w:sz [^>]*/>|<w:szCs [^>]*/>', '', inner)
        inner = re.sub(r'<w:rFonts [^>]*/>', '', inner)
        if sz is not None:
            szx = '<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (sz, sz)
            for key in ('<w:vertAlign', '<w:rtl'):
                if key in inner:
                    inner = inner.replace(key, szx + key, 1); break
            else:
                inner += szx
        return head + '<w:rPr>' + FONT + inner + '</w:rPr>' + tail
    return re.sub(r'(<w:r\b[^>]*>)((?:(?!</w:r>).)*</w:r>)', lambda m: run(m).replace('</w:r></w:r>', '</w:r>') if False else run(m), p, flags=re.S)

def ppr(keep=False, before=0, after=0, line=240, left=None, hanging=None, first=None, jc="both", tabs=None, mark=None):
    s = '<w:pPr>'
    if keep: s += '<w:keepNext w:val="1"/>'
    if tabs: s += '<w:tabs><w:tab w:val="left" w:pos="%d"/></w:tabs>' % tabs
    s += '<w:spacing w:before="%d" w:after="%d" w:line="%d" w:lineRule="auto"/>' % (before, after, line)
    if left is not None or first is not None or hanging is not None:
        s += '<w:ind' + (' w:left="%d"' % left if left is not None else '') + (' w:hanging="%d"' % hanging if hanging is not None else '') + (' w:firstLine="%d"' % first if first is not None else '') + '/>'
    s += '<w:jc w:val="%s"/>' % jc
    if mark: s += '<w:rPr>%s<w:sz w:val="%d"/><w:szCs w:val="%d"/></w:rPr>' % (FONT, mark, mark)
    return s + '</w:pPr>'

def set_ppr(p, new):
    p = re.sub(r'<w:pPr>.*?</w:pPr>|<w:pPr/>', '', p, count=1, flags=re.S)
    return re.sub(r'(<w:p\b[^>]*>)', lambda m: m.group(1) + new, p, count=1)

text_of = lambda p: "".join(re.findall(r'<w:t(?: [^>]*)?>(.*?)</w:t>', p, flags=re.S))
tbl_spans = [(m.start(), m.end()) for m in re.finditer(r'<w:tbl>.*?</w:tbl>', d, flags=re.S)]
tbl_idx = lambda pos: next((i for i, (a, b) in enumerate(tbl_spans) if a <= pos < b), None)

H1 = re.compile(r'^(I|II|III|IV|V|VI|VII|VIII|IX|X)\. [A-Z][A-Z ,&\-]+$')
H2 = re.compile(r'^[A-K]\. ')

out, pos, pend = [], 0, None
matches = list(re.finditer(r'<w:p[ >].*?</w:p>', d, flags=re.S))
skip = set()
n_empty_removed = 0
for k, m in enumerate(matches):
    if k in skip: continue
    out.append(d[pos:m.start()]); pos = m.end()
    p = m.group(0); t = text_of(p); ti = tbl_idx(m.start())
    has_sect, has_draw = '<w:sectPr' in p, '<w:drawing>' in p
    # --- merge caption continuation ("Fig. 1." + "Illustration: ...") into one paragraph
    if t.startswith("Fig.") and k + 1 < len(matches) and text_of(matches[k + 1].group(0)).startswith("Illustration:"):
        nxt = matches[k + 1].group(0); skip.add(k + 1); pos = matches[k + 1].end()
        runs2 = re.findall(r'<w:r\b.*?</w:r>', nxt, flags=re.S)
        p = p.replace('</w:p>', "".join(runs2) + '</w:p>'); t = text_of(p)
    if ti == 0:                                   # author table: +1 pt (8 -> 9 pt)
        p = fix_runs(set_ppr(p, ppr(after=0, jc="center", mark=18)), 18) if t.strip() else p
        out.append(p); continue
    if ti is not None:                            # data tables: 8 pt, centred, single spaced
        p = fix_runs(set_ppr(p, ppr(after=0, jc="center", mark=16)), 16)
        out.append(p); continue
    if has_sect:
        p = set_ppr(p, ppr(after=120, jc="left", mark=8)) .replace('</w:pPr>', '', 0)
        # keep the sectPr that lived inside the old pPr
        old = re.search(r'<w:sectPr.*?</w:sectPr>', m.group(0), flags=re.S).group(0)
        old = sect(re.match(r'.*', old, flags=re.S)) if False else re.sub(r'<w:pgMar [^>]*/>', '<w:pgMar w:bottom="%d" w:top="%d" w:left="%d" w:right="%d" w:header="708" w:footer="708" w:gutter="0"/>' % (MARG["bottom"], MARG["top"], MARG["left"], MARG["right"]), old)
        p = p.replace('</w:pPr>', old + '</w:pPr>', 1)
        out.append(p); continue
    if not t.strip() and not has_draw:            # stray empty paragraphs outside tables
        n_empty_removed += 1; continue
    if k == 0:                                    # title
        p = fix_runs(set_ppr(p, ppr(after=200, jc="center")), None)
    elif has_draw:
        p = set_ppr(p, ppr(keep=True, before=80, after=0, jc="center"))
        p = re.sub(r'<w:pPr>', '<w:pPr>', p)
    elif t.startswith("Abstract:") or t.startswith("Keywords:"):
        p = fix_runs(set_ppr(p, ppr(after=120, jc="both", mark=18)), 18)
    elif t.startswith("TABLE"):
        p = fix_runs(set_ppr(p, ppr(keep=True, before=120, after=60, jc="center", mark=16)), 16)
    elif t.startswith("Fig."):
        p = fix_runs(set_ppr(p, ppr(before=40, after=160, jc="both", mark=16)), 16)
    elif H1.match(t.strip()) or t.strip() == "REFERENCES":
        p = fix_runs(set_ppr(p, ppr(keep=True, before=200, after=80, jc="center", mark=20)), 20)
    elif H2.match(t) and len(t) < 90 and re.search(r'<w:i(?: w:val="(?:1|true)")?/>', p):
        p = fix_runs(set_ppr(p, ppr(keep=True, before=100, after=40, jc="left", mark=20)), 20)
    elif 'w:hanging="300"' in p or t.startswith("[") and re.match(r'^\[\d+\]', t):
        # references: 8 pt, hanging indent; split runs on line breaks into one paragraph per entry
        base = ppr(before=0, after=20, left=360, hanging=360, tabs=360, jc="both", mark=16)
        if '<w:br' in p:
            def expand(r):   # a run may contain several <w:br/>: split it into text runs separated by pure-break runs
                if '<w:br' not in r: return [r]
                m0 = re.match(r'(<w:r\b[^>]*>)(<w:rPr>.*?</w:rPr>|<w:rPr/>)?(.*)</w:r>$', r, flags=re.S)
                head, rpr, inner = m0.group(1), m0.group(2) or "", m0.group(3)
                parts = re.split(r'<w:br\b[^>]*/>', inner); out_ = []
                for k_, part in enumerate(parts):
                    if part.strip(): out_.append(head + rpr + part + '</w:r>')
                    if k_ < len(parts) - 1: out_.append('<w:r><w:br w:type="textWrapping"/></w:r>')
                return out_
            runs = [x_ for r_ in re.findall(r'<w:r\b.*?</w:r>', p, flags=re.S) for x_ in expand(r_)]; groups, cur = [], []
            for r in runs:
                if '<w:br' in r: groups.append(cur); cur = []
                else: cur.append(r)
            groups.append(cur)
            head = re.match(r'<w:p\b[^>]*>', p).group(0)
            p = "".join(fix_runs(head + base + "".join(g) + '</w:p>', 16) for g in groups if "".join(g).strip())
        else:
            p = fix_runs(set_ppr(p, base), 16)
    else:                                         # body text
        p = fix_runs(set_ppr(p, ppr(before=0, after=0, first=288, jc="both", mark=20)), 20)
    out.append(p)
out.append(d[pos:]); d = "".join(out)

d = re.sub(r'(<w:t(?: [^>]*)?>\[\d+\])(</w:t></w:r>)(<w:r\b)', lambda m: m.group(1) + m.group(2) + '<w:r><w:tab/></w:r>' + m.group(3), d)
# ---- widen the generated-anomaly table so headers do not wrap ----
def widen(m):
    t = m.group(0)
    if 'Sub-structure' not in t or 'Random' not in t: return t
    import itertools; W = [1000, 920, 900, 740, 620, 820]; k = itertools.cycle(W); t = re.sub(r'<w:gridCol w:w="\d+"/>', lambda _: '<w:gridCol w:w="%d"/>' % next(k), t)
    k = itertools.cycle(W); t = re.sub(r'<w:tcW w:w="\d+" w:type="dxa"/>', lambda _: '<w:tcW w:w="%d" w:type="dxa"/>' % next(k), t)
    return re.sub(r'<w:tblW w:w="[\d.]+" w:type="dxa"/>', '<w:tblW w:w="%d" w:type="dxa"/>' % sum(W), t)
d = re.sub(r'<w:tbl>.*?</w:tbl>', widen, d, flags=re.S)
# ---- all text black (automatic colour), no highlights, no shading ----
d = re.sub(r'<w:color [^>]*/>', '', d); d = re.sub(r'<w:highlight [^>]*/>', '', d)
d = re.sub(r'<w:shd [^>]*/>', '', d)
# ---- word spacing: collapse accidental double spaces inside text runs ----
d = re.sub(r'(<w:t[^>]*>)([^<]*)(</w:t>)', lambda m: m.group(1) + re.sub(r' {2,}', ' ', m.group(2)) + m.group(3), d)
# remove empty trailing paragraph(s) left at the very end of the body
d = re.sub(r'(<w:p\b(?:(?!</w:p>).)*?<w:rPr><w:rtl w:val="0"/></w:rPr></w:r></w:p>)(<w:sectPr)', r'\2', d, flags=re.S) if False else d
files["word/document.xml"] = d.encode("utf8")

# ---- default font/size: Times New Roman 10 pt ----
s = files["word/styles.xml"].decode("utf8")
s = re.sub(r'<w:docDefaults>.*?</w:docDefaults>', '<w:docDefaults><w:rPrDefault><w:rPr>' + FONT + '<w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>', s, flags=re.S)
files["word/styles.xml"] = s.encode("utf8")

with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as o:
    for n, b in files.items(): o.writestr(n, b)
print("empty paragraphs removed:", n_empty_removed)
