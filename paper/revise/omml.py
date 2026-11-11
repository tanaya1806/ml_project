"""Tiny OMML (Word equation) builder. All runs are red (changed content)."""
from xml.sax.saxutils import escape
M = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
SZ = 18
def R(t, up=False):
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if up else ""
    return (f'<m:r>{sty}<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:color w:val="FF0000"/><w:sz w:val="{SZ}"/><w:szCs w:val="{SZ}"/></w:rPr>'
            f'<m:t xml:space="preserve">{escape(t)}</m:t></m:r>')
def U(t): return R(t, True)
def cat(*parts): return "".join(parts)
def sub(b, s): return f'<m:sSub><m:e>{b}</m:e><m:sub>{s}</m:sub></m:sSub>'
def sup(b, s): return f'<m:sSup><m:e>{b}</m:e><m:sup>{s}</m:sup></m:sSup>'
def subsup(b, s, p): return f'<m:sSubSup><m:e>{b}</m:e><m:sub>{s}</m:sub><m:sup>{p}</m:sup></m:sSubSup>'
def frac(n, d): return f'<m:f><m:num>{n}</m:num><m:den>{d}</m:den></m:f>'
def rad(e): return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{e}</m:e></m:rad>'
def nary(ch, lo, hi, e): return f'<m:nary><m:naryPr><m:chr m:val="{ch}"/><m:limLoc m:val="subSup"/></m:naryPr><m:sub>{lo}</m:sub><m:sup>{hi}</m:sup><m:e>{e}</m:e></m:nary>'
def nary_u(ch, lo, e): return f'<m:nary><m:naryPr><m:chr m:val="{ch}"/><m:limLoc m:val="undOvr"/><m:supHide m:val="1"/></m:naryPr><m:sub>{lo}</m:sub><m:sup/><m:e>{e}</m:e></m:nary>'
def acc(ch, e): return f'<m:acc><m:accPr><m:chr m:val="{ch}"/></m:accPr><m:e>{e}</m:e></m:acc>'
def paren(e, b="(", c=")"): return f'<m:d><m:dPr><m:begChr m:val="{b}"/><m:endChr m:val="{c}"/></m:dPr><m:e>{e}</m:e></m:d>'
def hat(e): return acc("̂", e)
def bar(e): return acc("̄", e)
def norm(e): return paren(e, "‖", "‖")
def eq_par(inner, num):
    return (f'<w:p {M}><w:pPr><w:tabs><w:tab w:val="center" w:pos="2520"/><w:tab w:val="right" w:pos="5040"/></w:tabs><w:spacing w:before="40" w:after="40" w:line="240" w:lineRule="auto"/><w:jc w:val="left"/></w:pPr>'
            f'<w:r><w:tab/></w:r><m:oMath>{inner}</m:oMath>'
            f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:color w:val="FF0000"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr><w:tab/><w:t>({num})</w:t></w:r></w:p>')
def limlow(e, lim): return f'<m:limLow><m:e>{e}</m:e><m:lim>{lim}</m:lim></m:limLow>'
