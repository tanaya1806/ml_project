"""Check each DOI in build_refs.R against Crossref (title + year); list arXiv/no-identifier entries. Writes results/ref_check.json"""
import re, json, difflib, urllib.parse, subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build_refs.py")).read()
ns = {}; exec(src.split("def ")[0].split("order")[0] if False else src[:src.index("\nR = {")], ns)
m = re.search(r"\nR = \{.*?\n\}\n", src, re.S); exec("E = r'\\emph{et al.}'\n" + m.group(0), ns); R = ns["R"]
out = {}
for k, v in R.items():
    title = re.search(r"``(.+?),?''", v).group(1).rstrip(",?")
    year = re.search(r"((?:19|20)\d\d)[^0-9]*$", v)
    doi = re.search(r"doi: (10\.\S+?)\.?$", v); ax = re.search(r"arXiv:(\S+?)\.?$", v)
    rec = {"year": year.group(1) if year else None, "title": title, "doi": doi.group(1).replace("\\_","_") if doi else None, "arxiv": ax.group(1) if ax else None}
    if doi:
        try:
            j = json.loads(subprocess.run(["curl","-s","-m","30","-A","paper-check","https://api.crossref.org/works/"+urllib.parse.quote(doi.group(1).replace("\\_","_"))],capture_output=True,text=True).stdout)["message"]
            ct = j["title"][0]; sim = difflib.SequenceMatcher(None, title.lower(), ct.lower()).ratio()
            yr = (j.get("issued") or j.get("published") or {}).get("date-parts", [[None]])[0][0]
            rec.update(crossref_title=ct, title_similarity=round(sim, 2), crossref_year=yr, status="OK" if sim > 0.8 else "CHECK")
        except Exception as e:
            rec.update(status="ERROR", error=str(e)[:80])
    else:
        rec["status"] = "ARXIV-ONLY" if ax else "NO-IDENTIFIER"
    out[k] = rec
json.dump(out, open("results/ref_check.json", "w"), indent=1)
from collections import Counter; print(Counter(r["status"] for r in out.values()))
for k, r in out.items():
    if r["status"] not in ("OK", "ARXIV-ONLY"): print(k, r["status"], r.get("title_similarity"), r["title"][:60], "|", r.get("crossref_title", "")[:60], r.get("crossref_year"))
