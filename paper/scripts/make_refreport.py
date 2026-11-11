"""Writes REFERENCE_VERIFICATION.md from results/ref_check.json (run scripts/verify_refs.py first)."""
import json, datetime
from collections import Counter
d = json.load(open('results/ref_check.json'))
order = [l.split()[1] for l in open('refs_order.txt') if l.strip()]
num = {k: i + 1 for i, k in enumerate(order)}
note = {"yu2018netwalk": "Crossref title is the short title (\"NetWalk\"); full title/venue/pages as in the KDD'18 record.",
        "wang2021fewshot": "Corrected: Crossref gives 2020 (ACM Comput. Surv. 53(3), Art. 63); an earlier draft said 2021. Citation key keeps '2021'.",
        "chen2020gaan": "Crossref year 2020 matches (AAAI-20)."}
c = Counter(r['status'] for r in d.values())
L = ["# Reference verification report\n",
     f"Generated {datetime.date.today().isoformat()} by `scripts/verify_refs.py` (Crossref lookup by DOI: title similarity and year) plus an arXiv-API lookup by ID for arXiv-only entries.\n",
     "**Scope and honesty statement.** A pass means the DOI/arXiv ID resolves to a record whose title (and year) match the bibliography entry. It does *not* verify page ranges, authors beyond the first, or that the paper supports the sentence citing it; the last was judged while writing the related-work text and should be re-read by the authors. Entries without a DOI or arXiv ID (conference proceedings) were checked against publisher/proceedings pages in an earlier session and were *not* machine-re-verified in this pass; they are marked.\n",
     "## Summary\n",
     f"- Entries: {len(d)}; all cited in the text; every cited key exists in the bibliography (`build_refs.py` audit: cited=57, bib=57, missing=0, uncited=0).",
     f"- DOI resolves with matching title and year: {c['OK']}; DOI resolves but Crossref title is truncated: {c['CHECK']}.",
     "- arXiv ID verified by API (title and first author match): 9 (rossi2020tgn, sadeghian2024, wei2025, xu2019gin, madry2018, bardes2022, kingma2015, paszke2019, fey2019).",
     "- Venue-only (no DOI/arXiv ID; not machine-re-verified this pass): dai2018, kan2022, ruff2018, finn2017, behrouz2023.\n",
     "## Corrections made during verification\n",
     "- wang2021fewshot: year 2021 -> 2020 and article number 63 added.",
     "- DOIs in the .tex source contain escaped underscores (`\\_`), which print as `_`; no content change.\n",
     "## Entry table\n", "| # | Key | Status | Identifier | Notes |", "|---|---|---|---|---|"]
for k in order:
    r = d[k]; ident = r['doi'] or ('arXiv:' + r['arxiv'] if r['arxiv'] else '-')
    st = {'OK': 'DOI verified (title, year)', 'CHECK': 'DOI resolves; Crossref title truncated', 'ARXIV-ONLY': 'arXiv verified', 'NO-IDENTIFIER': 'venue-only, not re-verified'}[r['status']]
    L.append(f"| [{num[k]}] | {k} | {st} | {ident} | {note.get(k, '')} |")
L += ["\n## Research-gap check\n",
      "Targeted searches in an earlier session (not exhaustive) found brain anomaly detection with temporal multiplex networks [behrouz2023] and temporal hypergraphs [sadeghian2024] (control-only, edge/hyperedge level, injected anomalies); few-shot graph anomaly detection on non-brain graphs [ding2021fewshot]; and gradient-based structure perturbation for attacks [xu2019topology, zugner2018, dai2018]. None combined tracked seed-centred subgraph next-embedding prediction, gradient-guided weight-edit anomaly generation and few-shot adaptation for ASD. The paper therefore states the gap as 'to our knowledge, after a targeted search', never as 'no one has done this'. The search did not include IEEE Xplore/Scopus full-text search; authors should repeat it before submission.\n",
      "## Caveats for authors\n",
      "- Table I characterisations are our reading of abstracts/method sections; re-check against full texts.",
      "- Two preprints (sadeghian2024, wei2025) are cited as preprints; check for published versions before submission.",
      "- The bibliography is IEEE-style but hand-formatted in `build_refs.py`; run it through the target venue's style checks."]
open('REFERENCE_VERIFICATION.md', 'w').write("\n".join(L) + "\n")
