# Reference verification report

Generated 2026-10-17 by `scripts/verify_refs.py` (Crossref lookup by DOI: title similarity and year) plus an arXiv-API lookup by ID for arXiv-only entries.

**Scope and honesty statement.** A pass means the DOI/arXiv ID resolves to a record whose title (and year) match the bibliography entry. It does *not* verify page ranges, authors beyond the first, or that the paper supports the sentence citing it; the last was judged while writing the related-work text and should be re-read by the authors. Entries without a DOI or arXiv ID (conference proceedings) were checked against publisher/proceedings pages in an earlier session and were *not* machine-re-verified in this pass; they are marked.

## Summary

- Entries: 57; all cited in the text; every cited key exists in the bibliography (`build_refs.py` audit: cited=57, bib=57, missing=0, uncited=0).
- DOI resolves with matching title and year: 38; DOI resolves but Crossref title is truncated: 2.
- arXiv ID verified by API (title and first author match): 9 (rossi2020tgn, sadeghian2024, wei2025, xu2019gin, madry2018, bardes2022, kingma2015, paszke2019, fey2019).
- Venue-only (no DOI/arXiv ID; not machine-re-verified this pass): dai2018, kan2022, ruff2018, finn2017, behrouz2023.

## Corrections made during verification

- wang2021fewshot: year 2021 -> 2020 and article number 63 added.
- DOIs in the .tex source contain escaped underscores (`\_`), which print as `_`; no content change.

## Entry table

| # | Key | Status | Identifier | Notes |
|---|---|---|---|---|
| [1] | bullmore2009 | DOI verified (title, year) | 10.1038/nrn2575 |  |
| [2] | smith2011 | DOI verified (title, year) | 10.1016/j.neuroimage.2010.08.063 |  |
| [3] | dimartino2014 | DOI verified (title, year) | 10.1038/mp.2013.78 |  |
| [4] | parisot2018 | DOI verified (title, year) | 10.1016/j.media.2018.06.001 |  |
| [5] | li2021braingnn | DOI verified (title, year) | 10.1016/j.media.2021.102233 |  |
| [6] | kan2022 | venue-only, not re-verified | - |  |
| [7] | cui2023 | DOI verified (title, year) | 10.1109/TMI.2022.3218745 |  |
| [8] | abraham2017 | DOI verified (title, year) | 10.1016/j.neuroimage.2016.10.045 |  |
| [9] | heinsfeld2018 | DOI verified (title, year) | 10.1016/j.nicl.2017.08.017 |  |
| [10] | yu2018harm | DOI verified (title, year) | 10.1002/hbm.24241 |  |
| [11] | hutchison2013 | DOI verified (title, year) | 10.1016/j.neuroimage.2013.05.079 |  |
| [12] | allen2014 | DOI verified (title, year) | 10.1093/cercor/bhs352 |  |
| [13] | preti2017 | DOI verified (title, year) | 10.1016/j.neuroimage.2016.12.061 |  |
| [14] | kim2021 | arXiv verified | arXiv:2105.13495 |  |
| [15] | gadgil2020 | DOI verified (title, year) | 10.1007/978-3-030-59728-3_52 |  |
| [16] | marquand2016 | DOI verified (title, year) | 10.1016/j.biopsych.2015.12.023 |  |
| [17] | ma2023survey | DOI verified (title, year) | 10.1109/TKDE.2021.3118815 |  |
| [18] | gotham2009 | DOI verified (title, year) | 10.1007/s10803-008-0674-3 |  |
| [19] | xu2019gin | arXiv verified | arXiv:1810.00826 |  |
| [20] | cho2014 | DOI verified (title, year) | 10.3115/v1/D14-1179 |  |
| [21] | glasser2016 | DOI verified (title, year) | 10.1038/nature18933 |  |
| [22] | ktena2017 | DOI verified (title, year) | 10.1007/978-3-319-66182-7_54 |  |
| [23] | kawahara2017 | DOI verified (title, year) | 10.1016/j.neuroimage.2016.09.046 |  |
| [24] | kipf2017 | arXiv verified | arXiv:1609.02907 |  |
| [25] | lanciano2020 | DOI verified (title, year) | 10.1145/3394486.3403383 |  |
| [26] | rossi2020tgn | venue-only, not re-verified | - |  |
| [27] | pareja2020 | DOI verified (title, year) | 10.1609/aaai.v34i04.5984 |  |
| [28] | aggarwal2019 | DOI verified (title, year) | 10.1016/j.media.2019.05.007 |  |
| [29] | pang2021review | DOI verified (title, year) | 10.1145/3439950 |  |
| [30] | ding2019dominant | DOI verified (title, year) | 10.1137/1.9781611975673.67 |  |
| [31] | ruff2018 | venue-only, not re-verified | - |  |
| [32] | ma2022glocal | DOI verified (title, year) | 10.1145/3488560.3498473 |  |
| [33] | yu2018netwalk | DOI resolves; Crossref title truncated | 10.1145/3219819.3220024 | Crossref title is the short title ("NetWalk"); full title/venue/pages as in the KDD'18 record. |
| [34] | zheng2019addgraph | DOI verified (title, year) | 10.24963/ijcai.2019/614 |  |
| [35] | behrouz2023 | venue-only, not re-verified | - |  |
| [36] | sadeghian2024 | venue-only, not re-verified | - |  |
| [37] | chatterjee2021 | DOI verified (title, year) | 10.1038/s41598-021-87587-z |  |
| [38] | wang2021fewshot | DOI resolves; Crossref title truncated | 10.1145/3386252 | Corrected: Crossref gives 2020 (ACM Comput. Surv. 53(3), Art. 63); an earlier draft said 2021. Citation key keeps '2021'. |
| [39] | finn2017 | venue-only, not re-verified | - |  |
| [40] | snell2017 | arXiv verified | arXiv:1703.05175 |  |
| [41] | ding2021fewshot | DOI verified (title, year) | 10.1145/3442381.3449922 |  |
| [42] | pang2019devnet | DOI verified (title, year) | 10.1145/3292500.3330871 |  |
| [43] | yang2022meta | DOI verified (title, year) | 10.1145/3534678.3542680 |  |
| [44] | wei2025 | venue-only, not re-verified | - |  |
| [45] | dhinagar2023 | DOI verified (title, year) | 10.1109/EMBC40787.2023.10340852 |  |
| [46] | zhao2021smote | DOI verified (title, year) | 10.1145/3437963.3441720 |  |
| [47] | chen2020gaan | DOI verified (title, year) | 10.1145/3340531.3412070 | Crossref year 2020 matches (AAAI-20). |
| [48] | zugner2018 | DOI verified (title, year) | 10.1145/3219819.3220078 |  |
| [49] | dai2018 | venue-only, not re-verified | - |  |
| [50] | xu2019topology | DOI verified (title, year) | 10.24963/ijcai.2019/550 |  |
| [51] | madry2018 | arXiv verified | arXiv:1706.06083 |  |
| [52] | brahim2020 | DOI verified (title, year) | 10.1016/j.artmed.2020.101870 |  |
| [53] | abraham2014 | DOI verified (title, year) | 10.3389/fninf.2014.00014 |  |
| [54] | bardes2022 | arXiv verified | arXiv:2105.04906 |  |
| [55] | paszke2019 | arXiv verified | arXiv:1912.01703 |  |
| [56] | fey2019 | arXiv verified | arXiv:1903.02428 |  |
| [57] | kingma2015 | arXiv verified | arXiv:1412.6980 |  |

## Research-gap check

Targeted searches in an earlier session (not exhaustive) found brain anomaly detection with temporal multiplex networks [behrouz2023] and temporal hypergraphs [sadeghian2024] (control-only, edge/hyperedge level, injected anomalies); few-shot graph anomaly detection on non-brain graphs [ding2021fewshot]; and gradient-based structure perturbation for attacks [xu2019topology, zugner2018, dai2018]. None combined tracked seed-centred subgraph next-embedding prediction, gradient-guided weight-edit anomaly generation and few-shot adaptation for ASD. The paper therefore states the gap as 'to our knowledge, after a targeted search', never as 'no one has done this'. The search did not include IEEE Xplore/Scopus full-text search; authors should repeat it before submission.

## Caveats for authors

- Table I characterisations are our reading of abstracts/method sections; re-check against full texts.
- Two preprints (sadeghian2024, wei2025) are cited as preprints; check for published versions before submission.
- The bibliography is IEEE-style but hand-formatted in `build_refs.py`; run it through the target venue's style checks.
