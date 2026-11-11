// Builds GRAPH_ANOMALY_Paper.docx (two-column IEEE-style, short, simple language). Run: node build.js
const fs = require("fs");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType, BorderStyle, SectionType, VerticalAlign, ShadingType } = D;

const FONT = "Times New Roman";
const refsData = JSON.parse(fs.readFileSync("refs.json", "utf8"));
const order = []; // citation keys in order of first use
const num = (k) => { if (!refsData[k]) throw new Error("missing ref " + k); let i = order.indexOf(k); if (i < 0) { order.push(k); i = order.length - 1; } return i + 1; };

// mini-markup: <i> <b> <sub> <sup>, citations [[key,key]]
function runs(text, base = {}) {
  const out = []; const st = { i: false, b: false, sub: false, sup: false };
  for (const tok of text.split(/(\[\[.*?\]\]|<\/?(?:i|b|sub|sup)>)/).filter((t) => t !== "")) {
    const m = tok.match(/^<(\/?)(i|b|sub|sup)>$/);
    if (m) { st[m[2]] = m[1] === ""; continue; }
    let t = tok;
    if (/^\[\[.*\]\]$/.test(tok)) t = tok.slice(2, -2).split(",").map((k) => "[" + num(k.trim()) + "]").join(", ");
    out.push(new TextRun({ text: t, font: FONT, size: base.size || 22, italics: st.i || base.italics, bold: st.b || base.bold, subScript: st.sub, superScript: st.sup }));
  }
  return out;
}
const P = (text, o = {}) => new Paragraph({ alignment: o.align || AlignmentType.JUSTIFIED, spacing: { after: o.after ?? 80, before: o.before ?? 0, line: 240 }, indent: o.indent, keepNext: o.keepNext, children: runs(text, o) });
const H1 = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 140, after: 70 }, keepNext: true, children: [new TextRun({ text: t, font: FONT, size: 22, bold: true })] });
const H2 = (t) => new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 70, after: 40 }, keepNext: true, children: [new TextRun({ text: t, font: FONT, size: 22, bold: true, italics: true })] });
const bullet = (t) => P(t, { indent: { left: 200, hanging: 200 }, after: 40 });
const cap = (t) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 40, after: 100 }, children: runs(t, { size: 18 }) });
const tcap = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 40 }, keepNext: true, children: runs(t, { size: 18 }) });

function png(file, widthPx) {
  const b = fs.readFileSync(file); const w = b.readUInt32BE(16), h = b.readUInt32BE(20);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 0 }, keepNext: true, children: [new ImageRun({ type: "png", data: b, transformation: { width: widthPx, height: Math.round((widthPx * h) / w) } })] });
}

const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: none, bottom: none, left: none, right: none };
const line = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
const thin = { style: BorderStyle.SINGLE, size: 2, color: "000000" };

// compact table: header + rows, widths in DXA summing to total
function table(widths, head, rows, size = 17) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, w, hd, last, first) => new TableCell({
    width: { size: w, type: WidthType.DXA }, margins: { top: 25, bottom: 25, left: 50, right: 50 }, verticalAlign: VerticalAlign.CENTER,
    borders: { top: hd && first ? line : none, bottom: hd ? thin : last ? line : none, left: none, right: none },
    children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: runs(t, { size, bold: hd }) })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: head.map((t, i) => cell(t, widths[i], true, false, true)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, widths[i], false, ri === rows.length - 1, false)) }))],
  });
}

// ---------- authors: 3 on top, 2 below centred between 1-2 and 2-3 ----------
const AUTH = [["Dr. Prof. Premanand Ghadekar", "premanand.ghadekar@vit.edu"], ["Harsh Patel", "patel.harsh24@vit.edu"], ["Tanaya Hartalekar", "tanaya.hartalekar24@vit.edu"], ["Hanil Shah", "hanil.shah24@vit.edu"], ["Shravani Dixit", "shravani.dixit24@vit.edu"]];
const CW = 1726; // six columns
const aCell = (a, span) => new TableCell({
  columnSpan: span, width: { size: CW * span, type: WidthType.DXA }, borders: noBorders, margins: { top: 20, bottom: 60, left: 20, right: 20 },
  children: a ? [a[0], "Vishwakarma Institute of Technology,", "Pune, India", a[1]].map((t, i) =>
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: t, font: FONT, size: 16, bold: i === 0 })] })) : [new Paragraph({ children: [] })],
});
const authors = new Table({
  width: { size: CW * 6, type: WidthType.DXA }, columnWidths: Array(6).fill(CW), borders: { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none },
  rows: [
    new TableRow({ children: [aCell(AUTH[0], 2), aCell(AUTH[1], 2), aCell(AUTH[2], 2)] }),
    new TableRow({ children: [aCell(null, 1), aCell(AUTH[3], 2), aCell(AUTH[4], 2), aCell(null, 1)] }),
  ],
});

const TITLE = "Few-Shot Temporal Subgraph Anomaly Detection and Structure-Aware Anomaly Generation in Functional Brain Networks";
const FULLW = 691; // px, full text width (7.2 in)

// ---------- section 1: title + authors ----------
const s1 = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 140 }, children: [new TextRun({ text: TITLE, font: FONT, size: 40, bold: true })] }),
  authors,
  new Paragraph({ spacing: { after: 60 }, children: [] }),
];

// ---------- section 2 (2 columns): abstract, intro, literature ----------
const s2 = [
  P("<b><i>Abstract: Brain imaging (resting-state fMRI) is generally converted into one network per individual. This could potentially obscure the activity of a region of the brain that only operates for a short amount of time, and there are only limited labeled scans. We present a model that monitors the activity of various microregions of the brain network over time. The model only learns what constitutes a normal modification based on control scans. A region that acts in a strange manner earns a comparatively high anomaly score. We also generate artificial anomalies by changing actual control scans and test whether the artificial modifications combined with a few authentic scans from persons with ASD could provide any improvement. In the case of ABIDE I dataset (871 scans, 20 sites) the classifier performed no better than random guessing (AUROC is 0.453). The model could easily differentiate the synthetic anomalies (45.8% against 4.8% for random changes) since it is the same model that produced those changes. Minimal modifications increased F1 but had no effect on AUROC since the changes were flagged more often. An experimental severity measure produced no better outcomes than a guess. The results are presented openly and the pipeline is also available.</i></b>", { after: 80 }),
  P("<b><i>Keywords: </i></b><i>brain networks, anomaly detection, temporal graphs, few-shot learning, autism spectrum disorder, resting-state fMRI</i>", { after: 60 }),

  H1("I. INTRODUCTION"),
  P("Resting-state functional MRI (fMRI) involves scanning the brain while an individual is motionless. The information collected is used to create a brain network where every node corresponds to a region of the brain and every link indicates the amount of movement of the two regions together [[bullmore2009, smith2011]]. Most research on deep learning converts each single scan into one fixed brain network that determines whether there is autism (ASD) or not [[parisot2018, li2021braingnn, cui2023, kan2022]]. The results reported on public data ABIDE show slight differences and very much depend on the location of the scan [[abraham2017, yu2018harm]]."),
  P("It is important to keep in mind that brain networks can vary. The network can be changed during dozens of seconds [[allen2014, hutchison2013]]. So the available networks per scan might mask the regions having abnormal activity for a short period of time. Besides, because the number of scans of patients is very low, the diagnosis can only indicate the disorders in the brain at a very rough level just indicating the unusual activity in the brain."),
  P("The present research corresponds to how to follow these concepts. We have consulted on three main research matters. The RQ1 question: Does modelling a timeline assist relative to just one scanning (random) network? The RQ2 question: Do synthetic and real unusual things support each other? The RQ3 question: Can the scores approximate the severity of autism?"),
  P("Our contribution is established in the following four points: (1) The model that permits keeping track of small brain graphs and finding possible changes; (2) the generator creates synthetic anomalies from real connections and tests them against random benchmarks; (3) the experiments do not leak any answers and are totally honest; (4) the demo application with 20 sample scans. The research does not make any clinical claims."),

  H1("II. LITERATURE REVIEW"),
  P("<b>Brain networks and deep learning.</b> Many models classify whole brain networks. Population-graph models treat each person as a node [[parisot2018]]. BrainGNN [[li2021braingnn]], BrainGB [[cui2023]] and the Brain Network Transformer [[kan2022]] work on each person's network. All are supervised and use one network per scan."),
  P("<b>Changing connectivity.</b> Sliding-window studies show that connectivity changes over time [[allen2014, hutchison2013]]. STAGIN applies a graph network to each window and classifies the whole scan [[kim2021]], and a spatio-temporal graph convolution model does similar work [[gadgil2020]]. They give one decision per scan and do not say which brain part behaved oddly."),
  P("<b>Graph anomaly detection.</b> Graph anomaly detection looks for unusual nodes, edges, subgraphs or graphs [[ma2023survey]]; a classic method scores reconstruction error [[ding2019dominant]]. For brain data, two recent models train only on control scans and flag unusual edges or hyperedges, using synthetic anomalies for testing [[behrouz2023, sadeghian2024]]. Few-shot graph anomaly detection uses a handful of labelled anomalies, but on non-brain graphs [[ding2021fewshot]]."),
  P("<b>Synthetic anomalies.</b> Editing graph edges with gradients is well known in adversarial attacks [[xu2019topology, madry2018]]. We use the same idea to make anomalies, not to attack a model."),
  P("<b>Gap.</b> To our knowledge, after a limited search, no earlier work combines (a) following small brain subgraphs over time with next-step prediction, (b) gradient-guided editing of connection weights to create anomalies and (c) few-shot adaptation for ASD. Each piece exists on its own; our contribution is the combination and an honest test of it."),
];

// ---------- section 3 (1 column): Fig 1 ----------

// ---------- section 4 (2 columns): method + setup ----------
const s4 = [
  H1("III. METHOD"),
  P("Figure 1 shows the pipeline. It has nine steps."),
  png("assets/fig_flow_col.png", 333),
  cap("<b>Fig. 1.</b> The full pipeline in nine steps. Example images: brain-region subgraphs of a control and an ASD scan (input), and the demo application result with a real control subgraph and its synthetic anomaly, edited edges bold (output)."),
  H2("A. Data preparation and patient split"),
  P("We use ABIDE I [[dimartino2014]]: 871 quality-checked scans (403 ASD, 468 control) from 20 sites. Each scan gives the average signal of 360 brain regions from the Glasser atlas [[glasser2016]], extracted with Nilearn [[abraham2014]] after standard preprocessing [[brahim2020]]. We split by person: 609 for training (328 control, 281 ASD), 131 for validation and 131 for testing, keeping the ASD/control ratio. One person never appears in two splits."),
  H2("B. Sliding-window brain graph building"),
  P("We cut each scan into windows of 30 time points, moving 10 points each step. In each window we compute the correlation between every pair of regions and keep the strongest 20% of connections, with their sign. This gives one brain network per window (5 to 29 windows per scan)."),
  H2("C. Brain-region subgraph tracking"),
  P("Using all 360 regions is heavy and does not show where a change happens. We pick 20 regions spread across the brain. For each one we take the region and its direct neighbours in the first window, and follow this fixed set of regions through all later windows. Only the connections and their weights change. A subgraph has about 71 regions on average."),
  H2("D. GIN encoding and GRU prediction"),
  P("A two-layer graph isomorphism network (GIN) [[xu2019gin]] turns each subgraph into a vector of 128 numbers. A GRU [[cho2014]] reads the vectors up to window <i>t</i> and predicts the vector of window <i>t</i>+1. Both are trained together on control scans only. The loss is the prediction error plus a small penalty that stops all vectors from becoming equal [[bardes2022]]. We use PyTorch [[paszke2019]] and PyTorch Geometric [[fey2019]]."),
  H2("E. Temporal subgraph anomaly scoring"),
  P("The anomaly score of a subgraph at step <i>t</i> is the distance between the predicted and the real vector: <i>A</i><sub>t</sub> = ||<i>z</i><sub>t+1</sub> − <i>ẑ</i><sub>t+1</sub>||. A large score means the subgraph changed in a way that normal control scans do not predict. A person's score is the largest <i>A</i><sub>t</sub> over all 20 regions and all steps."),
  H2("F. Control-based threshold calibration"),
  P("The threshold is the 95th percentile of the scores of validation control scans. We use one value for single steps and one for the person score. The step threshold alone flagged all 131 test people, because a person has about 280 steps and 5% of them cross it by chance, so the person-level threshold is used for decisions. We found this after seeing the test result, so precision, recall and F1 are not fully blind to it; AUROC is not affected."),
  H2("G. Synthetic anomaly generation"),
  P("For a real control subgraph we edit existing connection weights. We measure how much each connection changes the model's error, choose the most influential ones (up to 15% of the connections) and move them in 5 small steps in the direction that raises the error [[xu2019topology, madry2018]]. Weights stay between −1 and 1 and no connection is added or removed. Severity (0.25, 0.55, 0.9) sets how many connections change and how far. A random baseline edits the same number of random connections."),
  H2("H. Few-shot adaptation"),
  P("The model is frozen. Each person is described by 20 numbers: the largest score of each tracked region. A logistic regression is trained with 1, 5 or 10 real ASD scans as positives, 30 synthetic anomalies as extra positives, and the 328 training controls as negatives. We repeat this with five random draws of ASD scans and compare three set-ups: real ASD scans only, plus guided synthetic anomalies, plus random synthetic anomalies."),
  H2("I. Detection and severity output"),
  P("A person is flagged if the score is above the threshold. For flagged people we estimate a severity band from the ADOS-2 calibrated severity score [[gotham2009]] (1–10): low (≤5), moderate (6–7) or high (≥8). This grouping is ours, not an official DSM-5 level. Scores exist for 193 of the 403 ASD people. We linked scans to them by repeating the quality filter of the data release; the counts match at all 20 sites, but the order inside each group is assumed. A logistic regression on 40 numbers (largest and mean score per region) is trained on training ASD people only."),

  H1("IV. EXPERIMENTAL SETUP"),
  P("Table I lists the settings. Nothing was tuned: settings were fixed in advance and the checkpoint with the lowest validation loss (epoch 4 of 5) was used. Test data were used only for the final results. We report AUROC and AUPRC, which do not depend on a threshold, and precision, recall and F1 at the threshold. Uncertainty is a 95% bootstrap interval (2,000 resamples) and a permutation test (5,000 shuffles). Few-shot results are the mean and standard deviation over five draws that share one test set. We also report the F1 that a random classifier would get when it flags the same share <i>q</i> of people: F1<sub>chance</sub> = 2π<i>q</i>/(π+<i>q</i>), where π = 0.466 is the share of ASD people in the test set."),
  tcap("<b>TABLE I.</b> Main settings."),
  table([1500, 3500], ["Item", "Setting"], [
    ["Windows", "30 time points, step 10; top 20% edges"],
    ["Tracked regions", "20, one-hop neighbours, fixed set"],
    ["GIN / GRU", "2 layers, hidden 64, output 128 / hidden 128"],
    ["Training", "Adam, rate 0.001, batch 16, 5 epochs, controls only"],
    ["Threshold", "95th percentile, validation controls"],
    ["Generator", "5 steps, ≤15% edges, 3 severities"],
    ["Few-shot", "Logistic regression, 1/5/10 scans + 30 synthetic"],
    ["Hardware", "Apple-silicon laptop (training 300 s)"],
  ]),
];

// ---------- section 5 (1 column): result figures ----------

// ---------- section 6 (2 columns): results, discussion, conclusion, references ----------
const s6 = [
  H1("V. RESULTS"),
  H2("A. Detector"),
  P("On the 131 test people the detector flagged 4 (2 correct): precision 0.500, recall 0.033, F1 0.062 and accuracy 53.4%. AUROC was 0.453 (95% CI 0.353–0.552, permutation p = 0.35) and AUPRC was 0.467, where chance is 0.466. So it does not separate ASD from control scans better than chance (Fig. 2)."),
  png("assets/fig_detector_col.png", 333),
  cap("<b>Fig. 2.</b> Detector results on the 131 test people. (a) Person scores of validation controls, test controls and test ASD people with the threshold (19.81). (b) ROC curve with a 95% bootstrap band; it follows the chance line."),
  H2("B. Does modelling time help? (RQ1)"),
  P("Table II compares three scores on the same frozen encoder: distance to the control average (static), change from the previous window (persistence) and our GRU error. All have AUROC between 0.45 and 0.49 with overlapping intervals, so time modelling gave no measurable benefit. The encoder was trained with the GRU, so the baselines are not fully independent. Whole-graph baselines were not evaluated."),
  tcap("<b>TABLE II.</b> Scoring methods on the test set."),
  table([1050, 1850, 450, 750, 850], ["Score", "AUROC [95% CI]", "p", "F1", "Flagged"], [
    ["Static", "0.452 [0.356, 0.553]", "0.34", "0.143", "9"],
    ["Persistence", "0.485 [0.380, 0.582]", "0.78", "0.062", "4"],
    ["GRU (ours)", "0.453 [0.353, 0.552]", "0.35", "0.062", "4"],
  ]),
  H2("C. Synthetic anomalies (RQ2)"),
  P("We made 7,200 synthetic records (60 control people × 20 regions × 3 severities × 2 methods) with no rule violations. Guided edits raised the mean score from 4.0 to 7.6 as severity grew, while random edits stayed near 2.9. At the highest severity 45.8% of guided records crossed the step threshold, against 4.8% for random edits (Fig. 3); real control and real ASD steps crossed it 5.0% and 4.9% of the time. This is expected by design, because the same model chooses the edits. It shows the generator is not random noise, not that the anomalies look like autism."),
  png("assets/fig_generation_col.png", 333),
  cap("<b>Fig. 3.</b> Synthetic anomalies. (a) Score after editing against severity. (b) Share of steps above the step threshold. (c) Influence of the connections in one example, with the 74 chosen out of 548."),
  H2("D. Few-shot adaptation (RQ2)"),
  P("Table III and Fig. 4 show the few-shot results. With one ASD scan the classifier almost never flags anyone (F1 0.006). Adding 30 synthetic anomalies raises 1-shot F1 to about 0.39–0.40, but AUROC stays near 0.50 in every case. These classifiers flag 35–39% of people, and a random classifier flagging as many would get F1 0.40–0.42. So the F1 gain comes from flagging more people, not from better ranking. Guided and random synthetic anomalies behave the same."),
  png("assets/fig_fewshot_col.png", 333),
  cap("<b>Fig. 4.</b> Few-shot results (mean \u00b1 SD, five draws): (a) F1, (b) recall, (c) AUROC. F1 and recall rise with synthetic anomalies, but AUROC stays near chance (0.5)."),
  tcap("<b>TABLE III.</b> Few-shot results (mean ± SD, five draws). SA = guided synthetic anomalies, R = random."),
  table([600, 1000, 1050, 1050, 500, 750], ["Shots", "Training set", "F1", "AUROC", "Flag", "Chance F1"], [
    ["1", "ASD only", "0.006±0.01", "0.516±0.06", "–", "–"],
    ["", "+ SA", "0.387±0.04", "0.497±0.03", "0.36", "0.41"],
    ["", "+ R", "0.403±0.07", "0.506±0.04", "0.35", "0.40"],
    ["5", "ASD only", "0.149±0.07", "0.485±0.02", "0.10", "0.16"],
    ["", "+ SA", "0.390±0.03", "0.486±0.03", "0.36", "0.41"],
    ["", "+ R", "0.399±0.05", "0.502±0.02", "0.37", "0.41"],
    ["10", "ASD only", "0.311±0.09", "0.524±0.05", "0.26", "0.33"],
    ["", "+ SA", "0.402±0.05", "0.501±0.05", "0.38", "0.42"],
    ["", "+ R", "0.431±0.04", "0.515±0.05", "0.39", "0.42"],
  ], 16),
  H2("E. Severity estimation (RQ3)"),
  P("On 60 held-out ASD people the severity estimate was right 43.3% of the time, while always answering “high” gives 55.0% (Table IV). Cross-validation over all 193 people gave 39.1% against 43.0%. Predicted and true severity scores were not related (Spearman 0.13, p = 0.32), and the detector score was also unrelated to severity (0.05, p = 0.52). So the scores carry no usable severity information here."),
  tcap("<b>TABLE IV.</b> Severity band estimation (three classes)."),
  table([1900, 450, 900, 900, 850], ["Evaluation", "n", "Accuracy", "Macro-F1", "Majority"], [
    ["Held-out ASD", "60", "0.433", "0.395", "0.550"],
    ["5-fold CV ×10", "193", "0.391", "0.372", "0.430"],
  ]),
  H2("F. Demo application"),
  P("We built a local web application. A user uploads a raw time series of 360 regions and gets the detector output and, if flagged, a severity band. On 20 randomly chosen held-out test files (10 control, 10 ASD, shuffled, none chosen by model output) the output was right for 11 and the detector flagged only 1 file. This matches the chance-level test result."),

  H1("VI. DISCUSSION AND LIMITATIONS"),
  P("The pipeline trains in a stable way and the generator works as designed, but the detector does not detect ASD. Possible reasons, none tested here, are that a diagnosis is a weak sign of unusual dynamics, that ASD is very varied, site effects [[yu2018harm]], noisy short windows [[hutchison2013]], and the use of only 20 of 360 regions. Main limits: one training run; two design choices (person-level threshold and the 20-number description) were made after seeing test output; no test across sites; no whole-graph baseline, other encoders or window sweeps; the severity link assumes subject order; and the synthetic anomalies were checked only with the same model. The system must not be used for diagnosis."),

  H1("VII. CONCLUSION"),
  P("We built and tested a pipeline that follows small brain subgraphs over time, scores unexpected changes, creates synthetic anomalies and adapts with a few ASD scans. The generator works as designed, but the detector is at chance on ABIDE I (AUROC 0.453), the few-shot F1 gain comes from flagging more people, and severity cannot be estimated from the scores. Next steps are whole-graph and other-encoder baselines, tests across sites, window and threshold sweeps, several training runs, and checking synthetic anomalies with an independent model."),
];

// ---------- references ----------
const refParas = () => order.map((k, i) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 30, line: 220 }, indent: { left: 300, hanging: 300 },
  children: [new TextRun({ text: "[" + (i + 1) + "]\t", font: FONT, size: 17 }), ...refsData[k].map(([t, it]) => new TextRun({ text: t, font: FONT, size: 17, italics: !!it }))],
  tabStops: [{ type: D.TabStopType.LEFT, position: 300 }] }));

const page = { size: { width: 12240, height: 15840 }, margin: { top: 1000, bottom: 1000, left: 940, right: 940 } };
const two = { count: 2, space: 360 };
// build content first (assigns citation numbers in document order), then references
const body = { s1, s2, s4, s6 };
const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  sections: [
    { properties: { page }, children: body.s1 },
    { properties: { page, type: SectionType.CONTINUOUS, column: two }, children: [...body.s2, ...body.s4, ...body.s6, H1("REFERENCES"), ...refParas()] },
  ],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync("../GRAPH_ANOMALY_Paper.docx", b); console.log("written; refs cited:", order.length); fs.writeFileSync("refs_order_docx.txt", order.map((k,i)=>"["+(i+1)+"] "+k).join("\n")); });
