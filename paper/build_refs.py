"""Builds refs.tex (thebibliography, numbered by first citation in main.tex) and audits cite<->bib consistency."""
import re, sys
E = r"\emph{et al.}"
R = {
"brahim2020": r"A. Brahim and N. Farrugia, ``Graph Fourier transform of fMRI temporal signals based on an averaged structural connectome for the classification of neuroimaging,'' \emph{Artif. Intell. Med.}, vol. 106, Art. no. 101870, 2020, doi: 10.1016/j.artmed.2020.101870.",
"gotham2009": r"K. Gotham, A. Pickles, and C. Lord, ``Standardizing ADOS scores for a measure of severity in autism spectrum disorders,'' \emph{J. Autism Dev. Disord.}, vol. 39, no. 5, pp. 693--705, May 2009, doi: 10.1007/s10803-008-0674-3.",
"abraham2014": r"A. Abraham " + E + r", ``Machine learning for neuroimaging with scikit-learn,'' \emph{Front. Neuroinform.}, vol. 8, Art. no. 14, 2014, doi: 10.3389/fninf.2014.00014.",
"dimartino2014": r"A. Di Martino " + E + r", ``The autism brain imaging data exchange: Towards a large-scale evaluation of the intrinsic brain architecture in autism,'' \emph{Mol. Psychiatry}, vol. 19, no. 6, pp. 659--667, Jun. 2014, doi: 10.1038/mp.2013.78.",
"parisot2018": r"S. Parisot " + E + r", ``Disease prediction using graph convolutional networks: Application to autism spectrum disorder and Alzheimer's disease,'' \emph{Med. Image Anal.}, vol. 48, pp. 117--130, Aug. 2018, doi: 10.1016/j.media.2018.06.001.",
"bullmore2009": r"E. Bullmore and O. Sporns, ``Complex brain networks: Graph theoretical analysis of structural and functional systems,'' \emph{Nat. Rev. Neurosci.}, vol. 10, no. 3, pp. 186--198, Mar. 2009, doi: 10.1038/nrn2575.",
"smith2011": r"S. M. Smith " + E + r", ``Network modelling methods for FMRI,'' \emph{NeuroImage}, vol. 54, no. 2, pp. 875--891, Jan. 2011, doi: 10.1016/j.neuroimage.2010.08.063.",
"ktena2017": r"S. I. Ktena " + E + r", ``Distance metric learning using graph convolutional networks: Application to functional brain networks,'' in \emph{Med. Image Comput. Comput.-Assist. Intervent. (MICCAI)}, 2017, pp. 469--477, doi: 10.1007/978-3-319-66182-7\_54.",
"li2021braingnn": r"X. Li " + E + r", ``BrainGNN: Interpretable brain graph neural network for fMRI analysis,'' \emph{Med. Image Anal.}, vol. 74, Art. no. 102233, Dec. 2021, doi: 10.1016/j.media.2021.102233.",
"xu2019gin": r"K. Xu, W. Hu, J. Leskovec, and S. Jegelka, ``How powerful are graph neural networks?'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2019, arXiv:1810.00826.",
"hutchison2013": r"R. M. Hutchison " + E + r", ``Dynamic functional connectivity: Promise, issues, and interpretations,'' \emph{NeuroImage}, vol. 80, pp. 360--378, Oct. 2013, doi: 10.1016/j.neuroimage.2013.05.079.",
"cho2014": r"K. Cho " + E + r", ``Learning phrase representations using RNN encoder-decoder for statistical machine translation,'' in \emph{Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)}, Doha, Qatar, Oct. 2014, pp. 1724--1734, doi: 10.3115/v1/D14-1179.",
"rossi2020tgn": r"E. Rossi, B. Chamberlain, F. Frasca, D. Eynard, F. Monti, and M. Bronstein, ``Temporal graph networks for deep learning on dynamic graphs,'' arXiv:2006.10637, Jun. 2020.",
"pareja2020": r"A. Pareja " + E + r", ``EvolveGCN: Evolving graph convolutional networks for dynamic graphs,'' in \emph{Proc. AAAI Conf. Artif. Intell.}, vol. 34, no. 4, 2020, pp. 5363--5370, doi: 10.1609/aaai.v34i04.5984.",
"ma2023survey": r"X. Ma " + E + r", ``A comprehensive survey on graph anomaly detection with deep learning,'' \emph{IEEE Trans. Knowl. Data Eng.}, vol. 35, no. 12, pp. 12012--12038, Dec. 2023, doi: 10.1109/TKDE.2021.3118815.",
"ding2019dominant": r"K. Ding, J. Li, R. Bhanushali, and H. Liu, ``Deep anomaly detection on attributed networks,'' in \emph{Proc. SIAM Int. Conf. Data Mining (SDM)}, 2019, pp. 594--602, doi: 10.1137/1.9781611975673.67.",
"ding2021fewshot": r"K. Ding, Q. Zhou, H. Tong, and H. Liu, ``Few-shot network anomaly detection via cross-network meta-learning,'' in \emph{Proc. Web Conf. (WWW)}, 2021, pp. 2448--2456, doi: 10.1145/3442381.3449922.",
"zugner2018": r"D. Z{\"u}gner, A. Akbarnejad, and S. G{\"u}nnemann, ``Adversarial attacks on neural networks for graph data,'' in \emph{Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining (KDD)}, 2018, pp. 2847--2856, doi: 10.1145/3219819.3220078.",
"dai2018": r"H. Dai " + E + r", ``Adversarial attack on graph structured data,'' in \emph{Proc. Int. Conf. Mach. Learn. (ICML)}, PMLR vol. 80, 2018, pp. 1115--1124.",
"xu2019topology": r"K. Xu " + E + r", ``Topology attack and defense for graph neural networks: An optimization perspective,'' in \emph{Proc. Int. Joint Conf. Artif. Intell. (IJCAI)}, 2019, pp. 3961--3967, doi: 10.24963/ijcai.2019/550.",
"madry2018": r"A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, ``Towards deep learning models resistant to adversarial attacks,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2018, arXiv:1706.06083.",
"zhao2021smote": r"T. Zhao, X. Zhang, and S. Wang, ``GraphSMOTE: Imbalanced node classification on graphs with graph neural networks,'' in \emph{Proc. ACM Int. Conf. Web Search Data Mining (WSDM)}, 2021, pp. 833--841, doi: 10.1145/3437963.3441720.",
"chen2020gaan": r"Z. Chen, B. Liu, M. Wang, P. Dai, J. Lv, and L. Bo, ``Generative adversarial attributed network anomaly detection,'' in \emph{Proc. ACM Int. Conf. Inf. Knowl. Manage. (CIKM)}, 2020, pp. 1989--1992, doi: 10.1145/3340531.3412070.",
"allen2014": r"E. A. Allen, E. Damaraju, S. M. Plis, E. B. Erhardt, T. Eichele, and V. D. Calhoun, ``Tracking whole-brain connectivity dynamics in the resting state,'' \emph{Cereb. Cortex}, vol. 24, no. 3, pp. 663--676, Mar. 2014, doi: 10.1093/cercor/bhs352.",
"preti2017": r"M. G. Preti, T. A. Bolton, and D. Van De Ville, ``The dynamic functional connectome: State-of-the-art and perspectives,'' \emph{NeuroImage}, vol. 160, pp. 41--54, Oct. 2017, doi: 10.1016/j.neuroimage.2016.12.061.",
"abraham2017": r"A. Abraham " + E + r", ``Deriving reproducible biomarkers from multi-site resting-state data: An autism-based example,'' \emph{NeuroImage}, vol. 147, pp. 736--745, Feb. 2017, doi: 10.1016/j.neuroimage.2016.10.045.",
"heinsfeld2018": r"A. S. Heinsfeld, A. R. Franco, R. C. Craddock, A. Buchweitz, and F. Meneguzzi, ``Identification of autism spectrum disorder using deep learning and the ABIDE dataset,'' \emph{NeuroImage Clin.}, vol. 17, pp. 16--23, 2018, doi: 10.1016/j.nicl.2017.08.017.",
"kawahara2017": r"J. Kawahara " + E + r", ``BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment,'' \emph{NeuroImage}, vol. 146, pp. 1038--1049, Feb. 2017, doi: 10.1016/j.neuroimage.2016.09.046.",
"kan2022": r"X. Kan, W. Dai, H. Cui, Z. Zhang, Y. Guo, and C. Yang, ``Brain network transformer,'' in \emph{Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol. 35, 2022.",
"cui2023": r"H. Cui " + E + r", ``BrainGB: A benchmark for brain network analysis with graph neural networks,'' \emph{IEEE Trans. Med. Imag.}, vol. 42, no. 2, pp. 493--506, Feb. 2023, doi: 10.1109/TMI.2022.3218745.",
"ruff2018": r"L. Ruff " + E + r", ``Deep one-class classification,'' in \emph{Proc. Int. Conf. Mach. Learn. (ICML)}, PMLR vol. 80, 2018, pp. 4393--4402.",
"pang2021review": r"G. Pang, C. Shen, L. Cao, and A. van den Hengel, ``Deep learning for anomaly detection: A review,'' \emph{ACM Comput. Surv.}, vol. 54, no. 2, 2022, doi: 10.1145/3439950.",
"yu2018netwalk": r"W. Yu, W. Cheng, C. C. Aggarwal, K. Zhang, H. Chen, and W. Wang, ``NetWalk: A flexible deep embedding approach for anomaly detection in dynamic networks,'' in \emph{Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining (KDD)}, 2018, pp. 2672--2681, doi: 10.1145/3219819.3220024.",
"zheng2019addgraph": r"L. Zheng, Z. Li, J. Li, Z. Li, and J. Gao, ``AddGraph: Anomaly detection in dynamic graph using attention-based temporal GCN,'' in \emph{Proc. Int. Joint Conf. Artif. Intell. (IJCAI)}, 2019, pp. 4419--4425, doi: 10.24963/ijcai.2019/614.",
"bardes2022": r"A. Bardes, J. Ponce, and Y. LeCun, ``VICReg: Variance-invariance-covariance regularization for self-supervised learning,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2022, arXiv:2105.04906.",
"kingma2015": r"D. P. Kingma and J. Ba, ``Adam: A method for stochastic optimization,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2015, arXiv:1412.6980.",
"finn2017": r"C. Finn, P. Abbeel, and S. Levine, ``Model-agnostic meta-learning for fast adaptation of deep networks,'' in \emph{Proc. Int. Conf. Mach. Learn. (ICML)}, PMLR vol. 70, 2017, pp. 1126--1135.",
"wang2021fewshot": r"Y. Wang, Q. Yao, J. T. Kwok, and L. M. Ni, ``Generalizing from a few examples: A survey on few-shot learning,'' \emph{ACM Comput. Surv.}, vol. 53, no. 3, Art. no. 63, 2020, doi: 10.1145/3386252.",
"yu2018harm": r"M. Yu " + E + r", ``Statistical harmonization corrects site effects in functional connectivity measurements from multi-site fMRI data,'' \emph{Hum. Brain Mapp.}, vol. 39, no. 11, pp. 4213--4227, Nov. 2018, doi: 10.1002/hbm.24241.",
"pang2019devnet": r"G. Pang, C. Shen, and A. van den Hengel, ``Deep anomaly detection with deviation networks,'' in \emph{Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining (KDD)}, 2019, pp. 353--362, doi: 10.1145/3292500.3330871.",
"lanciano2020": r"T. Lanciano, F. Bonchi, and A. Gionis, ``Explainable classification of brain networks via contrast subgraphs,'' in \emph{Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining (KDD)}, 2020, pp. 3308--3318, doi: 10.1145/3394486.3403383.",
"kim2021": r"B.-H. Kim, J. C. Ye, and J.-J. Kim, ``Learning dynamic graph representation of brain connectome with spatio-temporal attention,'' in \emph{Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol. 34, 2021, arXiv:2105.13495.",
"gadgil2020": r"S. Gadgil, Q. Zhao, A. Pfefferbaum, E. V. Sullivan, E. Adeli, and K. M. Pohl, ``Spatio-temporal graph convolution for resting-state fMRI analysis,'' in \emph{Med. Image Comput. Comput.-Assist. Intervent. (MICCAI)}, 2020, pp. 528--538, doi: 10.1007/978-3-030-59728-3\_52.",
"marquand2016": r"A. F. Marquand, I. Rezek, J. Buitelaar, and C. F. Beckmann, ``Understanding heterogeneity in clinical cohorts using normative models: Beyond case-control studies,'' \emph{Biol. Psychiatry}, vol. 80, no. 7, pp. 552--561, Oct. 2016, doi: 10.1016/j.biopsych.2015.12.023.",
"ma2022glocal": r"R. Ma, G. Pang, L. Chen, and A. van den Hengel, ``Deep graph-level anomaly detection by glocal knowledge distillation,'' in \emph{Proc. ACM Int. Conf. Web Search Data Mining (WSDM)}, 2022, pp. 704--714, doi: 10.1145/3488560.3498473.",
"kipf2017": r"T. N. Kipf and M. Welling, ``Semi-supervised classification with graph convolutional networks,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2017, arXiv:1609.02907.",
"snell2017": r"J. Snell, K. Swersky, and R. Zemel, ``Prototypical networks for few-shot learning,'' in \emph{Adv. Neural Inf. Process. Syst. (NIPS)}, vol. 30, 2017, arXiv:1703.05175.",
"glasser2016": r"M. F. Glasser " + E + r", ``A multi-modal parcellation of human cerebral cortex,'' \emph{Nature}, vol. 536, no. 7615, pp. 171--178, Aug. 2016, doi: 10.1038/nature18933.",
"fey2019": r"M. Fey and J. E. Lenssen, ``Fast graph representation learning with PyTorch Geometric,'' in \emph{ICLR Workshop Representation Learn. Graphs Manifolds}, 2019, arXiv:1903.02428.",
"paszke2019": r"A. Paszke " + E + r", ``PyTorch: An imperative style, high-performance deep learning library,'' in \emph{Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol. 32, 2019, arXiv:1912.01703.",
"sadeghian2024": r"S. Sadeghian, X. Li, and M. Seltzer, ``HyperBrain: Anomaly detection for temporal hypergraph brain networks,'' arXiv:2410.02087, Oct. 2024.",
"behrouz2023": r"A. Behrouz and M. Seltzer, ``Anomaly detection in human brain via inductive learning on temporal multiplex networks,'' in \emph{Proc. Mach. Learn. Healthcare Conf.}, PMLR vol. 219, 2023, pp. 50--75.",
"chatterjee2021": r"T. Chatterjee, R. Albert, S. Thapliyal, N. Azarhooshang, and B. DasGupta, ``Detecting network anomalies using Forman-Ricci curvature and a case study for human brain networks,'' \emph{Sci. Rep.}, vol. 11, no. 1, Art. no. 8121, Apr. 2021, doi: 10.1038/s41598-021-87587-z.",
"aggarwal2019": r"P. Aggarwal and A. Gupta, ``Multivariate graph learning for detecting aberrant connectivity of dynamic brain networks in autism,'' \emph{Med. Image Anal.}, vol. 56, pp. 11--25, Aug. 2019, doi: 10.1016/j.media.2019.05.007.",
"yang2022meta": r"Y. Yang " + E + r", ``Data-efficient brain connectome analysis via multi-task meta-learning,'' in \emph{Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining (KDD)}, 2022, pp. 4743--4751, doi: 10.1145/3534678.3542680.",
"wei2025": r"X. Wei, K. Zhao, Y. Jiao, L. He, and Y. Zhang, ``A brain graph foundation model: Pre-training and prompt-tuning across broad atlases and disorders,'' arXiv:2506.02044, 2025.",
"dhinagar2023": r"N. J. Dhinagar, V. Santhalingam, K. E. Lawrence, E. Laltoo, and P. M. Thompson, ``Few-shot classification of autism spectrum disorder using site-agnostic meta-learning and brain MRI,'' in \emph{Proc. Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC)}, 2023, doi: 10.1109/EMBC40787.2023.10340852.",
}
tex = open("main.tex").read()
body = tex.split(r"\begin{document}")[1]
order = []
for m in re.finditer(r"\\cite\{([^}]*)\}", body):
    for k in m.group(1).split(","):
        k = k.strip()
        if k not in order: order.append(k)
missing = [k for k in order if k not in R]; unused = [k for k in R if k not in order]
print(f"cited={len(order)} bib={len(R)} missing_in_bib={missing} uncited_bib_entries={unused}")
if missing: sys.exit(1)
with open("refs.tex", "w") as f:
    f.write("\\begin{thebibliography}{%d}\n" % len(order))
    for k in order: f.write("\\bibitem{%s}\n%s\n\n" % (k, R[k]))
    f.write("\\end{thebibliography}\n")
open("refs_order.txt", "w").write("\n".join(f"[{i+1}] {k}" for i, k in enumerate(order)))
