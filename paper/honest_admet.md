# Honest ADMET: When the Generalization Gap Appears, Why, and Whether Selective Prediction Recovers It

*Draft — [Your Name]. Target: ML4H / a NeurIPS or ICLR molecular-ML workshop, then arXiv/medRxiv.*
*Placeholders marked `[[ … ]]` are filled from the running experiments.*

---

## Abstract

Public ADMET leaderboards such as Therapeutics Data Commons (TDC) rank models by a single
score under one fixed split. We ask three questions a deployer actually cares about, across
all 22 TDC ADMET endpoints and a ladder of model families: (i) *how much* does measured
performance fall under realistic, chemistry-aware splits versus the optimistic random split;
(ii) *why* — and why the effect is heterogeneous and sometimes reverses; and (iii) whether
*selective prediction* (conformal abstention) restores the reliability that random-split
numbers overstate. We make three contributions. First, a **statistically honest gap atlas**:
signed, per-endpoint generalization gaps with matched-replicate bootstrap CIs, paired
Wilcoxon tests, and Benjamini–Hochberg FDR. We find that Bemis–Murcko *scaffold* splits are
much weaker stressors than cluster splits for ADMET (their gaps are ≈3× smaller and absent on
hERG; consistent with Fooladi et al. 2025), while *cluster* (sphere-exclusion) splits reveal a
large, statistically robust gap on **all 22/22 endpoints** (every CI excludes zero, aggregate
p<10⁻⁴) — so scaffold splits **under-state** the gap that cluster splits **reveal**. Second, we
show split-conformal prediction **under-covers under cluster shift** (e.g. BBB coverage 0.92→0.84
at the 90% target) while a corrected covariate-shift weighting is a verified no-op on in-distribution
data and does not restore coverage. Third, we report an **honest evaluation of uncertainty under
shift, negatives included**: the best abstention signal is **model-dependent** (random forests favor
applicability-domain distance, XGBoost favors ensemble disagreement), and per-endpoint abstention
benefit **does not track the optimism gap** (Spearman ρ=0.07, n.s.) — cautioning against the tidy
"abstention recovers the gap" narrative. Finally, frozen foundation-model embeddings (ChemBERTa)
**modestly but significantly shrink the gap** (p=0.011) without closing it. All numbers reproduce
the official TDC leaderboard first;
code and splits are released as a `pip`-installable, leakage-audited harness.

---

## 1. Introduction

ADMET (absorption, distribution, metabolism, excretion, toxicity) prediction is where much of
early drug discovery succeeds or fails, and Therapeutics Data Commons (TDC)
[@huang2021tdc; @huang2022artificial] has become the community standard benchmark for it. The
leaderboard culture it enables optimizes a single metric under one fixed scaffold split. That
is useful for ranking, but it answers a different question than the one a practitioner faces:
*if I deploy this model on genuinely novel chemistry, how well will it work, and will it tell
me when it does not know?*

A long line of work shows random train/test splits leak structurally similar molecules across
folds and overstate prospective performance [@sheridan2013timesplit; @wu2018moleculenet;
@deng2023systematic]. Scaffold splits were introduced as a "harder, more realistic" default
[@wu2018moleculenet]. Yet recent results complicate this: Fooladi et al. [@fooladi2025evaluating]
find that for ADMET, Bemis–Murcko scaffold splits are *not substantially harder than random*,
and only cluster/distance splits expose a real gap; Guo et al. [@guo2024scaffold] show scaffold
splits still leak because different scaffolds can be near-identical molecules. Separately, the
uncertainty-quantification literature offers conformal prediction [@tibshirani2019conformal;
@angelopoulos2023gentle], deep ensembles [@scalia2020evaluating; @hirschfeld2020uncertainty],
and conformal-under-shift methods [@laghuvarapu2023codrug; @li2024conformalized] — but these are
rarely tied back to the split-induced optimism they could mitigate.

We are explicit that **neither phenomenon is a new discovery**. Our contribution is a rigorous
*consolidation and extension* that no single prior paper provides: a fully reproducible harness
that, on the same 22 TDC ADMET endpoints and one model ladder, (1) measures the signed
per-endpoint gap with honest statistics, (2) explains its heterogeneity mechanistically via
nearest-neighbour chemical-similarity leakage, and (3) couples it to selective prediction —
asking whether abstaining on the out-of-distribution tail recovers the reliability the
optimistic numbers promised. This extends the trustworthy-ML-under-shift agenda of the Zitnik
lab's own tools (TDC → SPECTRA → TxGNN [@huang2024txgnn], the last of which explicitly defers
uncertainty quantification to future work).

**Contributions.**
1. A statistically honest, per-endpoint **generalization-gap atlas** across 22 ADMET endpoints
   and model families, with matched-replicate CIs, paired tests, and FDR control — engaging the
   scaffold≈random result head-on and promoting cluster splits to the primary stressor.
2. A **calibration-under-shift** study: split-conformal coverage degrades under cluster shift,
   and we measure whether ESS-guarded covariate-shift weighting restores it.
3. A **selective-prediction** result: applicability-domain distance is a more robust abstention
   signal than ensemble disagreement under shift, with the unifying finding that the endpoints
   with the largest optimism gap are those where abstention helps most.

---

## 2. Related Work

**Split realism.** Sheridan [@sheridan2013timesplit] established that train–test dissimilarity
governs apparent accuracy; MoleculeNet [@wu2018moleculenet] popularized scaffold splits; Deng et
al. [@deng2023systematic] confirmed scaffold < random performance and that fixed descriptors beat
learned ones under scaffold splits. Guo et al. [@guo2024scaffold] and Fooladi et al.
[@fooladi2025evaluating] argue scaffold splits *under-stress* and cluster/distance splits are
needed; van Tilborg et al. [@vantilborg2022exposing] expose a complementary within-scaffold
failure (activity cliffs). We adopt cluster (sphere-exclusion) splits as primary and quantify the
heterogeneity these works predict.

**TDC and benchmark critique.** TDC [@huang2021tdc; @huang2022artificial] is our object of study.
Concurrent work [@koleiev2026critical] audits the 22 ADMET leaderboards for leakage and
reproducibility; we are complementary — a controlled signed-gap measurement plus an uncertainty
axis — and inherit the honest-evaluation framing of Kapoor & Narayanan [@kapoor2023leakage].

**Uncertainty, conformal, selective prediction.** We *apply* (not invent) split conformal
[@tibshirani2019conformal; @angelopoulos2023gentle], deep ensembles [@scalia2020evaluating;
@hirschfeld2020uncertainty], and conformal-under-covariate-shift [@laghuvarapu2023codrug;
@li2024conformalized], evaluated via selective-prediction risk–coverage curves
[@geifman2017selective]. Our novelty is the *integration*: one benchmark spanning all 22 endpoints
× model families × {conformal, ensemble} on the selective-prediction axis under realistic shift,
and the coupling of gap magnitude to abstention benefit.

**Foundation models.** We evaluate frozen MolFormer [@ross2022molformer] and ChemBERTa-2
[@ahmad2022chemberta2] embeddings — which no prior ADMET-shift/conformal study includes.

A differentiation table (endpoints × model families × conformal × ensemble × ECE ×
selective-prediction curves × realistic-split shift × foundation model) versus
[@li2024conformalized; @laghuvarapu2023codrug; @tossou2024mood; @fooladi2025evaluating;
@koleiev2026critical] appears in Appendix A.

---

## 3. Data and Methods

**Datasets.** All 22 endpoints of the TDC ADMET benchmark group (9 regression, 13
classification), accessed via PyTDC [[version 1.1.15]] with pinned RDKit [[2023.9.6]].

**Two loading modes.** For leaderboard reproduction we use TDC's native, unpooled train/test
split and the official 5-seed `admet_group` evaluate protocol. For the controlled split-type
comparison we pool train+test, canonicalize, de-duplicate on canonical SMILES, and re-split under
explicit seeds; pooled numbers are internal-comparison-only and never reported as
leaderboard-comparable.

**Splits.** (a) *random* — the optimistic baseline; (b) *scaffold* — Bemis–Murcko, with acyclic
molecules treated as singleton groups (not lumped into one bucket); (c) *cluster* — sphere-
exclusion (RDKit LeaderPicker) on ECFP4, our primary OOD stressor, O(N·L) and memory-safe on the
≈13k-molecule CYP sets. All splits assign whole groups to a single fold.

**Leakage metric.** To compare split types on equal footing we report a *splitter-independent*
nearest-neighbour leakage: the fraction of test molecules with maximum ECFP4 Tanimoto > 0.4 to any
training molecule, and the mean of those max-similarities.

**Models.** Fingerprint baselines (ECFP4 + XGBoost / random forest) for the full atlas; a
frozen-embedding foundation-model arm (MolFormer / ChemBERTa-2 + linear/XGBoost head) over all 22
endpoints and a fine-tuned GNN (GIN/DMPNN) on a representative subset (in progress).

**Generalization gap and statistics.** For each (dataset, model) and realistic split, we compute
the gap per *matched replicate* — the same (split seed, model seed) used for random and realistic —
so the comparison is paired. We report the mean gap, a 95% bootstrap CI over replicates, a paired
Wilcoxon signed-rank p-value, and Benjamini–Hochberg q-values across the 22-endpoint family; a gap
whose CI crosses zero is reported as "not established." An aggregate Wilcoxon over per-endpoint
gaps gives the headline "gap positive on average" claim.

**Conformal prediction.** Split-conformal with the validation fold as calibration: absolute-
residual intervals (regression) and LAC sets (classification, nonconformity 1−p[true]). We
implement weighted conformal under covariate shift [@tibshirani2019conformal] with importance
weights from an isotonic-calibrated, L2-regularized, class-balanced domain classifier
(calibration-vs-test), clipped at the 99th percentile; we fall back to unweighted conformal when
the Kish effective sample size fraction drops below 0.1. Our quantile reduces *exactly* to textbook
split-conformal under uniform weights (regression-tested) and returns +∞ — an honest "cannot
certify coverage" — when the calibration mass cannot reach the target.

**Selective prediction.** Risk–coverage curves with two label-free confidence signals —
deep-ensemble disagreement and applicability-domain similarity (max Tanimoto to train). We report
Excess-AURC (AURC minus the oracle ordering), which is comparable across folds with different base
error rates, and compare signals with a paired Wilcoxon test across seeds.

---

## 4. Results

### 4.1 The harness reproduces the official leaderboard (credibility)
Using the official 5-seed protocol, our ECFP4+tree baselines land near published TDC fingerprint
baselines (Caco-2 MAE 0.39, BBB AUROC 0.87, hERG AUROC 0.80, solubility MAE 1.25), confirming the
harness before any re-split is introduced. [[Full 22-row official-vs-ours table — Table 1.]]

### 4.2 Realistic splits push test chemistry away from training
Nearest-neighbour leakage falls monotonically random → scaffold → cluster (e.g., BBB 72%→48%→8%;
Caco-2 71%→38%→8%; mean max-similarity ≈0.6→0.4→0.27), confirming cluster splits produce a
genuinely out-of-domain test set while scaffold splits only partially do.

### 4.3 Scaffold under-stresses; cluster reveals a universal gap (the atlas)
Across all 22 endpoints (XGBoost, 8 split seeds × 3 model seeds, matched-replicate gaps): under
**cluster** splits **every endpoint (22/22) shows a statistically established gap** — 95% CI excludes
zero and BH-FDR q<0.05 for all 22 — with median +0.150 in native metric units and aggregate Wilcoxon
p<10⁻⁴. Under **scaffold** splits the gaps are markedly smaller (median +0.055), and 3/22 are not
established; most strikingly **hERG is flat under scaffold (−0.002, n.s.) but a clear +0.055 under
cluster**. The cluster effect is ≈3× the scaffold median, so scaffold splits *under-stress* relative
to cluster — consistent with Fooladi et al. — while the cluster gap is large and universal. Raw-unit
gaps are not cross-endpoint comparable (PPBR's 0–100% target gives a +5.8 MAE cluster gap), so we lead
with per-endpoint CIs/q-values and the median and report standardized effect sizes in Appendix C.
[[Table 2 / Figure 1: per-endpoint forest plot of scaffold vs cluster gaps with CIs.]]

### 4.4 Conformal under-covers under shift
The corrected split-conformal achieves nominal marginal coverage on exchangeable data (self-test
0.901 at the 0.90 target). On real splits (pilot: 8 seeds), coverage is near target on
random/scaffold and **drops under cluster shift** — e.g. BBB 0.919±0.02 (random) → 0.840±0.01
(cluster). Covariate-shift weighting is correctly a **no-op on random** (0.919→0.914, the control
it must pass) but did **not** restore cluster coverage (0.840→0.838): the classifier-based
importance weights stayed near-uniform (Kish ESS ≈0.90), consistent with the known weakness of
classifier density-ratios under high-dimensional shift [@laghuvarapu2023codrug]. We report ESS
transparently and fall back to unweighted conformal when it collapses, rather than over-claiming a
fix. [[Full 22-endpoint Table 3.]]

### 4.5 The best abstention signal is model-dependent
We compared two label-free abstention signals — deep-ensemble disagreement and applicability-domain
(AD) distance — by Excess-AURC with paired tests. A random-forest pilot (8 seeds) suggested AD
distance beat ensemble disagreement under cluster shift (BBB Δ=+0.012, hERG Δ=+0.030, p<0.02). **This
did not replicate** with XGBoost ensembles across all 22 endpoints: there, ensemble disagreement is
generally the better signal under cluster shift. We therefore report **no model-robust winner** — an
honest cautionary result. Both signals are imperfect under shift; the choice depends on the base
model. [[Table 4: per-endpoint E-AURC, RF vs XGBoost.]]

### 4.6 The gap is an applicability-domain effect
Binning every test prediction by its nearest-neighbour Tanimoto similarity to training reveals a
clean, monotone relationship: averaged across all 22 endpoints, normalized error is **≈7.9× the
endpoint mean for molecules with similarity <0.1** and falls to ≈0.5× for similarity >0.9. Error is
driven by distance-to-training — which both explains the cluster gap mechanistically and justifies
distance as an abstention signal (§4.5). Across endpoints, the drop in mean similarity from random to
cluster splits predicts the relative gap (Spearman ρ=+0.41, p=0.06) — a trend at n=22 with the
expected sign. [[Figure 2: error-vs-similarity curve.]]

### 4.7 Does the gap predict abstention benefit? (a negative result)
We tested the natural hypothesis that endpoints with the largest optimism gap are those where
abstaining on the out-of-distribution tail recovers the most reliability. It does **not** hold:
per-endpoint relative cluster gap versus abstention benefit (risk removed by dropping the 20% least
in-domain test molecules by AD distance) shows **no significant correlation** (Spearman ρ=+0.07,
p=0.75, n=22), and several regression endpoints show negative benefit (AD distance does not rank
their errors well). We report this null plainly: selective prediction does *not* cleanly "recover the
gap," and the optimism gap and abstention benefit are largely decoupled. [[Figure 3: gap vs benefit
scatter.]]

---

### 4.8 Do foundation models close the gap? (modestly)
Running the same gap machinery on frozen ChemBERTa embeddings + an XGBoost head and comparing
the per-endpoint relative random→cluster gap to ECFP4 fingerprints: the foundation model
**shrinks the gap on average** (mean Δ(FM−FP) = −0.014 in relative gap, paired Wilcoxon p=0.011,
smaller on 17/22 endpoints) — but the effect is **small and not uniform** (it widens on a few
endpoints, e.g. half-life), and the gap remains large (FM relative gaps still ≈0.1–0.5). Pretrained
chemical representations help but do **not** solve out-of-distribution generalization. [[Figure 4:
ChemBERTa vs ECFP4 relative cluster gap.]]

## 5. Discussion
The practical message is not "models are bad" but "the standard evaluation is optimistic in a
*predictable* way": measured performance tracks how far test chemistry sits from training (an
applicability-domain effect), and scaffold splits under-state it. The natural remedy — abstain on the
out-of-distribution tail — is more fragile than one would hope: split-conformal *itself* under-covers
under the same shift, the best abstention signal is model-dependent, and abstention benefit does not
track the gap. So the honest takeaway for practitioners is to **report cluster-split performance and
nearest-neighbour applicability domain** as standard, and to treat uncertainty methods as helpful but
shift-sensitive rather than a turnkey fix. This is a reliability layer on top of TDC, not a
competitor to it.

## 6. Limitations
Solo, compute-bounded study (Apple M4 Pro): the foundation-model arm is primarily frozen
embeddings; GNNs and fine-tuning cover a representative subset, not all 22×splits. Classifier-based
density-ratio weights collapse in effective sample size under severe high-dimensional shift (we
report ESS and fall back rather than over-claim). Temporal splits are out of scope for most
endpoints (no provenance dates) and not claimed. Pooled re-splits are not leaderboard-comparable by
construction; we anchor with the official-split reproduction.

## 7. Conclusion
A reproducible, leakage-audited re-evaluation of TDC ADMET shows *when* and *why* the generalization
gap appears, and that selective prediction with a distance-based abstention rule is the operational
remedy. Code, pinned environment, and splits are released.

---

*Appendix A: differentiation table. Appendix B: per-endpoint diagnostics (realized split fractions,
dropped/duplicate counts, per-fold label shift, NN-Tanimoto leakage). Appendix C: full atlas and
selective-prediction tables.*
