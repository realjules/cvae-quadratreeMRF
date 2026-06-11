# NeurIPS Research Direction Review — June 2026

Produced by a 29-agent review pipeline: 5 agents re-audited the experimental record file-by-file,
6 agents swept 2023–2026 literature with fetch-verified citations, 4 lenses generated candidate
directions, which were consolidated to 6 and each attacked by an adversarial novelty referee
(web-searching for killer prior work) and a feasibility skeptic (verifying premises against this
repo). All paper URLs below were verified by fetching unless marked otherwise.

---

## PART 0 — Two things to fix before anything else

### 0.1 Integrity blocker (fix today)

`docs/paper_plan.md` line 14 (the draft abstract) reports unconstrained degeneracy results
**"on 3 seeds" with values 0.729 / 0.196 that exist nowhere in this repository**. The only real
unconstrained measurement is 0.789 / 0.094 from a single March 2026 run, and the same file lists
Experiment H (unconstrained training) as PENDING. These are placeholder numbers written in the
past tense with fabricated precision. If they survive into a submission they constitute
fabrication. Replace with `[TBD after Exp H]` immediately, and adopt the rule: no number enters
paper text without a pointer to a results file.

### 0.2 Numbers that must be retired from all drafts (audit verdicts)

| Claim | Verdict | Why |
|---|---|---|
| cos = −0.028 anti-correlation at 10% labels | **INVALID** | Single degraded checkpoint; the 3-seed reproduction gives **+0.302 ± 0.19** (min seed +0.175). Did not reproduce. |
| encoder.layer1 redirected below batch-noise baseline | **INVALID** | 0.264 ± 0.31 vs baseline 0.290 — zero separation, sign flips across seeds. |
| 7–10x gradient amplification + ~2x/level depth law | **INVALID** | Loss-magnitude confound (already acknowledged), but the dead numbers still sit in the "Future Research Direction" section of the experiment log. |
| "BP helps all classes" | **INVALID** | Contradicted by your own 3-seed data: Trees −6.75pp. |
| 3-level quadtree is optimal depth | **INVALID** | 5-epoch single runs; with entropy gating, 2 levels (62.91%) ≈ 3 levels (62.80%). |
| System beats linear-probe ceiling by +2.5pp | **INVALID** | Within noise; the probe reference itself is inconsistent in the log (67.47 vs 71.5, never reconciled). |
| "11.5x" encoder.layer1 loss-normalized amplification | **WEAK** | Spans 2.25x–11.47x across checkpoints, no std, near-zero denominator. Say "2–12x across checkpoints" or drop. Also optimization-irrelevant under AdamW + grad clipping (scale invariance). |

### 0.3 What SURVIVES the re-audit

| Claim | Verdict | Scope |
|---|---|---|
| Entropy-gated BP +4.65 ± 1.1pp over no-BP | **SOLID** | 3 paired seeds, 6/6 positive deltas, t≈7.3. Scope strictly: best-epoch metric, 30 epochs, 10% labels, Vaihingen, this pipeline. At 50 epochs the final-epoch delta inverted (−2.8pp) — report both. |
| Unary-head gradient redirection (cos 0.302 ± 0.19 vs 0.913 baseline) | **SOLID** | 3 seeds, ~3σ below baseline. Descriptive only — your own April review reframed divergence as expected "inference offloading". Confined to unary-head params. |
| BP-only parameter groups (encoder.layer3, pairwise heads) | **SOLID but near-tautological** | Deterministic, but architectural: the direct path reads only p1 by construction. Present as a design observation, not a discovery. |
| Inference dependency (36.7% vs 62.0% unary-only) | **SOLID qualitatively, n=1** | 25.4pp ≫ any noise floor, but single run on 50-epoch checkpoints that differ from the seeded 30-epoch ones. Free to replicate: `evaluate_encoder.py` Stage B on the 6 existing checkpoints, ~2h, zero training. |
| Label scaling 10%→50% helps (+7–10pp) | **SOLID directionally** | The only conclusion the 12-agent review validated; epoch confound biases against the trend yet it appears. The 50–100% "plateau / encoder bottleneck crossover" is UNTESTED storytelling (100% arm trained 8 epochs, family peaks ~15). |
| Pairwise degeneracy (diag ratio 0.094, Tree→Building 0.789) | **WEAK — n=1 with confounds** | One March-pipeline checkpoint (label fraction not even recorded); unconstrained head was deleted in commit 503db8a; ~0.62 of the 0.094-vs-0.784 gap is initialization prior (unconstrained inits at 0.167, constrained at ~0.83 and α barely moved from 0.80→0.74); era confound (constrained-but-ungated gave 0.270 — a value that is also mathematically impossible given α's lower bound, so one logged number is wrong); chance level is 1/K = 0.167, not the log's ">0.5". |
| Dose-response of gradient conflict (10% vs 50%) | **WEAK** | 50% arm is ONE checkpoint, ONE batch; the 5-batch sweep accidentally ran on a random-init model. Two dose levels ≠ dose-response. |
| +4.65pp is "structured inference" | **UNTESTED** | The receptive-field-matched dumb-pooling baseline is **already coded** (`DumbPoolingModule`, net/dhbp.py:47, wired via `--dumb_pooling`) and has never been run. The no-BP comparator reads only p1, so the delta conflates message passing with multi-scale feature access. Honest internal estimate: 55–65% probability pooling matches BP. |

### 0.4 Methodology problems found in the gradient instrument

`test_gradient_cosine.py` mechanics are clean (same FocalLoss both paths, correct zero_grad/clone;
the feared double log-softmax is an exact mathematical no-op through F.cross_entropy). But:

1. **Every reported cosine was measured on Gaussian-noise inputs with uniform random labels**, in
   train-mode BatchNorm, omitting the boundary loss term. Claims about training dynamics rest on
   off-manifold gradients. Re-measure on ≥30 real Vaihingen batches with the full training loss.
2. **The null is wrong on two axes**: same-path/different-batch answers a different question than
   different-path/same-batch, and the baseline was computed once at fresh weights but compared
   against trained checkpoints. Needed: full 2×2 (path × data) with per-checkpoint baselines for
   both paths, plus an architecture-matched non-BP second head (dumb pooling) as the proper control.
3. Entropy gating has **two code inconsistencies** that block any clean formalization: gates are
   computed from post-message beliefs, not unary potentials (dhbp.py:300), and the top-down cavity
   subtracts the ungated message while the bottom-up used the gated sum (dhbp.py:345 vs :308) —
   a child's own evidence leaks into its downward message. Fixable (~1–2 days); the +4.65pp must
   then be re-verified, since it was measured with the inconsistent version.

---

## PART 1 — What the literature actually says (verified June 2026)

### Confirmed-open gaps (multiple search angles each, fetch-verified)

- **No paper 2015–2026 documents unconstrained learned K×K pairwise/compatibility potentials
  collapsing to class remapping in end-to-end differentiable inference.** The two systems that
  learn full K×K matrices through differentiable inference — Larsson et al. (arXiv:1701.06805)
  and BP-Layers (Knöbelreiter et al., CVPR 2020, arXiv:2003.06258) — report benign interpretable
  potentials, but **only under full supervision**. Nobody studies learned potentials as labeled
  fraction shrinks.
- **No paper measures module-level "removal gap"** (a unary head collapsing when its jointly
  trained inference module is lesioned at test time), and no paper stochastically drops a BP/CRF
  layer during training. The field engineers around the dependency (Aux-NAS, ICLR 2024) without
  measuring it.
- **No formalization of confidence/entropy-gated message aggregation** with free-energy semantics
  (entropy-derived per-edge counting numbers).
- **No label-fraction dose-response of any gradient-geometry quantity** anywhere.
- **No 2024–2026 paper learns structured-inference potentials end-to-end on frozen foundation
  features** for label-efficient segmentation (DINOv3 report uses only linear probes/light
  decoders; Docherty et al. arXiv:2410.19836 is post-hoc unlearned DenseCRF).
- **"Spatial confirmation bias" remains unformalized** (arXiv:2509.16704 and arXiv:2601.11670 both
  exist but are sample/confidence-level, explicitly ignoring spatial structure).

### Near-preemptions the verifiers found (must cite and distinguish)

- **CPGNN** (Zhu et al., AAAI 2021, arXiv:2009.13566) — learned compatibility matrix in BP-style
  GNNs fails (~30% drop) without empirical-statistics init + centering regularizer. Reports only
  the accuracy drop, never characterizes remapping, on graphs not MRFs. The umbrella claim
  "unconstrained compatibility matrices need symmetry-breaking" is loosely precedented here.
- **SGA** (Li et al., ECCV 2022) — **kills any "first to measure path-gradient conflict" claim**:
  measures cosine conflict between gradient branches of a single loss (skip vs attention paths),
  tracks it over training, fixes with stop-gradient. The surviving delta is the controlled 2×2
  protocol, supervision dose-response, structured-inference setting, and per-class mechanism link.
- **Chertkov et al.** (arXiv:2301.10369) — instance-dependent fractional/counting parameters with
  per-instance exactness already exist; "first input-adaptive counting numbers" must narrow to
  "first confidence-DERIVED, per-edge, predictive counting numbers for dense prediction".
- **Chen et al.** (ICCV 2025, arXiv:2504.02008) — pairwise CRF-style loss + entropy as test-time
  objective on a frozen foundation medical segmenter. Kills "first test-time spatial energy";
  binary masks/latent updates leave a multi-class/parametric/label-efficiency delta.
- **CRFNet** (Pastorino et al., IEEE TGRS 2024) — already learns CRF potentials end-to-end at
  10%/30% sparse GT on Vaihingen/Potsdam. Note: its 83–84% is **overall accuracy under an
  erosion-based sparse-GT protocol**, NOT mIoU under the SSL community's image-split protocol
  (RegionMatch, IJCAI 2025: Vaihingen 69.25 mIoU at 1/8 labels). Any comparison must bridge
  protocols explicitly or it is attackable from both sides.
- **Neural collapse already reached segmentation** — the March idea list's assumption is FALSE:
  Zhong et al., CVPR 2023 (arXiv:2301.01100) extended NC to semantic segmentation; minority
  collapse proven for SimCLR-family losses (Nguyen et al., BU preprint, Sep 2025). Idea 4 from
  research_ideas.md is dead as originally framed.
- **Werner, TPAMI 2007** — classical non-identifiability of unary+pairwise energies up to
  equivalent transformations; **Liu/Cheng/Zhang (arXiv:2202.02016)** — K×K transition matrices
  identifiable only up to label permutation, diagonal dominance removes the ambiguity (label-noise
  setting, never connected to CRF/MRF compatibility learning — that bridge is free to claim).

### Venue calendar (verified against official pages 2026-06-09/10)

| Venue | Deadline | Status |
|---|---|---|
| NeurIPS 2026 main | May 4/6, 2026 | **PASSED** — next main NeurIPS is 2027 (~May 2027) |
| TPM workshop @ UAI 2026 | **June 12, 2026** | 1–2 days away; best topical fit but only if a 4-pager can be compressed immediately (long shot) |
| NeurIPS 2026 workshops | **~Aug 29, 2026** (list announced July 11) | The realistic near-term target for the degeneracy 4-pager (non-archival; preserves main-track novelty) |
| ICLR 2027 | ~Sep 24, 2026 (projected) | Nearest main-track shot (~14 weeks) |
| AISTATS 2027 | ~Oct 10, 2026 (projected) | Fallback, fits a light-theory variational framing |
| CVPR 2027 | ~Nov 13, 2026 (projected) | Only with the frozen-DINOv2/v3 platform (absolute accuracy must be defensible) |
| UAI 2027 | ~Feb 10, 2027 (projected) | The natural home for the identifiability framing |
| TMLR | rolling | Acceptance criteria exclude novelty/significance; the scoped single-dataset version already qualifies |

### The acceptance recipe for analysis/characterization papers (from 10 verified 2024–2026 acceptances)

1. A **named mechanism** (attention sinks, register tokens, …) — you have "compatibility collapse /
   inference offloading". ✔
2. A **causal counterfactual that switches the phenomenon off** (registers, StableMax) — your
   α·I+(1−α)·R constraint is exactly this move. ✔
3. **Breadth or ubiquity**: every main-track acceptance had ≥3 architectures/model families or a
   ubiquitous object. **You currently have 1 dataset, 1 backbone, ~65% accuracy — this is the bar
   you do not yet clear.** ✘
4. A minimal fix elevates (optional). ✔ (the constraint)
5. Resource release can substitute (optional).

---

## PART 2 — The six candidate directions, with adversarial verdicts

Feasibility scores = a calibrated skeptic's probability (0–10) of a credible **NeurIPS-2027
main-track** submission by May 2027 for this team specifically, after reading this repo.
Workshop-level success probabilities are much higher for the top candidates (~60–70%).

### #1 — Compatibility Collapse: class-remapping degeneracy as an identifiability failure, with a supervision-scarcity law
**Novelty: NOVEL (the only one). Feasibility: 3/10 main-track; ~60–70% workshop/TMLR.**

Thesis: unconstrained learned K×K pairwise potentials trained only through the composed inference
output are non-identifiable up to a label-permutation orbit (cite Liu/Cheng/Zhang 2202.02016 +
Werner 2007 — bridge unclaimed); class imbalance selects the remapped orbit member; the drift from
smoothing-like to permutation-like solutions follows the labeled fraction (measure with Hungarian
permutation distance + KL to the empirical parent-child co-occurrence null, not diag ratio);
any symmetry-breaking intervention (constraint, aux unary loss, balance) eliminates it.

- Verifier searched 11 angles and found **no preemption**; CPGNN is the closest threat and must be
  engaged ("the bare discovery that learned compatibility matrices can fail is loosely precedented;
  sell the identifiability reframing + the law + the synthetic ground-truth anchor").
- The genre slot — a posterior-collapse-style "degeneracy iff non-identifiability" paper
  (Wang/Blei/Cunningham, NeurIPS 2021) for **discriminative structured inference** — is unoccupied.
- **Top failure mode**: the P0 gate. The headline 0.094 is one confounded, unreproduced run from a
  regime where the unary itself had collapsed; ~0.62 of the gap is init prior. The 3-seed
  unconstrained rerun (+ diagonal-init control) is a genuine coin-flip and costs ~6h.
- Required experiment matrix: unconstrained 3-seed gate → metric hardening (pure analysis) →
  diagonal-init disambiguation → mechanism controls (aux-loss / class-balance / linear-mixing-no-BP)
  → label-fraction dose-response {5,10,25,50,100}% → mean-field CRF arm (graph-agnosticism; 2–4
  days code) → synthetic quadtree-MRF testbed with known ground-truth potentials (3–5 days code,
  runs on the local RTX 2050, the only line not bound by Kaggle quota) → Potsdam.

### #2 — Inference Offloading: naming, measuring, and curing the removal gap (BP-Drop)
**Novelty: PARTIAL. Feasibility: 3.5/10 (highest) — "best-grounded candidate one could build from this repo".**

Thesis: joint training with an always-on corrective inference module induces a measurable removal
gap (your 36.7 vs 62.0), seed-robust, supervision-dependent, and driven toward zero by stochastic
module-drop training (BP applied with probability q) along a dependency-performance frontier —
"train structured, deploy unary" at zero inference cost.

- Survives: first quantification on inference modules; stochastic dropping of a BP/CRF layer; the
  (q, λ) frontier; first measured test of whether the ConvCRF-era auxiliary-unary-loss folk fix
  actually cures dependency. Does NOT survive: "train structured, deploy free" as a goal (GLNN
  arXiv:2110.08727 and AIN arXiv:2009.08229 do it via distillation — a drop-vs-distill comparison
  is mandatory) and the drop mechanism in spirit (FractalNet drop-path arXiv:1605.07648 states the
  anti-co-adaptation rationale almost verbatim; cite it or a reviewer will).
- **Free first step**: the seeded replication is zero-training — run `evaluate_encoder.py` Stage B
  on all 6 existing checkpoints (~2h, local). Note the 36.7/62.0 came from different (50-epoch)
  checkpoints, so this is a real test.
- Caveat: part of the gap is wired in (unary reads only p1; BP is sole gradient source for
  layer2-3) — the multi-scale-unary control or explicit scoping is mandatory, and your own April
  review reframed offloading as expected behavior: present the measured frontier, never bare pathology.

### #3 — Path-Gradient Alignment: one loss, many paths (2×2 protocol + dose-response + per-class mechanism)
**Novelty: PARTIAL (SGA ECCV 2022 is the existence proof — reposition, don't claim firsts). Feasibility: 3/10.**

The cheapest direction (almost all local compute, parasitizes every checkpoint other directions
produce). Its real value is as the **mechanism section of #1**: per-class masked-loss gradient
decomposition — if misalignment concentrates on the remapped classes (Tree/Building, Cars), that
is the first mechanistic bridge from gradient geometry to compatibility collapse and the strongest
figure across all directions. Standalone it caps at TMLR without a cross-architecture battery.
Gate: rebuild the probe on real data (the noise-input numbers already moved 0.33 once), and the
dumb-pooling head must show LESS misalignment than BP, else this collapses into known MTL territory
(Du 2018; Elich GCPR 2024).

### #4 — Frozen foundation features + constrained near-parameter-free structured inference
**Novelty: PARTIAL ("CRFNet with a DINOv3 encoder" is the reviewer frame). Feasibility: 3/10.**

The PANGAEA gap (arXiv:2412.04204 — frozen GFMs don't consistently beat supervised UNets at low
labels; nobody tests whether structured inference closes it) is real and the conjunction is unrun.
But your own Exp 5 diagnosis ("quadtree BP is spatial averaging, the wrong graph") predicts the
parameter-matched decoder control will match the structured head, and your data show the BP margin
shrinking as the unary strengthens (+2.5pp at 100% labels) — DINOv3 strengthens the unary far more
than that. Run the 1-day linear-probe headroom pilot + post-hoc DenseCRF before any adapter build.
Even the negative result is publishable against PANGAEA (workshop-grade). This is also the platform
that fixes the "65% vs 83–84%" credibility problem for any main-track version of #1/#2.

### #5 — Spatial MRF energy as a test-time objective on frozen features
**Novelty: PARTIAL (Chen ICCV 2025 occupies the lane; multi-class/parametric/label-efficiency delta survives). Feasibility: 3/10.**

Most new infrastructure of any direction; modal outcome is post-hoc DenseCRF matching the TTA arm
(and pairwise-agreement energies share entropy minimization's uniform-collapse optimum — you'd need
anti-collapse regularization, and class collapse is this project's recurring failure family).
Start only if #4's platform exists and its pilot is positive.

### #6 — Entropy-gated message passing formalized as input-adaptive fractional BP
**Novelty: PARTIAL. Feasibility: 2/10 — three serial kill-gates.**

The math is real (in log space the gated aggregation is exactly messages raised to fractional
powers summing to 4 — power-EP/fractional-BP counting numbers c_e = 4·softmax(−H); TRW is provably
unavailable on trees) and the audit sketched the derivation. But: gate 1 = dumb pooling (55–65%
kill by your own estimate), gate 2 = the corrected scheme must retain +4.65pp (it was measured with
the inconsistent cavity — coin flip), gate 3 = gated must beat exact sum-product (thread the
hardcoded `use_attention=True` to the CLI, ~2h). Naive chaining ≈ 12–18% the empirical story
survives. Run the three cheap gates (~18h total) before investing any theory time. Even full
success is an AISTATS/UAI paper, not NeurIPS main.

---

## PART 3 — Recommendation

### The strategic read

No single candidate honestly clears the NeurIPS-main bar alone by May 2027 — every feasibility
skeptic converged on 2–3.5/10, mostly because of the breadth requirement (1 dataset, 1 backbone,
uncompetitive absolute accuracy) and unrun P0 gates. But the top three candidates **compose into
one main-track paper** and share gate experiments:

> **"Compatibility Collapse: how learned potentials degenerate under scarce supervision —
> the phenomenon (#1), its optimization signature (#3 as mechanism section), and its
> dependency cost (#2)"** — with the constrained decomposition as the causal counterfactual
> that switches it off, a synthetic ground-truth anchor, and a mean-field CRF arm for
> graph-agnosticism.

This matches the verified acceptance recipe for analysis papers lever-for-lever (named mechanism ✔,
causal counterfactual ✔, minimal fix ✔) and the breadth lever is exactly what the full experiment
matrix adds. Pipeline: **NeurIPS 2026 workshop (Aug 29) → ICLR 2027 (Sep 24) or UAI 2027 (Feb 2027)
→ NeurIPS 2027 main (May 2027)** depending on how much of the matrix lands.

### Week 1 — the gates (~1 Kaggle week + local compute, mostly zero new code)

| # | Experiment | Cost | Gates |
|---|---|---|---|
| 1 | Fix paper_plan.md fabricated numbers; purge −0.028 and the retired claims | 1h | integrity |
| 2 | Pin the eval-noise floor: k=10 re-evals of one checkpoint | ~2h local | every later claim |
| 3 | **Dumb-pooling 3-seed** (`--dumb_pooling`, coded, never run) + lesion eval | ~6h T4, zero code | #1's framing, #2's control, #3's control, #6's gate 1 — the single highest-information run |
| 4 | **Unconstrained-pairwise 3-seed** (restore head from `git 503db8a^`, ~0.5 day flag) + diagonal-init control arm, same diagnostics protocol | ~1 day + 12h T4 | #1's existence (the coin-flip) |
| 5 | **Removal-gap seeded replication**: `evaluate_encoder.py` Stage B on all 6 checkpoints | ~2h local, zero code | #2's headline |
| 6 | Back up the contrastive checkpoint off Kaggle (single point of failure) | minutes | everything |

### Decision tree after week 1

- **Unconstrained rerun reproduces collapse + dumb pooling ≠ BP** → strongest world: full #1+#3+#2
  merged program; workshop 4-pager by Aug 29; mean-field arm + synthetic testbed + Potsdam toward
  ICLR 2027/UAI 2027; #6's formalization becomes worth 2–4 weeks.
- **Unconstrained reproduces + dumb pooling ≈ BP** (modal outcome, 55–65%) → degeneracy paper
  survives and gets *cleaner* as pure characterization (drop all "structured inference benefit"
  language); #6 dies; #2 reframes around removability/deployment; #4's platform becomes the route
  to breadth.
- **Unconstrained fails to reproduce** (diag ratio stays >0.4 or diagonal-init survives training) →
  the March 0.094 was a collapsed-unary/init artifact. Pivot to #2 (removal gap, which passed its
  own free replication by then or didn't) + the synthetic testbed to find the regime where
  collapse *does* occur (label fraction × imbalance × K sweep, local GPU) — "when do learned
  potentials degenerate?" is still a paper if the answer has structure.

### Hygiene for whichever paper ships

Use one diagnostics script/protocol for both arms; state the 1/K = 0.167 chance level; report the
init-time diag ratio of both heads (decompose constraint vs init honestly); reconcile or retract
the impossible 0.270; report best-epoch AND final-epoch; per-class claims only from 3-seed tables
(including Trees −6.75pp); bridge the OA-vs-mIoU protocol gap explicitly when citing CRFNet vs
RegionMatch; cite CPGNN, SGA, Werner 2007, Liu/Cheng/Zhang, FractalNet, GLNN/AIN, Chen ICCV 2025
where the verdicts above demand.

---

## Appendix — provenance

Workflow run `wf_632b8d3e-dda` (29 agents, ~1.5M agent tokens, 337+218 tool calls). Full structured
outputs: audits, literature (6 sweeps, ~70 verified papers), candidates, and per-candidate verdicts
are in the session task output (`wvn9b9qdi.output`). Key single-source caveats flagged by the
verifiers: CRFNet's 83–84% could not be re-verified from the paywalled text (protocol verified via
Semantic Scholar + official repo); ConvCRF's auxiliary-loss training protocol claim rests on the
team's reading of the full text (abstract inconclusive); ICLR 2027 / AISTATS 2027 / CVPR 2027 /
UAI 2027 dates are aggregator projections, not official CFPs.
