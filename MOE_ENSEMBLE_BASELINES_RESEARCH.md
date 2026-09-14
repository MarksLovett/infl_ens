# Comparative Baselines for Game-Theoretic LoRA Ensemble Training

## Executive recommendation

The existing list is already strong. In particular, the fixed-σ soft-clustering arm is the cleanest ablation of the paper's theoretical contribution. It should be presented explicitly as a parameter-efficient, trait-space analogue of cluster-Branch-Train-Merge (c-BTM): cluster the corpus, train independent experts, and use distance-to-centroid weights at inference.[^1] The label-partitioned arm is the corresponding BTM/DEMix-style supervised-domain comparator.[^2][^3]

The largest remaining holes are not another large MoE architecture. They are inexpensive controls over the *same trained experts* and a middle ground between a fixed geometric router and joint end-to-end MoE training.

| Priority | Addition | Referee question answered | New LoRA training | Recommendation |
|---|---|---|---:|---|
| 1 | Same-expert routing nulls: shuffled expert identities, uniform random hard routing, and uniform mixture | Did the learned geometry align prompts with the experts, or are seven experts useful under almost any combination rule? | None | Must run |
| 2 | Frozen-expert soft gate trained from the full expert-NLL vector | Can an ordinary predictive router beat the game router without changing expert quality? | None | Must run |
| 3 | BASE-style balanced assignment / entropic optimal transport using the same Gaussian affinities | Are the gains caused by load balance rather than the strategic positioning rule? | One arm | Must run if routing loads are unequal |
| 4 | c-BTM-faithful balanced clustering and top-k centroid routing | Does a published cluster-specialize-ensemble recipe explain the result? | One arm; largely overlaps the planned clustering arm | Rename and strengthen the planned arm |
| 5 | EWoRA: partition one pooled LoRA rank budget into context-weighted sub-adapters | Does a parameter-budget-matched learned low-rank mixture suffice? | One new joint arm | Strongest recent capacity-controlled PEFT-MoE addition |
| 6 | AdaMix-style stochastic LoRA routing with consistency regularization and post-training merge | Is stochastic parameter ensembling, without semantic specialization, sufficient? | One arm | Strong low-to-medium-cost comparator |
| 7 | MoLoRA: a jointly trained mixture of low-rank experts | Why not use the standard PEFT-MoE alternative? | New joint architecture | Best established external MoE comparator |
| 8 | MoDULA-Res: universal LoRA plus routed domain LoRAs | Is a shared/general component plus residual specialization enough? | Generalist, domain experts, then router | Strong if general-capability retention is a paper claim |
| 9 | KnOTS plus linear/TIES/DARE merging | Can the experts be collapsed into one LoRA without routing? | None after domain experts exist | Add KnOTS to the planned merge suite |
| 10 | Pooled-seed deep ensemble | Is the effect just independent fine-tuning variance? | Expensive unless seed runs already exist | Useful ceiling, not a fairness-matched main baseline |

The smallest defensible expansion is therefore: retain the capacity, metadata, and clustering questions, but redesign the random-shard and rank-112 controls; add same-expert routing nulls, a frozen-expert soft gate, and BASE-style balanced assignment; call the clustering arm c-BTM/Gaussian-mixture EM; and use MoLoRA as the one named PEFT-MoE comparator. Add MoDULA and merging only if the paper makes claims about shared general knowledge or single-model deployment.

## Direct critique of the proposed choices

Several proposed choices are good ideas with imprecise experimental definitions. Two are weak enough in their current form that I would not make them headline baselines.

| Proposed choice | Verdict | Main problem | Recommended version |
|---|---|---|---|
| Rank-112 pooled generalist | Keep, but split into two controls | It matches stored trainable parameters, not active parameters, optimizer behavior, or necessarily compute. A high-rank LoRA can also be unintentionally weakened by its scaling. | Keep pooled rank 16 as the ordinary generalist; add rank 112 as a storage-capacity control with tuned/rank-stabilized scaling; report a small rank sweep if affordable. |
| Label-partitioned experts | Keep; one of the strongest baselines | A benchmark-label router is metadata-supervised, not an oracle. Routing these experts with unrelated game positions can manufacture a bad result. | Use train-only label centroids for geometric routing, an explicit metadata router, and a deployable prompt-to-label classifier. Reserve “oracle” for response-dependent best-expert selection. |
| Hard and soft clustering | Highest-priority training ablation | “Same σ” has no content for nearest-centroid hard k-means because a common σ does not change the winning centroid. Static versus round-wise updates can also confound the positioning rule. | Use balanced k-means/c-BTM, fixed-covariance Gaussian-mixture EM, and the game update with identical initialization, update cadence, kernel, data budget, and routing sparsity. |
| Random equal shards with the same router | Demote in its current form | A random expert has no canonical geometric position, so the result can be driven by an arbitrary router-to-expert permutation. It is not a clean null for multiplicity. | Treat it as a placebo and evaluate the same experts with uniform routing, validation-selected identity permutation, and a frozen fitted gate. A pooled-seed ensemble is the cleaner multiplicity control. |
| Expert-choice routing | Appendix unless capacity is central | It is not just a sibling top-k mask: experts choose prompts, so prompts may be duplicated or unassigned and inference fan-out varies. It changes both load balance and computation semantics. | Prefer BASE/optimal-transport assignment as the main load-balanced comparator; keep Expert Choice only if variable fan-out is itself of interest. |
| Hard oracle-distilled classifier | Keep, but strengthen | Hard `argmin-NLL` targets discard loss margins and are noisy; labels produced on expert-training examples leak membership/memorization. | Cross-fit the teacher scores and compare hard classification, soft regret-aware targets, and direct expert-loss regression. |
| Generic Switch/GShard learned gate | Replace as the main external baseline | A token-level, jointly trained Transformer MoE changes architecture, granularity, and training regime simultaneously. A loss is hard to interpret causally. | Use prompt-level MoLoRA with the same LoRA expert budget, plus a frozen-expert gate to separate routing gains from joint co-specialization. |
| TIES/DARE/task arithmetic | Keep only as a deployment comparison | Merging removes routing and tests a different claim. Applying merge rules directly to LoRA factors is basis-dependent and can be misleading. | Merge reconstructed updates `ΔW = BA`; include linear averaging and task arithmetic controls; add KnOTS before TIES/DARE; match final serving rank. |

### The rank-112 control needs special care

The current model fragment sets `lora_r: 16` and `lora_alpha: 32`. Under conventional LoRA scaling, changing only the rank to 112 reduces `alpha / r` from 2 to about 0.286, a sevenfold change in update scale. That would make a weak rank-112 result ambiguous: it could be a scaling/optimization failure rather than evidence for specialization. Rank-Stabilized LoRA was proposed precisely because conventional `alpha / r` scaling can impair learning as rank increases.[^24]

Call this arm **parameter-count-matched**, not simply “matched.” For every method, report separately:

1. stored adapter parameters;
2. active adapter parameters and forward passes per prompt;
3. training FLOPs or measured GPU-hours; and
4. total token exposures.

A pooled rank-16 LoRA remains necessary even if rank 112 is run: it is the ordinary generalist baseline, while rank 112 is a capacity stress test. For rank 112, tune the learning rate and scale on validation data or include a rank-stabilized setting; do not inherit `lora_alpha: 32` silently. If budget allows, ranks 16, 32, 64, and 112 are more convincing than a single extreme rank because they expose whether the generalist is still improving with capacity.

### The word “oracle” should be used narrowly

There are three different upper bounds in the proposed design:

- **metadata router:** uses the known benchmark/domain label at deployment;
- **prompt-only oracle-distilled router:** learns from held-out expert losses but receives only prompt information at deployment; and
- **hindsight response oracle:** selects the expert with minimum NLL after seeing the reference response.

Only the third is a true unattainable oracle. Calling the metadata router an oracle obscures how much privileged information it receives, while calling the distilled router an oracle obscures its generalization error. Report all three names explicitly.

### What I would drop or demote

- Do **not** report random shards routed by one arbitrary game-expert identity as the only random-shard result.
- Do **not** spend the main engineering budget on a generic token-level Switch/GShard implementation; MoLoRA is much closer to the actual intervention.
- Do **not** treat Expert Choice as a small mask change or as the cleanest load-balancing ablation; BASE/optimal transport is cleaner here.
- Do **not** present TIES/DARE as evidence for or against the game-theoretic training rule. They answer a router-free deployment question.
- Do **not** claim the rank-112 model is compute-matched unless it is actually measured and matched; parameter count alone is insufficient.

## 1. What the present list already covers

The proposed arms answer four separate causal questions:

1. **Capacity:** a rank-112 pooled LoRA asks whether total trainable parameter count, rather than specialization, explains the result.
2. **Available supervision:** benchmark-partitioned experts ask whether observed domain metadata is sufficient.
3. **Positioning:** hard and soft clustering ask whether the game-theoretic position update improves on ordinary centroid estimation.
4. **Multiplicity:** random equal shards ask whether merely training several learners on disjoint data is enough.

That is an unusually good starting design. BTM found that domain specialization was important and that random data splits did not provide the same benefit.[^2] c-BTM then replaced known domains with unsupervised clusters, trained one model per cluster, and sparsely ensembled experts using context-to-centroid similarity.[^1] These are direct precedents for the label-partitioned, random-shard, and clustering arms. They are closer to the proposed system than Switch or GShard because they train independent whole experts on data partitions and combine them later, whereas modern Transformer MoE usually learns token-level feed-forward experts jointly.

Two refinements would make the current arms harder to criticize:

- Use **balanced k-means** as the primary c-BTM comparator, with unconstrained Lloyd k-means as an appendix ablation. c-BTM explicitly studied balanced clustering, and equal shard mass prevents data volume and optimization steps from becoming accidental treatments.[^1]
- Evaluate random-shard experts under both a **null router** and a **router fitted to those experts**. Sending random experts through an arbitrary geometric expert identity is a useful placebo, but it is not the best achievable random-shard ensemble. Uniform mixing and a validation-fitted router distinguish poor expert quality from a meaningless expert-position correspondence.

## 2. Same-expert inference controls

These controls reuse an existing set of trained experts, so they are nearly free. They isolate the routing contribution without changing expert training.

### 2.1 Shuffled or permuted game router

Keep every prompt weight produced by the game router but permute the mapping from router positions to LoRA identities. Report the mean and interval across many of the `K!` possible mappings, or across 100 random permutations when exhaustive enumeration is unnecessary.

This is stronger than random routing alone. It preserves the router's entropy, concentration, load distribution, and prompt dependence while destroying only the learned alignment between regions and trained experts. A substantial gap between the correct identity map and the permutation distribution is direct evidence that expert specialization and geometry co-adapted.

### 2.2 Uniform random hard routing

For each prompt, select one expert uniformly. This preserves one-expert inference cost and estimates the performance of a routing policy with no information. Repeat the sampling enough times, or compute the exact expected NLL as the uniform average of expert NLLs.

### 2.3 Uniform probability mixture and validation-fitted global stacking

A uniform probability mixture asks whether routing is needed at all once all experts are available. A global stacking baseline learns only `K-1` simplex weights on a router-validation split. It can be optimized from cached expert likelihoods without training a language model. DEMix found that parameter-free weighted expert mixtures could improve generalization to heterogeneous or unseen domains, making this a literature-grounded control rather than a cosmetic ensemble statistic.[^3]

The current `routing_eval.py` work already computes game-weighted and uniform **sequence-level** mixtures. That is valuable, but its label should remain precise. Three quantities answer different questions:

\[
L_{\mathrm{sample}}(x,y)=\sum_e w_e(x)L_e(x,y),
\]

\[
L_{\mathrm{seqmix}}(x,y)=-\frac{1}{T}\log\sum_e w_e(x)
\exp\{-T L_e(x,y)\},
\]

and

\[
L_{\mathrm{tokenmix}}(x,y)=-\frac{1}{T}\sum_t\log\sum_e
w_e(x,y_{<t})p_e(y_t\mid x,y_{<t}).
\]

The first is the expected NLL after sampling one expert. The second is the exact marginal likelihood of a latent expert chosen once per sequence. The third is a token-level probability ensemble and requires logits from all active experts at every decoding step. They should not share one headline label or inference-cost claim.

### 2.4 SMEAR-style input-conditioned LoRA-update mixture

There is also an intermediate composition point between selecting one expert and mixing seven predictive distributions. Use the router weights to construct an input-conditioned adapter update,

`ΔW(x) = Σ_i q_i(x) B_i A_i`,

and perform generation with that update. This is the LoRA analogue of Soft Merging of Experts with Adaptive Routing (SMEAR), which forms one input-conditioned expert by weighting expert parameters and keeps the operation differentiable.[^25] Evaluate both game weights and frozen learned-gate weights.

This is **not** equivalent to separately averaging the `A_i` and `B_i` factors, because that introduces cross terms. The exact combined update can have rank as high as `K r`; therefore it belongs in the rank-112/full-mixture compute stratum, not the top-1 rank-16 stratum. A rank-16 truncated-SVD version is a separate lossy serving baseline. SMEAR-style composition is scientifically useful, but it does not get “one rank-16 forward for free.”

## 3. Frozen-expert learned routing

The proposed oracle-distilled classifier is excellent, but hard `argmin-NLL` labels discard almost all information in the expert-loss matrix. A router should also see whether the winning expert wins by 0.001 or by 1.0 NLL.

Train the experts once, freeze them, and score every expert on a held-out router-training set. Then compare three very small prompt-embedding routers:

- **Hard classification:** predict `argmin_e L_e(x,y)`.
- **Soft distillation:** form `q_e \propto exp[-(L_e-min L)/tau]` from mean-token NLLs and minimize `KL(q || g_phi(x))`.
- **Loss regression:** predict the full vector `(L_1,...,L_K)` with a multi-output regressor and route to the smallest predicted loss.

The soft and regression variants preserve near-ties and expert regret. A linear model should be the primary baseline because the game router is simple and geometric; a two-layer MLP can be an appendix capacity check. A k-nearest-neighbor router over the same prompt coordinates is another cheap nonlinear check.

This family is the prompt-level analogue of two-stage adapter composition. AdapterFusion first trains task adapters and then learns a separate composition module while keeping the knowledge in the adapters fixed.[^4] LoRA-Flow similarly learns a small dynamic fusion gate over existing LoRAs, although it operates at a finer token/layer granularity.[^5] The frozen-expert gate is therefore a more relevant step before full joint Switch-style training.

### Leakage-safe split

Do not generate teacher labels on the test responses. Also avoid generating labels on examples seen by any expert, because in-sample memorization can make expert membership masquerade as routability.

Use four logical partitions, implemented by splitting the current train/validation data if necessary:

1. `expert_train`: fit all LoRAs;
2. `router_train`: obtain out-of-expert-training NLL vectors and fit the gate;
3. `router_val`: choose temperature, regularization, and top-k;
4. `test`: route from the prompt only, then reveal the response solely for scoring.

Cross-fitting can recover data efficiency: train fold-specific experts to label held-out folds, train the router on the union of out-of-fold predictions, and retrain final experts on the full expert-training set.

## 4. Capacity-balanced routing

Expert-choice routing, already proposed, is one answer to load imbalance: each expert selects a fixed-capacity set of prompts, so prompts may receive a variable number of experts.[^6] A cleaner one-expert-per-prompt comparator for this project is a BASE-style balanced assignment.[^7]

For a batch of `M` prompts and `K` experts, use the same Gaussian log affinities as the game router,

\[
a_{em}=-\tfrac{1}{2}(x_e-b_m)^\top\Sigma^{-1}(x_e-b_m),
\]

but choose assignments by maximizing total affinity subject to every prompt being assigned once and every expert receiving `M/K` prompts. An exact min-cost assignment gives a hard BASE-like arm. Entropic optimal transport/Sinkhorn gives a soft arm with the same temperature or σ.

This separates three mechanisms that otherwise travel together:

- affinity geometry;
- load balancing;
- the game's `G(1-G)` strategic position update.

If balanced assignment matches the game, the likely mechanism is stable effective sample size per expert. If the game still wins, the result is much stronger because it beats a router with both matched geometry and guaranteed utilization. BASE is preferable to a generic load-balancing auxiliary loss for the main text: it guarantees balanced batch allocation and introduces no coefficient that can be tuned favorably.[^7]

## 5. A c-BTM-faithful clustering baseline

The planned hard and soft k-means arms should be retained, but defined as a small family:

1. **Balanced hard c-BTM:** balanced k-means on train-only trait coordinates, one LoRA per cluster, top-1 nearest-centroid routing.
2. **Sparse c-BTM:** the same experts, top-`k` centroid-similarity probability mixture at evaluation.
3. **Fixed-σ Gaussian-mixture EM:** responsibilities from the same Gaussian kernel and same σ as the game; centroids updated by the ordinary responsibility-weighted mean.
4. **Game method:** the same kernel and σ; positions updated using the strategic `G(1-G)` mass.

Rows 3 and 4 are the decisive theoretical contrast. Calling row 3 "fixed-σ Gaussian-mixture EM" is more precise than "soft k-means" when the assignments are normalized Gaussian responsibilities. It also makes clear that the only intended difference is the M-step/positioning objective.

Fit centroids only on the training coordinates. Fix initialization across rows 3 and 4, or report enough restarts that local optima cannot explain the difference. Match per-expert optimizer updates and either cap or explicitly report the effective weighted sample size.

## 6. AdaMix: stochastic adapters rather than semantic experts

AdaMix is meaningfully different from random equal shards. It randomly routes batches through multiple adaptation modules over training, uses consistency regularization between stochastic routes, shares part of the adaptation parameters, and averages the modules for single-module inference.[^8] Thus every module sees the overall task distribution over time; the method seeks multiple parameter-space views rather than semantic domains.

An adapted arm for this project would:

- instantiate seven rank-16 LoRA modules;
- randomly select active modules per batch or example;
- use two stochastic routes and a symmetric KL consistency term;
- train on the pooled data for the same number of optimizer tokens;
- evaluate both random single-module inference and the paper-faithful merged module.

This asks whether the benefit comes from stochastic parameter ensembling and regularization rather than trait specialization. It is a better literature comparator than another arbitrary random partition. Because the consistency loss requires an additional forward route, report its actual FLOPs separately even if active LoRA parameter count matches.

THOR is the related full-MoE precedent for stochastic routing, showing that random expert activation with consistency regularization can rival learned routing in some settings.[^9] AdaMix is the more relevant citation and implementation because it was designed for adapters and LoRA.

## 7. The one joint PEFT-MoE baseline to implement

If only one jointly trained neural MoE is feasible, use **MoLoRA** from *Pushing Mixture of Experts to the Limit*, not a generic full-parameter Switch Transformer. MoLoRA combines lightweight low-rank experts with a learned router for instruction tuning and is much closer in parameterization to this project.[^10]

Use prompt-level/per-example routing first so the comparison shares the game's decision granularity. Match:

- base model and target modules;
- total rank-equivalent stored across experts;
- number of active LoRA ranks per prompt;
- total supervised tokens and optimizer steps;
- top-k and capacity, where possible.

No single MoLoRA configuration can generally match both total stored rank and active rank when a shared router or shared expert is introduced. Report both ledgers rather than claiming one ambiguous "parameter match."

Token-level MoLoRA or LoRA-Flow can be a second, explicitly architectural comparison. It may produce better raw performance because it can change experts within a response and within layers, but then a win or loss does not isolate the game-theoretic training rule. A study of mixture-of-parameter-efficient experts on decoder-only instruction models found that gains over a single PEFT expert were inconsistent and that per-example routing could collapse at downstream evaluation, so utilization and gate entropy must be reported rather than assumed.[^11]

### 7.1 EWoRA is the cleaner parameter-budget-matched variant

EWoRA partitions one rank-`r` adapter into `n` independent rank-`r/n` sub-adapters and learns lightweight context-dependent aggregation weights. Its reported comparison is explicitly under the same total low-rank parameter budget.[^26] With `r = 112` and `n = 7`, it yields seven rank-16 components plus a small router. That directly answers a hole the proposed rank-112 generalist does not: whether **structured, jointly learned multiplicity at the same total low-rank budget** is sufficient.

This is arguably a better main capacity comparator than a generic Switch/GShard gate. It uses pooled data and does not reproduce disjoint-data specialization or strategic position updates, so a win by the game ensemble would isolate those ingredients. Its router parameters should be included in the storage ledger, and its aggregation granularity should be matched to the paper's prompt-level decision if possible.

## 8. Universal plus specialist LoRAs

The label-partitioned experts can be strengthened with a **MoDULA-Res-style** arm: train a pooled universal LoRA, train domain LoRAs separately, freeze both, and learn a router over residual domain contributions. MoDULA was designed around a three-stage separation of universal experts, domain-specific experts, and router training; its residual variant aims to preserve general capability while adding specialized behavior.[^12]

This baseline answers a different question from the rank-112 generalist. It asks whether the best inductive bias is not competition among experts but a shared path plus specialized residuals.

It is most valuable if the paper claims one of the following:

- less catastrophic forgetting;
- preservation of general instruction following;
- plug-in addition of new domains;
- improved out-of-domain behavior.

If the claims are limited to held-out NLL on the seven safety benchmarks, this can remain Tier 2 because it changes the architecture and cannot cleanly match both stored and active ranks.

## 9. Pooled-seed ensemble

A standard deep ensemble trains several copies on the same data with independent randomness and combines their predictive distributions.[^13] It is the clean answer to "does any collection of independently fine-tuned LoRAs work?" Random disjoint shards do not fully answer that question because each member receives less data.

Train seven pooled rank-16 LoRAs with different seeds and evaluate:

- mean individual NLL;
- best validation-selected member;
- uniform sequence and token probability mixtures;
- uniform random one-member inference;
- a greedy or validation-weighted model/adapter soup.

This costs roughly seven pooled runs, so it is an expensive ceiling rather than a compute-matched main arm. The current workspace already contains multiple `generalist_replay_dseed*` configurations; if those runs are completed, their checkpoints can be combined into this baseline without new training.

Model soups established weight averaging as a no-extra-inference-cost alternative to prediction ensembling for models fine-tuned from a shared initialization.[^14] For LoRAs, however, do not conflate averaging factor matrices with averaging weight updates.

## 10. Router-free merging: add a LoRA-specific method

The proposed TIES/DARE/task-arithmetic suite is salient, but it needs two controls and one LoRA-specific addition:

1. uniform linear averaging of reconstructed task deltas;
2. validation-tuned task arithmetic;
3. TIES, which trims small changes and resolves sign conflicts;[^15]
4. DARE-linear and/or DARE-TIES, because DARE is a drop-and-rescale preprocessing step rather than a complete combination rule;[^16]
5. **KnOTS + linear/TIES**, which uses SVD to align LoRA update subspaces before merging and was developed specifically because full-model merging recipes transfer imperfectly to LoRA models.[^17]

For each adapted weight, reconstruct `Delta W_e = B_e A_e` before linear, task-arithmetic, TIES, or DARE merging. Averaging the factors separately is not equivalent:

\[
\left(\frac{1}{K}\sum_e B_e\right)
\left(\frac{1}{K}\sum_e A_e\right)
\neq
\frac{1}{K}\sum_e B_eA_e.
\]

The left side introduces cross-expert terms. After merging full deltas, refactor to the desired serving rank with an SVD if a single fixed-rank LoRA is required, and report the truncation rank as an inference-capacity treatment.

## 11. Lower-priority comparisons

These are legitimate literature methods but answer weaker or more distant questions for the present paper.

### Hash or fixed random routing

Hash layers showed that routing-free deterministic hashes could match learned sparse routers in some large-model settings.[^18] At prompt level, however, a hash of prompt identity has no meaningful generalization to new examples; locality-sensitive hashing over embeddings begins to resemble clustering. Keep a fixed balanced hash as a cheap appendix null, not a headline baseline.

### DSelect-k

DSelect-k provides a continuously differentiable sparse gate and avoids the discontinuity of ordinary top-k selection.[^19] With only seven prompt-level experts, its engineering cost is unlikely to buy much additional scientific discrimination beyond a dense softmax gate and top-k MoLoRA. Use it only if gate optimization instability becomes an observed result.

### Token/layer-level LoRA fusion

LoRA-Flow and X-LoRA dynamically compose existing LoRAs at token and layer granularity.[^5][^20] They are good performance comparators but poor causal ablations because they use information and composition granularity unavailable to a one-shot prompt router. They belong in an appendix or follow-up unless the paper claims state-of-the-art adapter composition.

### Explicit diversity regularization

Contrastive or orthogonality losses can encourage different LoRA experts to learn distinct representations; MoELoRA is a direct example.[^21] This tests whether diversity alone substitutes for strategic positioning, but it adds a sensitive regularization coefficient and representation choice. Run it only after the clustering and balanced-assignment baselines establish that expert collapse is a real failure mode.

### Newer layer-allocation and rank-composition PEFT-MoEs

MoLA allocates different numbers of LoRA experts across layers, motivated by evidence that lower-layer experts are more redundant.[^27] LoRACoE dynamically constructs experts by composing task-relevant components along the LoRA rank dimension.[^28] Both are current and legitimate performance baselines, but both alter the definition and placement of an expert. They are weaker causal tests of a method whose claim concerns **which data an expert occupies and how positions update**. Cite them as recent related work; implement one only if the paper claims state-of-the-art PEFT-MoE performance. EWoRA is the more useful first implementation because it supplies the cleaner fixed-rank-budget contrast.

### Classical EM/alternating loss assignment

The original adaptive mixture-of-experts formulation jointly learns a gate and local experts by making experts compete for examples.[^22] A modern approximation would alternate between scoring each training example under all LoRAs, forming hard or soft loss-based responsibilities, and updating experts and the gate. It is conceptually strong but costs `K` model evaluations per E-step and risks rich-get-richer collapse. The frozen-expert loss router captures most of its diagnostic value first.

### Boosting and snapshot ensembles

Sequentially reweighting high-loss prompts for later LoRAs is a plausible generative analogue of boosting, and cyclic learning rates can produce snapshot ensembles from one trajectory.[^23] Neither is as close to the paper's central positioning claim as c-BTM, BASE, or MoLoRA. Snapshotting is especially hard to interpret here because rounds change routed data and positions, not only optimizer location.

## 12. Revised experiment tiers

### Tier 0: no new LoRA training

| Baseline | Exact evaluation |
|---|---|
| Correct game router | Existing expected, sampled, and argmax scores |
| Permuted game router | Distribution over LoRA-identity permutations |
| Uniform hard router | Exact expected NLL and repeated sampled realizations |
| Uniform mixture | Sequence-level mixture; token-level mixture if logits are cached |
| Global stacked mixture | Simplex weights fit only on router-validation data |
| Oracle-distilled router | Hard class, soft target, and loss-regression variants |
| DEMix-style self router | Prompt-only expert likelihood with a fitted prior/temperature |
| Hindsight oracle | Per-response `argmin` NLL, clearly labeled unattainable |

### Tier 1: causal training baselines

| Baseline | Main causal contrast |
|---|---|
| Rank-112 pooled generalist | Total trainable capacity |
| Benchmark-partitioned experts | Available domain supervision |
| Random equal shards | Number of learners and total data mass |
| Balanced hard c-BTM | Unsupervised domain discovery |
| Fixed-σ Gaussian-mixture EM | Same kernel and soft assignment, ordinary centroid M-step |
| BASE/optimal-transport assignment | Same affinity with guaranteed load balance |
| AdaMix-LoRA | Stochastic multi-view adaptation without semantic domains |

### Tier 2: named external architectures

| Baseline | When it is worth the cost |
|---|---|
| MoLoRA per-example gate | Required answer to the standard PEFT-MoE objection |
| EWoRA (`r=112`, seven rank-16 components) | A parameter-budget-matched learned low-rank mixture is required |
| MoLoRA/LoRA-Flow token gate | Only for a broader adapter-composition performance claim |
| MoDULA-Res | General-capability retention or plug-in specialization claim |
| Expert-choice routing | Variable experts per prompt and fixed expert capacity matter |
| SMEAR-style adaptive `ΔW` mixture | Parameter-space composition versus probability-space composition matters |
| TIES/DARE/KnOTS merge | Single-model, router-free serving claim |
| Pooled-seed deep ensemble | Expensive upper bound on ordinary ensembling |

## 13. Fairness and reporting protocol

### 13.1 Keep three resource ledgers

Report all three; none is a substitute for the others.

1. **Stored trainable parameters:** sum of every LoRA and router parameter.
2. **Training work:** supervised tokens processed, base-model forward/backward FLOPs, optimizer steps, and wall-clock GPU-hours.
3. **Active inference work:** active LoRA rank, number of base forward passes per token, router latency, peak memory, and total resident adapter memory.

A rank-112 generalist matches the stored parameter count of seven rank-16 LoRAs, but not necessarily their optimizer dynamics, active inference rank, or training FLOPs. A dense seven-expert probability ensemble matches stored expert capacity but requires roughly seven adapted forward paths. A top-1 router activates rank 16 but still may keep all adapters resident.

### 13.2 Match data in two ways

Report both raw example/token exposure and weighted effective sample size:

\[
\mathrm{ESS}_e=\frac{(\sum_m w_{em})^2}{\sum_m w_{em}^2}.
\]

Two experts can receive the same total loss mass but very different ESS. This is especially important when comparing soft `G`, `G(1-G)`, top-k, and unit-weight losses.

### 13.3 Use the same frozen split manifest

All partitions, clusters, centroids, gates, temperatures, merge coefficients, and early-stopping choices must be fitted without test responses. Benchmark metadata may be used only in arms explicitly labeled metadata-supervised.

Call the benchmark-label rule a **metadata router**, not simply an oracle. Reserve **hindsight oracle** for the unattainable per-response `argmin` expert. This prevents two very different information assumptions from being collapsed in figures.

### 13.4 Report route regret, not only agreement

Router classification accuracy treats all mistakes equally. Report:

\[
\mathrm{regret}(g)=\frac{1}{M}\sum_m
\left[L_{g(x_m)}(x_m,y_m)-\min_eL_e(x_m,y_m)\right].
\]

Also report oracle headroom, `L_pooled - L_oracle`, and the fraction closed by a router,

\[
\frac{L_{\mathrm{pooled}}-L_{\mathrm{router}}}
{L_{\mathrm{pooled}}-L_{\mathrm{oracle}}}.
\]

This cleanly decomposes expert quality from routing quality.

### 13.5 Report specialization and utilization

At minimum include:

- per-expert allocation mass and ESS;
- gate entropy and fraction of dead experts;
- pairwise expert NLL correlations;
- expert advantage over the pooled model by benchmark;
- router/benchmark mutual information as a diagnostic, not a target;
- performance over at least three training seeds for headline arms;
- bootstrap confidence intervals over held-out prompts, clustered by benchmark if prompts within a benchmark are not independent.

## 14. Decision rule for expensive work

The cached expert-NLL matrix can determine which expensive baseline is worth building.

1. If the hindsight oracle barely improves on the pooled rank-112 generalist, the experts are the bottleneck. Prioritize domain/c-BTM/MoDULA training, not a better gate.
2. If the oracle is strong but the game router is weak, train the frozen soft gate. If that closes most of the gap, a joint MoLoRA gate becomes scientifically relevant.
3. If correct and permuted game routers are similar, expert-position identities did not co-adapt; focus on the position update and training assignment.
4. If BASE matches the game, load balance/ESS is the likely mechanism. If the game wins at matched utilization, the strategic positioning claim is substantially strengthened.
5. If uniform mixture is strong but hard routing is weak, the experts are complementary but not separable from prompts. LoRA-Flow or a token-level fusion method becomes more relevant than top-1 routing.
6. If merged KnOTS/TIES models match routed experts, frame routing as an interpretability or modularity mechanism rather than a necessary accuracy mechanism.

## Conclusion

The most valuable addition is a **two-stage frozen-expert router trained from soft expert-loss targets**, followed closely by a **same-expert permutation null** and **BASE-style balanced assignment**. They are cheap, make few architectural assumptions, and separately test routing alignment, ordinary predictive routing, and load balance.

The planned fixed-σ clustering baseline remains the single most important training ablation. Present it as a c-BTM/fixed-covariance Gaussian-mixture comparator and keep the initialization, σ, data mass, and top-k identical to the game arm. EWoRA is the cleanest recent parameter-budget-matched learned-mixture arm; MoLoRA is the more established external PEFT-MoE architecture. Both are more defensible than a generic Switch/GShard implementation because they preserve the low-rank expert parameterization. For router-free merging, add KnOTS because LoRA factor subspaces create problems that TIES and DARE alone were not designed to solve.

## Sources

[^1]: Suchin Gururangan, Margaret Li, Mike Lewis, Weijia Shi, Tim Althoff, Noah A. Smith, and Luke Zettlemoyer. "[Scaling Expert Language Models with Unsupervised Domain Discovery](https://arxiv.org/abs/2303.14177)." 2023.
[^2]: Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A. Smith, and Luke Zettlemoyer. "[Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models](https://arxiv.org/abs/2208.03306)." 2022.
[^3]: Suchin Gururangan, Mike Lewis, Ari Holtzman, Noah A. Smith, and Luke Zettlemoyer. "[DEMix Layers: Disentangling Domains for Modular Language Modeling](https://aclanthology.org/2022.naacl-main.407/)." NAACL 2022.
[^4]: Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. "[AdapterFusion: Non-Destructive Task Composition for Transfer Learning](https://aclanthology.org/2021.eacl-main.39/)." EACL 2021.
[^5]: Hanqing Wang, Bowen Ping, Shuo Wang, Xu Han, Yun Chen, Zhiyuan Liu, and Maosong Sun. "[LoRA-Flow: Dynamic LoRA Fusion for Large Language Models in Generative Tasks](https://aclanthology.org/2024.acl-long.695/)." ACL 2024.
[^6]: Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew Dai, Zhifeng Chen, Quoc Le, and James Laudon. "[Mixture-of-Experts with Expert Choice Routing](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html)." NeurIPS 2022.
[^7]: Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer. "[BASE Layers: Simplifying Training of Large, Sparse Models](https://proceedings.mlr.press/v139/lewis21a.html)." ICML 2021.
[^8]: Yaqing Wang, Sahaj Agarwal, Subhabrata Mukherjee, Xiaodong Liu, Jing Gao, Ahmed Hassan Awadallah, and Jianfeng Gao. "[AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning](https://aclanthology.org/2022.emnlp-main.388/)." EMNLP 2022.
[^9]: Simiao Zuo, Xiaodong Liu, Jian Jiao, Young Jin Kim, Hany Hassan, Ruofei Zhang, Tuo Zhao, and Jianfeng Gao. "[Taming Sparsely Activated Transformer with Stochastic Experts](https://openreview.net/forum?id=B72HXs80q4)." ICLR 2022.
[^10]: Ted Zadouri, Ahmet Üstün, Arash Ahmadian, Beyza Ermiş, Acyr Locatelli, and Sara Hooker. "[Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE for Instruction Tuning](https://openreview.net/forum?id=EvDeiLv7qc)." ICLR 2024.
[^11]: Oleksiy Ostapenko, Lucas Caccia, Zhan Su, Nicolas Le Roux, Laurent Charlin, and Alessandro Sordoni. "[A Case Study of Instruction Tuning with Mixture of Parameter-Efficient Experts](https://openreview.net/forum?id=Rye1neGGUd)." NeurIPS 2023 workshop.
[^12]: Yufei Ma et al. "[MoDULA: Mixture of Domain-Specific and Universal LoRA for Multi-Task Learning](https://aclanthology.org/2024.emnlp-main.161/)." EMNLP 2024.
[^13]: Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. "[Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)." NeurIPS 2017.
[^14]: Mitchell Wortsman et al. "[Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy without Increasing Inference Time](https://proceedings.mlr.press/v162/wortsman22a.html)." ICML 2022.
[^15]: Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. "[TIES-Merging: Resolving Interference When Merging Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html)." NeurIPS 2023.
[^16]: Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. "[Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://icml.cc/virtual/2024/poster/33453)." ICML 2024.
[^17]: George Stoica, Pratik Ramesh, Boglarka Ecsedi, Leshem Choshen, and Judy Hoffman. "[Model Merging with SVD to Tie the Knots](https://proceedings.iclr.cc/paper_files/paper/2025/file/0d4f8a5109c5083b5307fcd0bddae7a7-Paper-Conference.pdf)." ICLR 2025.
[^18]: Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, and Jason Weston. "[Hash Layers for Large Sparse Models](https://arxiv.org/abs/2106.04426)." 2021.
[^19]: Hussein Hazimeh et al. "[DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html)." NeurIPS 2021.
[^20]: Rafael Buehler and Markus J. Buehler. "[X-LoRA: Mixture of Low-Rank Adapter Experts](https://arxiv.org/abs/2402.07148)." 2024.
[^21]: Xun Wu et al. "[MoELoRA: Contrastive Learning Guided Mixture of Experts on Parameter-Efficient Fine-Tuning for Large Language Models](https://arxiv.org/abs/2402.12851)." 2024.
[^22]: Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. "[Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)." *Neural Computation* 3(1), 1991.
[^23]: Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, John E. Hopcroft, and Kilian Q. Weinberger. "[Snapshot Ensembles: Train 1, Get M for Free](https://arxiv.org/abs/1704.00109)." ICLR 2017.
[^24]: Damjan Kalajdzievski. "[A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732)." 2023.
[^25]: Mohammed Muqeeth, Haokun Liu, and Colin Raffel. "[Soft Merging of Experts with Adaptive Routing](https://arxiv.org/abs/2306.03745)." *Transactions on Machine Learning Research*, 2024.
[^26]: Harsh Kohli, Helian Feng, Lenon Minorics, Bhoomit Vasani, Xin He, and Ali Kebarighotbi. "[EWoRA: Expert Weighted Low-Rank Adaptation for Heterogeneous Data](https://aclanthology.org/2025.findings-ijcnlp.108/)." Findings of IJCNLP-AACL 2025.
[^27]: Chongyang Gao et al. "[MoLA: MoE LoRA with Layer-wise Expert Allocation](https://aclanthology.org/2025.findings-naacl.284/)." Findings of NAACL 2025.
[^28]: Guanyu Li, Zhiheng Xi, Zhihao Zhang, Boyang Hong, Tao Gui, Qi Zhang, and Xuanjing Huang. "[LoRACoE: Improving Large Language Model via Composition-based LoRA Expert](https://aclanthology.org/2025.emnlp-main.1594/)." EMNLP 2025.
