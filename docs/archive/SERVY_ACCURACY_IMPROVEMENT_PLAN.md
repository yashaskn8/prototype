# SERVY — Accuracy Improvement Plan (Next Development Phase)

**Author:** Principal Machine Learning Engineer, Information Retrieval Researcher, RAG Systems Architect  
**Baseline Model:** Servy V5 BGE CrossEncoder (`b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda`)  
**Baseline Verified Accuracy:** 35/42 = **83.33% Hit@1**, 37/42 = **88.10% Hit@5**, Candidate Recall@10 = **90.48%**  
**Ceiling Constraint:** Head-5 Top-One Oracle Ceiling on BM25 = **88.10%**  

---

## Executive Strategic Guidance

Before any engineering effort is spent on fine-tuning a "V7" reranker, our empirical audit establishes three foundational facts:
1. **The Primary Bottleneck is Candidate Retrieval, Not Reranker Capacity:** 4 out of 7 failures (57.1%) occur because the gold passage is completely absent from the top 10 candidates. A reranker cannot rank what the retriever failed to return.
2. **Offline Pool-Size Sweep Proves Pool Truncation is Not the Blocker:** Expanding the reranker pool from 5 to 10 on BM25 candidates produces **0.00% gain in Hit@1** (remains 35/42 = 83.33%).
3. **Model Retraining Without New Data is Blocked by Governance & Leakage:** The historical holdout has 10/10 source-family leakage, and the 762-example training package lacks authorized human sign-off (`PENDING_AUTHORIZED_SIGN_OFF`).

Therefore, our improvement strategy prioritizes:
1. **First-Stage Candidate Retrieval Upgrade (Dense & Hybrid)**
2. **Equipment & Asset Metadata Pre-Filtering**
3. **Passage Chunking & Section Prepending**
4. **Silver Label Adjudication & Evaluation Governance**
5. **Separately Gated Reranker Training (Only with New Sources)**

---

## Prioritized Improvement Experiments

---

### Experiment 1: Dense Semantic Candidate Retrieval (Qwen3-Embedding & Hybrid RRF)
* **Priority:** **P1 (Highest Impact)**
* **Hypothesis:** Upgrading candidate retrieval from BM25 lexical search to Qwen3-0.6B dense semantic embeddings (or reciprocal rank fusion hybrid BM25 + dense) will bridge token mismatches (e.g., `"operating v specification 5"` $\leftrightarrow$ `"5V bus power"`), recovering all 4 Category C misses and elevating candidate Recall@10 from 90.48% to $\ge 97.6\%$.
* **Existing Baseline:** BM25 lexical candidate retrieval. Recall@10: 38/42 = 90.48%. Top-1 Oracle Ceiling: 88.10%.
* **Exact Implementation Change:**
  - In `core/services/retriever.py` and offline benchmark runner: implement reciprocal rank fusion ($RRF_{60}$) combining dense cosine similarity from Qwen3-Embedding-0.6B with BM25 lexical scores.
  - Query embedding instruction prefix: `"Represent this query for retrieving relevant service documentation: "`.
* **Required Inputs:**
  - `servy_corpus_recovered_20261018T150933Z.jsonl`
  - Qwen3-Embedding-0.6B model weights
  - `dev_unapproved_candidate.jsonl`
* **Controlled Evaluation Method:**
  - Generate new frozen candidates via Qwen3 / Hybrid on the identical 42 development queries.
  - Replay verified V5 CrossEncoder with identical `pool_size = 5`.
* **Success Criteria:**
  - Candidate Recall@10 increases from 38/42 (90.48%) to $\ge 41/42$ (97.62%).
  - V5 Hit@1 increases from 35/42 (83.33%) to $\ge 38/42$ (90.48%).
* **Regression Safeguards:**
  - Verify tenant isolation filter executes before vector similarity scoring.
  - Candidate generation latency must remain $< 50\text{ ms}$ on GPU / $< 200\text{ ms}$ on CPU.
* **GPU Required?** Yes (for embedding encoding).
* **Retraining Required?** **NO.** Uses off-the-shelf embedding model and existing verified V5 checkpoint.

---

### Experiment 2: Equipment and Asset Metadata Pre-Filtering
* **Priority:** **P2 (Immediate Architecture Improvement)**
* **Hypothesis:** Pre-filtering the retrieval candidate space by equipment / product category (e.g., restricting retrieval to `FAMILY-ASUS_CELERON` when the query or customer context specifies the terminal) will eliminate cross-equipment dilution, resolving Category B (`OP26-SLV-145-2`) and preventing wrong-equipment distractors from entering the top 5.
* **Existing Baseline:** Global tenant-wide search without equipment scoping. Candidates for terminal queries get diluted with hydraulic pump and paper shredder chunks.
* **Exact Implementation Change:**
  - In `core/services/retriever.py`, update `retrieve()`: When `asset` or `product` is provided in the service call or user turn, add `{"product_id": asset.product_id}` or `{"equipment": asset.equipment_name}` to Chroma / candidate search filter.
* **Required Inputs:**
  - Asset-to-Document metadata associations in `core_knowledgedocument` and `core_asset`.
* **Controlled Evaluation Method:**
  - Re-evaluate candidate generation on the 42 dev queries with equipment-restricted search.
  - Measure candidate ranks of gold chunks and re-score with V5.
* **Success Criteria:**
  - Query `OP26-SLV-145-2` gold chunk moves from candidate rank 7 to rank 1.
  - Zero cross-equipment distractors in the top 5 candidates.
  - Hit@1 increases by +1 query (+2.38%).
* **Regression Safeguards:**
  - If a query has no matching product document, gracefully fall back to tenant-wide search.
  - Synthetic test covering multi-equipment customer environments.
* **GPU Required?** No.
* **Retraining Required?** **NO.** Pure retrieval metadata filtering.

---

### Experiment 3: Title and Metadata Prepending in Passage Chunking
* **Priority:** **P3 (High Value / Low Complexity)**
* **Hypothesis:** Prepending document title and section hierarchy to each text chunk (e.g. `"[Doc: Exterior Stabilizer Jack] ## Stop and escalate if..."`) ensures that generic section headings contain the equipment context, resolving lexical and dense retrieval misses like `OP26-SLV-74-1`.
* **Existing Baseline:** Raw heading and body text chunking without document-level title prepending.
* **Exact Implementation Change:**
  - In `core/services/indexing.py` and corpus generation: format chunk text as:  
    `"{doc.title} > {section_heading}\n{body_text}"`
* **Required Inputs:**
  - Raw document corpus.
* **Controlled Evaluation Method:**
  - Re-index the corpus with title prepending; regenerate BM25 and dense indices; evaluate candidate recall on 42 dev queries.
* **Success Criteria:**
  - `OP26-SLV-74-1` gold chunk moves into the top 5 candidates.
* **Regression Safeguards:**
  - Ensure token length does not truncate substantive body text within the 256-token limit.
* **GPU Required?** No.
* **Retraining Required?** **NO.**

---

### Experiment 4: Benchmark Label Adjudication & Multi-Chunk Evidence Grouping
* **Priority:** **P4 (Essential for Measurement Integrity)**
* **Hypothesis:** Human review and adjudication of the silver development labels will eliminate artificial penalties where the model makes a semantically superior choice (e.g., `OP26-SLV-76-2` choosing escalation criteria Chunk 79 instead of touch cleaning Chunk 76) or where multiple consecutive chunks contain the complete answer (`OP26-SLV-143-0` chunks 142 & 143).
* **Existing Baseline:** Weakly supervised silver labels with single-chunk gold annotations, unreviewed by human engineers.
* **Exact Implementation Change:**
  - Conduct formal engineering review of the 7 failure cases.
  - Allow multi-chunk gold lists where consecutive or co-relevant chunks provide the supporting answer.
  - Update `dev_unapproved_candidate.jsonl` with `review_status="expert_adjudicated"`.
* **Required Inputs:**
  - Subject matter expert (SME) / field technician review package.
* **Controlled Evaluation Method:**
  - Re-evaluate V5 on the adjudicated benchmark.
* **Success Criteria:**
  - Elimination of false negative evaluations on `OP26-SLV-76-2` and `OP26-SLV-143-0`.
  - Development Hit@1 increases by +2 queries (+4.76%) to 37/42 = **88.10%** under faithful semantic evaluation.
* **Regression Safeguards:**
  - Keep full audit trail of all label modifications with reviewer identity and justification.
  - Do not alter the unadjudicated historical benchmark; save adjudicated set as `dev_adjudicated_v1.jsonl`.
* **GPU Required?** No.
* **Retraining Required?** **NO.**

---

### Experiment 5: Defensible V7 Reranker Fine-Tuning (Strictly Gated)
* **Priority:** **P5 (Deferred until P1–P4 are complete and new sources acquired)**
* **Hypothesis:** Fine-tuning BGE CrossEncoder on hard negatives mined from the new hybrid retriever will improve distinction between subtle procedural steps and failure symptoms.
* **Existing Baseline:** Verified V5 checkpoint (`b1cf23ba...`).
* **Prerequisites & Gates (Mandatory Before Retraining):**
  1. **Organizational Approval:** Formal written sign-off from data owner for cloud processing (`PENDING_AUTHORIZED_SIGN_OFF` cleared).
  2. **Unseen Source Families:** Acquisition of at least 5 new, distinct equipment families disjoint from the 18 historical corpus families.
  3. **Leakage-Free Independent Holdout:** Creation of a verified untouched evaluation benchmark passing `tools/servy_v6_holdout_audit.py` with `independent_holdout_verified=True`.
* **Controlled Evaluation Method:**
  - Train V7 initialized from V5 (not from scratch).
  - Pinned `sentence-transformers==5.1.0`.
  - Max 2 epochs, learning rate $1\times 10^{-5}$, warmup ratio 0.1.
  - Checkpoint selection based strictly on the independent holdout, NOT the development set.
* **Success Criteria:**
  - Statistically significant improvement on the independent holdout ($\ge +3.0\%$ Hit@1 at $p < 0.05$).
  - Zero regression on existing verified V5 development set.
* **GPU Required?** Yes (CUDA L4 / A100).
* **Retraining Required?** Yes.

---

## 3. Projected Accuracy Roadmap

| Development Milestone | Interventions Applied | Projected Candidate Recall@10 | Projected Dev Hit@1 | Cumulative Gain |
|---|---|---|---|---|
| **Current Baseline (Verified)** | V5 BGE + BM25 Lexical + Pool 5 | 90.48% (38/42) | **83.33% (35/42)** | Baseline |
| **Phase 1: Hybrid Retrieval** | Qwen3 Dense + BM25 ($RRF_{60}$) | 97.62% (41/42) | **90.48% (38/42)** | **+7.15%** |
| **Phase 2: Metadata & Prepending** | Equipment pre-filtering + Title prepend | 100.0% (42/42) | **92.86% (39/42)** | **+9.53%** |
| **Phase 3: Label Adjudication** | SME review of silver labels (76-2, 143-0) | 100.0% (42/42) | **97.62% (41/42)** | **+14.29%** |
| **Phase 4: Gated V7 Reranker** | Fine-tuned on new sources + hard negatives | 100.0% (42/42) | **$\ge 97.6\%$ (Dev) / $\ge 90\%$ (Holdout)** | Independent Generalization |
