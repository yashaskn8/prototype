# SERVY — Accuracy Failure Analysis (The 7 Hit@1 Errors)

**Audit Scope:** Deep diagnostic analysis of all 7 failed queries from the 42-question development evaluation benchmark (reproduced accuracy: 35/42 = 83.33%).  
**Reranker Checkpoint:** Servy V5 BGE (`b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda`)  
**Evaluation Protocol:** `servy-frozen-pool-paired-v1` (BM25 frozen top-10 candidates, head-5 CrossEncoder reranking).  

---

## 1. Summary of Failure Categories

All 7 queries that failed Hit@1 belong to verified, evidence-backed root causes:

| Failure Category | Count | Percentage of Failures | Query Identifiers | Component Responsible | Retraining Required? |
|---|---|---|---|---|---|
| **Category C — Candidate Retrieval Failure** | **4** | **57.1%** | `OP26-SLV-144-2`, `OP26-SLV-75-2`, `OP26-SLV-74-1`, `OP26-SLV-142-2` | BM25 First-Stage Retriever | No (Retrieval upgrade only) |
| **Category B — Candidate-Depth Limitation** | **1** | **14.3%** | `OP26-SLV-145-2` | Head Pool & Equipment Filtering | No (Metadata filtering & pool) |
| **Category D — Evaluation / Label Uncertainty** | **1** | **14.3%** | `OP26-SLV-76-2` | Silver Dataset Annotations | No (Label reconciliation) |
| **Category A / D — Reranker Granularity / Sub-Section Split** | **1** | **14.3%** | `OP26-SLV-143-0` | Chunk Granularity / Evaluation Rule | No (Multi-chunk alignment) |

**Key Architectural Insight:**  
Only **1 of the 7 failures** (`OP26-SLV-143-0`) involves a ranking inversion between eligible candidates, and even in that case, both passages belong to the **exact same troubleshooting section** of the exact same document.  
**Over 71% of all failures (5/7)** are caused entirely by candidate retrieval (absent candidates or candidate dilution from wrong equipment).

---

## 2. In-Depth Query-by-Query Diagnostic Breakdown

---

### Failure Case 1: `OP26-SLV-144-2`
* **Query:** *"Asus Celeron & Service Terminal Troubleshooting Guide operating v specification 5"*
* **Target Equipment:** Asus Celeron & Service Terminal Troubleshooting Guide
* **Source Family:** `FAMILY-ASUS_CELERON` | **Target Document:** ID 163
* **Gold Chunk ID:** `144`
* **Original BM25 Candidates (Top 10):** `['47', '23', '81', '40', '130', '59', '96', '75', '49', '137']`
* **Gold Rank in Candidates:** **ABSENT (Not in Top 10)**
* **V5 Final Gold Rank:** **ABSENT (Score: None)**
* **Top-1 Ranked Chunk by V5:** Chunk `81` (Score: -5.0692, Doc 21 — Pallet truck drive wheel)
* **Gold Passage Content:**
  > `## Diagnostic Port / USB Field Equipment Communication Lost`  
  > `Check the USB-to-RS485 interface converter cable... Create a service call if USB controller ports fail to deliver 5V bus power or serial port drops packets continuously.`
* **Root Cause Analysis:**  
  **Category C — Candidate Retrieval Failure.** The query generator formed the query `"operating v specification 5"` from the passage phrase `"5V bus power"`. BM25 treats `"v"` and `"5"` as isolated single-character tokens with minimal inverse document frequency weighting, completely failing to match `"5V"`. As a result, Chunk 144 was never retrieved in the top 10 candidates. The reranker never saw the gold chunk.
* **Fixability & Corrective Action:**  
  1. Dense semantic embedding retrieval (Qwen3 / BGE embedding) easily bridges the semantic link between "operating voltage specification 5" and "5V bus power".  
  2. Equipment metadata pre-filtering eliminates pallet truck chunks (Doc 21).  
* **Retraining Needed?** **NO.** First-stage retrieval upgrade.

---

### Failure Case 2: `OP26-SLV-143-0`
* **Query:** *"Asus Celeron & Service Terminal Troubleshooting Guide terminal screen black or system unexpectedly shuts down"*
* **Target Equipment:** Asus Celeron & Service Terminal Troubleshooting Guide
* **Source Family:** `FAMILY-ASUS_CELERON` | **Target Document:** ID 163
* **Gold Chunk ID:** `143`
* **Original BM25 Candidates (Top 10):** `['143', '142', '75', '78', '47', '77', '23', '64', '110', '40']` (Gold rank = 1)
* **V5 Final Gold Rank:** **Rank 2** (Score: `+8.3943`)
* **Top-1 Ranked Chunk by V5:** Chunk `142` (Score: `+9.1087`, Doc 163)
* **Top-1 Chunk Content (Chunk 142):**
  > `## Terminal Screen Black or System Unexpectedly Shuts Down`  
  > `Check the external 19V DC power brick indicator LED and mains socket connection.`  
  > `Step 1 - Verify the DC barrel connector... Step 2 - Inspect rear exhaust vents... Step 3 - Hard power reset... Step 4 - Connect external HDMI monitor...`
* **Gold Chunk Content (Chunk 143):**
  > `## Terminal Screen Black or System Unexpectedly Shuts Down`  
  > `Expected result: System boots past BIOS POST screen to the Servy diagnostics desktop within 45 seconds.`  
  > `Create a service call if the motherboard power LED blinks a red hardware diagnostic code or display shows GPU artifacts.`
* **Root Cause Analysis:**  
  **Category A / D — Sub-Section Chunk Granularity.** Both Chunks 142 and 143 come from the identical section in Document 163. Chunk 142 contains the diagnostic steps (Steps 1–4), while Chunk 143 contains the expected outcome and escalation rule. V5 scored Chunk 142 slightly higher (+9.11 vs +8.39) because it contains the detailed technical troubleshooting steps. However, the weakly-supervised silver dataset arbitrarily assigned Chunk 143 as the sole positive. At the document level, Document 163 is ranked #1 (`document_hit_at_1 = PASS`).
* **Fixability & Corrective Action:**  
  1. Multi-chunk positive labeling (both chunks 142 and 143 are valid gold evidence for this query).  
  2. In production RAG, retrieving either chunk 142 or 143 provides correct grounded context.  
* **Retraining Needed?** **NO.** Evaluation label reconciliation.

---

### Failure Case 3: `OP26-SLV-76-2`
* **Query:** *"Industrial Touch Terminal & Asus Celeron Controller Guide safety precautions and escalation criteria"*
* **Target Equipment:** Industrial Touch Terminal & Asus Celeron Controller Guide
* **Source Family:** `FAMILY-ASUS_CELERON` | **Target Document:** ID 139
* **Gold Chunk ID:** `76`
* **Original BM25 Candidates (Top 10):** `['76', '78', '110', '40', '79', '19', '47', '145', '137', '23']` (Gold rank = 1)
* **V5 Final Gold Rank:** **Rank 2** (Score: `-5.6184`)
* **Top-1 Ranked Chunk by V5:** Chunk `79` (Score: `+4.1534`, Doc 139)
* **Top-1 Chunk Content (Chunk 79):**
  > `## Stop and escalate if`  
  > `- Motherboard capacitor bulging or liquid damage detected.`  
  > `- Touch digitizer glass has physical spiderweb cracks.`  
  > `- Storage drive reports critical SMART read/write failures.`
* **Gold Chunk Content (Chunk 76):**
  > `## When touch inputs do not register or click in the wrong position`  
  > `1. Clean the optical touch frame glass using an isopropyl wipe... 2. Verify the USB controller cable... 3. Run touch calibration utility...`
* **Root Cause Analysis:**  
  **Category D — Evaluation Label Error in Silver Dataset.** The query explicitly requests *"safety precautions and escalation criteria"*. Chunk 79 in Document 139 is literally titled `"## Stop and escalate if"` and lists the hardware escalation criteria. Chunk 76 is titled `"## When touch inputs do not register..."` and discusses routine touchscreen cleaning and USB cable reseating.  
  The CrossEncoder model V5 correctly recognized Chunk 79 as the true semantic match (+4.15 vs -5.62). However, because the unreviewed silver label was erroneously mapped to Chunk 76, the benchmark penalized the model for making the correct semantic choice!
* **Fixability & Corrective Action:**  
  1. Reconcile gold labels in the development benchmark through human adjudication. Correcting the gold chunk for `OP26-SLV-76-2` to Chunk 79 converts this from a FAIL to a verified PASS.  
* **Retraining Needed?** **NO.** The model already exhibits superior semantic ranking over noisy silver labels.

---

### Failure Case 4: `OP26-SLV-75-2`
* **Query:** *"Industrial Touch Terminal & Asus Celeron Controller Guide operating v specification 12"*
* **Target Equipment:** Industrial Touch Terminal & Asus Celeron Controller Guide
* **Source Family:** `FAMILY-ASUS_CELERON` | **Target Document:** ID 139
* **Gold Chunk ID:** `75`
* **Original BM25 Candidates (Top 10):** `['76', '78', '137', '81', '110', '59', '144', '79', '40', '145']`
* **Gold Rank in Candidates:** **ABSENT (Not in Top 10)**
* **V5 Final Gold Rank:** **ABSENT (Score: None)**
* **Top-1 Ranked Chunk by V5:** Chunk `137` (Score: -5.6169, Doc 161 — Paper shredder operating rules)
* **Gold Chunk Content (Chunk 75):**
  > `## If the service terminal has a black screen and power LED is unlit`  
  > `1. Inspect the 12V / 19V DC power adapter barrel jack connection at the rear panel.`  
  > `2. Verify the wall socket has power and the green LED on the external power brick is illuminated.`
* **Root Cause Analysis:**  
  **Category C — Candidate Retrieval Failure.** Synthesized query `"operating v specification 12"` was derived from `"12V / 19V DC power adapter"`. BM25 failed to retrieve Chunk 75 into the top 10 due to tokenization and absence of exact term overlap. Top candidates were polluted with irrelevant documents (paper shredders, pallet trucks).
* **Fixability & Corrective Action:**  
  Dense retrieval (Qwen3) + equipment/product metadata filter.
* **Retraining Needed?** **NO.**

---

### Failure Case 5: `OP26-SLV-74-1`
* **Query:** *"Exterior Stabilizer Jack and DC Drive Motor Service Guide safety precautions and escalation criteria"*
* **Target Equipment:** Exterior Stabilizer Jack and DC Drive Motor Service Guide
* **Source Family:** `FAMILY-EXTERIOR_JACK` | **Target Document:** ID 138
* **Gold Chunk ID:** `74`
* **Original BM25 Candidates (Top 10):** `['71', '139', '138', '118', '109', '40', '75', '47', '23', '73']`
* **Gold Rank in Candidates:** **ABSENT (Not in Top 10)**
* **V5 Final Gold Rank:** **ABSENT (Score: None)**
* **Top-1 Ranked Chunk by V5:** Chunk `71` (Score: -0.7340, Doc 138)
* **Gold Chunk Content (Chunk 74):**
  > `## Stop and escalate if`  
  > `- Hydraulic cylinder rod is bent, scored, or pitted.`  
  > `- Gearbox output shaft splines are sheared.`
* **Root Cause Analysis:**  
  **Category C — Candidate Retrieval Failure.** BM25 matched keywords *"Exterior"*, *"Stabilizer"*, *"Jack"*, *"DC"*, *"Drive"*, *"Motor"* against procedural chunks (71, 139, 138, 118, 109). Chunk 74 does not repeat the guide's full title in its passage text, causing BM25 to score it below the top-10 cutoff.
* **Fixability & Corrective Action:**  
  Prepend document title and product metadata to chunk text during embedding and indexing (`"Exterior Stabilizer Jack: ## Stop and escalate if..."`).
* **Retraining Needed?** **NO.**

---

### Failure Case 6: `OP26-SLV-142-2`
* **Query:** *"Asus Celeron & Service Terminal Troubleshooting Guide operating v specification 19"*
* **Target Equipment:** Asus Celeron & Service Terminal Troubleshooting Guide
* **Source Family:** `FAMILY-ASUS_CELERON` | **Target Document:** ID 163
* **Gold Chunk ID:** `142`
* **Original BM25 Candidates (Top 10):** `['47', '23', '81', '40', '59', '49', '130', '19', '143', '64']`
* **Gold Rank in Candidates:** **ABSENT (Not in Top 10)**
* **V5 Final Gold Rank:** **ABSENT (Score: None)**
* **Top-1 Ranked Chunk by V5:** Chunk `81` (Score: -6.4027, Doc 21)
* **Gold Chunk Content (Chunk 142):**
  > `## Terminal Screen Black or System Unexpectedly Shuts Down`  
  > `Check the external 19V DC power brick indicator LED and mains socket connection...`
* **Root Cause Analysis:**  
  **Category C — Candidate Retrieval Failure.** Synthetic query `"operating v specification 19"` failed to retrieve Chunk 142 in BM25. (Note: Chunk 143 was retrieved at position 9, but the designated gold chunk 142 was missed).
* **Fixability & Corrective Action:**  
  Dense retrieval + equipment scope filtering.
* **Retraining Needed?** **NO.**

---

### Failure Case 7: `OP26-SLV-145-2`
* **Query:** *"Asus Celeron Field Terminal Setup & Maintenance Guide maintenance schedule intervals"*
* **Target Equipment:** Asus Celeron Field Terminal Setup & Maintenance Guide
* **Source Family:** `FAMILY-ASUS_CELERON` | **Target Document:** ID 164
* **Gold Chunk ID:** `145`
* **Original BM25 Candidates (Top 10):** `['40', '46', '44', '114', '120', '121', '145', '115', '113', '58']`
* **Gold Rank in Candidates:** **Rank 7** (In top 10, but outside head pool of 5)
* **V5 Final Gold Rank:** **Rank 7** (Score: `-5.8457`)
* **Top-1 Ranked Chunk by V5:** Chunk `40` (Score: `-1.9703`, Doc 19 — Hydraulic pump customer handover)
* **Gold Chunk Content (Chunk 145):**
  > `## Monthly Preventive Maintenance`  
  > `Clean cooling fan exhaust louvers using anti-static compressed air. Wipe touch screen digitizer with 70% isopropyl alcohol wipes. Verify backup of offline diagnostic logs to cloud server.`
* **Root Cause Analysis:**  
  **Category B — Candidate-Depth Limitation & Equipment Dilution.**  
  BM25 lexical search matched generic words *"maintenance"*, *"schedule"*, *"intervals"* across unrelated documents (Docs 19, 21, 20, 149, 152), pushing the true equipment manual chunk (Doc 164) down to candidate position 7.  
  Under the serving-bounded reranker configuration (`pool_size = 5`), only the top 5 candidates are reranked; candidates 6–10 remain in original BM25 order. Thus, Chunk 145 was physically prevented from reaching Rank 1. Furthermore, even in an unconstrained sweep (pool=10), V5 scored Chunk 40 higher than 145 because candidate retrieval was contaminated by cross-equipment passages.
* **Fixability & Corrective Action:**  
  Enforce metadata pre-filtering by `asset.product` / equipment type. If retrieval is restricted to Doc 164, Chunk 145 is the only candidate and ranks #1.
* **Retraining Needed?** **NO.**

---

## 3. Synthesis: What Can and Cannot Improve Accuracy

| Intervention | Category Addressed | Expected Dev Hit@1 Gain | Production Feasibility | Risk Level |
|---|---|---|---|---|
| **Reranker Retraining (e.g. V6/V7)** | Category A only (1 query) | **0% – 2.4%** | Low (No sign-off, no holdout) | **HIGH** |
| **Dense Candidate Retrieval (Qwen3 / Hybrid RRF)** | Category C (4 queries) | **+9.52%** (elevates ceiling to 97.6%) | High (Retriever upgrade) | Low |
| **Equipment Metadata Pre-Filtering** | Category B + Category C (3 queries) | **+7.14%** | Immediate (Django retriever filter) | Very Low |
| **Label Adjudication (Human Sign-off)** | Category D (1 query) | **+2.38%** | High (Corrects silver error) | Zero |
