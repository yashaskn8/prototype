# SERVY — GitHub Branch & Pull Request Audit (`SERVY_GITHUB_BRANCH_AUDIT.md`)

**Audit Date:** 2026-10-29  
**Repository:** `https://github.com/yashaskn8/prototype`  
**Current `main` HEAD:** `b602587c42359908e20bcc88428cd7d4c012a732` (includes PR #16 squash-merged)

---

## 1. PR #16 Verification & Merge Status

| Field | Value |
|---|---|
| **Pull Request** | [#16 — Require real serving-path scoring in V5 verification and startup gate](https://github.com/yashaskn8/prototype/pull/16) |
| **Source Branch** | `feature/servy-v5-serving-verification-20261025` (HEAD `0fb8447`) |
| **CI Check 1** | `Backend Security & Tests (MySQL 8.0)` — **SUCCESS** |
| **CI Check 2** | `Frontend Build & Lint` — **SUCCESS** |
| **Exception Assertion Fix** | Verified in [`core/tests_v5_activation.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/core/tests_v5_activation.py#L126-L130) (`Required V5 reranker failed to load`) |
| **Merge Status** | **MERGED** (`2026-10-29T07:53:46Z`) via squash commit `b602587c42359908e20bcc88428cd7d4c012a732` |

---

## 2. Complete Remote Branch Audit

All 12 remote feature/fix branches on `origin` were audited against `origin/main` (`b602587`) and cross-referenced with all 16 historical Pull Requests (`#1` through `#16`). Because this repository uses **Squash and Merge**, each feature branch tip has its own commit hash while its exact file diff is already present on `main`.

| Remote Branch | Branch Tip SHA | Associated PR | PR State | Squash Merge Commit on `main` | Merged At (UTC) | Diff vs PR Squash Commit | Safe to Delete |
|---|---|---|---|---|---|---|---|
| `origin/feature/servy-v5-serving-verification-20261025` | `0fb8447` | **PR #16** | `MERGED` | `b602587` | `2026-10-29T07:53:46Z` | Identical (`4 files changed, 189+, 16-`) | **YES** |
| `origin/feature/servy-v5-verified-checkpoint-install` | `91573f3` | **PR #15** | `MERGED` | `ffdbdb1` | `2026-10-20T14:21:11Z` | Identical (`9 files changed, 487+, 9-`) | **YES** |
| `origin/fix/servy-v5-recall-serving-hardening` | `e4214eb` | **PR #14** | `MERGED` | `1cb3ba9` | `2026-10-20T08:11:36Z` | Identical (`7 files changed, 526+, 12-`) | **YES** |
| `origin/fix/v6-negative-only-holdout-categories-20261020` | `785e707` | **PR #13** | `MERGED` | `bbca266` | `2026-10-20T05:31:02Z` | Identical (`3 files changed, 98+, 18-`) | **YES** |
| `origin/fix/v6-holdout-empty-vs-unresolved-gold-20261020` | `32f1db9` | **PR #12** | `MERGED` | `e5c452d` | `2026-10-20T04:55:21Z` | Identical (`3 files changed, 71+, 12-`) | **YES** |
| `origin/feature/v6-holdout-leakage-audit-20261020` | `5eb6f1f` | **PR #11** | `MERGED` | `e8a431b` | `2026-10-19T18:55:19Z` | Identical (`3 files changed, 495+, 0-`) | **YES** |
| `origin/feature/v6-continuation-readiness-20261019` | `071f23c` | **PR #10** | `MERGED` | `258b66f` | `2026-10-19T17:17:43Z` | Identical (`6 files changed, 991+, 0-`) | **YES** |
| `origin/fix/v5-private-artifact-audit-20261019` | `8b715f9` | **PR #9** | `MERGED` | `3d2ba3d` | `2026-10-19T13:57:50Z` | Identical (`4 files changed, 577+, 0-`) | **YES** |
| `origin/feature/v5-reranker-provenance-20261019` | `1a0fb43` | **PR #8** | `MERGED` | `664fc25` | `2026-10-19T12:12:48Z` | Identical (`6 files changed, 377+, 17-`) | **YES** |
| `origin/feature/qwen3-accuracy-preflight-20261019` | `b499260` | **PR #7** | `MERGED` | `c8a8a93` | `2026-10-19T10:34:57Z` | Identical (`9 files changed, 987+, 21-`) | **YES** |
| `origin/fix/servy-accuracy-redteam-20261019` | `1ebfd32` | **PR #6** | `MERGED` | `781f90c` | `2026-10-19T06:51:43Z` | Identical (`11 files changed, 813+, 43-`) | **YES** |
| `origin/feature/servy-accuracy-evidence-gates` | `68d3043` | **PR #5** | `MERGED` | `b1ab286` | `2026-10-19T05:24:38Z` | Identical (`14 files changed, 2610+, 9-`) | **YES** |

---

## 3. Earlier Pull Requests (`#1`–`#4`)

| PR | Title | State | Squash Merge Commit | Remote Branch Status |
|---|---|---|---|---|
| **PR #4** | Harden V4 research artifact preparation and Colab runner | `MERGED` | `66cd4fa` | Already deleted |
| **PR #3** | Add V4 mentor experiment and AI evidence auditor | `MERGED` | `e7a0691` | Already deleted |
| **PR #2** | Harden BGE indexing, retrieval gates, and evaluation | `MERGED` | `f446322` | Already deleted |
| **PR #1** | Initial hardened Servy RAG baseline | `MERGED` | `be0e247` | Already deleted |

---

## 4. Branch Cleanup Commands

Every remote branch listed in Section 2 has been squash-merged into `main` with zero unmerged commits. They can be pruned at any time using:

```powershell
git push origin --delete `
  feature/servy-v5-serving-verification-20261025 `
  feature/servy-v5-verified-checkpoint-install `
  fix/servy-v5-recall-serving-hardening `
  fix/v6-negative-only-holdout-categories-20261020 `
  fix/v6-holdout-empty-vs-unresolved-gold-20261020 `
  feature/v6-holdout-leakage-audit-20261020 `
  feature/v6-continuation-readiness-20261019 `
  fix/v5-private-artifact-audit-20261019 `
  feature/v5-reranker-provenance-20261019 `
  feature/qwen3-accuracy-preflight-20261019 `
  fix/servy-accuracy-redteam-20261019 `
  feature/servy-accuracy-evidence-gates
```
