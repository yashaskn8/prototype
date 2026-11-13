# Servy customer support release report — 2026-11-05

**Final release state: BLOCKED. Mandatory gates PASS 0 / 15.** This is a code-change checkpoint, not a staging verification, release certification, or production deployment.

## Scope and version

GitHub: https://github.com/yashaskn8/prototype
Target branch: `main`
Baseline commit before this work: `5f8f66f2a0e2c8e57702ef81d789e8159943b33c`
Latest audited implementation commit: `38bd0999c05172908316aa5b493ad367698e0c7f`
Environment: GitHub connector access; **no** direct access to the original Windows process, private approved manuals, local SELECT-only MySQL identity, local Ollama model, production systems or browser session.
Corpus/index/model versions: not inspectable from this environment; therefore not verified.

## Changes on main

1. `RELEASE_CONTRACT_2026-11-05.md`: 15 evidence-gated mandatory requirements, explicit BLOCKED/NOT RUN semantics.
2. `SERVY_RAG_ADR_2026-11-05.md`: checked official architecture sources, alternative decisions, capability coverage matrix and deferrals.
3. `tools/servy_real_demo_dialogue.py`: `vague_problem_clarification()` asks a focused question for generic failure reports unless signed prior customer turns already describe a specific issue; recognizes recurrence as contextual follow-up.
4. `tools/servy_real_demo.py`: routes nonspecific failure to a clarification before source retrieval while maintaining existing hazard/out-of-scope priority.
5. `tools/test_servy_real_demo_dialogue.py`: regression cases for first-turn ambiguity, known symptoms, topic switching and recurrence.

## Verification evidence (not acceptance PASS)

- Exact file update responses returned new commit/blob hashes for each change on `main`.
- The current GitHub history contains the commits and expected order.
- GitHub combined status for `38bd0999c05172908316aa5b493ad367698e0c7f` returned an empty set; associated PR-filtered workflow lookup also returned no run. This is **not** a passing build.
- Local execution and real-browser reproduction **NOT RUN**: original `127.0.0.1:5175` is machine-local and unavailable from this environment.

## Reproduction and required remaining commands

On the authorized Windows clone (no live company writes):
```powershell
git fetch origin
git status --short
git log -1 --format="%H %s" origin/main
python -m unittest tools.test_servy_real_demo_dialogue -v
python -m unittest tools.test_servy_real_demo tools.test_servy_real_demo_local_planner -v
cd frontend
node --test test/customerAnswerContract.test.mjs
npm run lint
npm run build
```
Then run the existing approved launcher only in its authorized read-only environment, inspect browser requests and opened citations against actual pages, and record a fully versioned evaluation. Do not execute migrations, model downloads, reindexing or ticket writes against the existing company data without separate authorization.

## Mandatory blockers

- Real ticket write API, isolated staging database and authorization are not available to this session. The current local demo intentionally confirms **preview only**, with no created record; do not misrepresent it as ticket submission.
- Authorized original documents and operational permissions are not accessible, so citation support, revocation, retrieval performance, parser recovery and tenant isolation cannot be independently verified.
- Full acceptance, browser, adversarial, load, streaming and rollback suites are not executed here. No independent specialist agent invocation or signoff occurred.
- The private production Servy repository is not identified as this prototype. This operation modified only the matching connected `yashaskn8/prototype` repository; no production deployment was performed.

## Deployment

**NOT DEPLOYED.** `http://127.0.0.1:5175/` is the pre-existing host-local demo endpoint, not a remotely accessible verified URL. Production readiness, integration completion and 100% completion are not claimed.
