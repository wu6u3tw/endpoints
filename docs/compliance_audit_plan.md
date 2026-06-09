# Compliance Audit Module — Design Plan

Status: **Proposed** · Scope of first cut: **TEST04**, with TEST06/07/09 as planned extensions.

This document plans a compliance/audit module for the endpoint benchmarking tool that
re-implements the _intent_ of the MLPerf Inference compliance ("audit") tests. The
reference implementation lives in the MLCommons inference repo (compliance/nvidia/TESTxx).

---

## 1. Background: what MLPerf audit tests do

MLPerf compliance tests detect that a submitter is not gaming the benchmark (caching,
truncating outputs, running a different/cheaper model in the perf run, EOS exploits).
They are built on three LoadGen-specific pieces:

1. **`audit.config`** — a file LoadGen reads at `StartTest()` that overrides run settings
   to enable the test (e.g. issue duplicate samples, log a sample of outputs, fix seeds).
2. **`mlperf_log_accuracy.json`** — the SUT logs raw **output token IDs** during the run.
3. **`run_verification.py`** — a post-run script that consumes the logs and emits
   `verify_*.txt` with a `Performance check pass: True/False` / `TEST PASS` line.

### Test matrix (LLM-relevant subset)

| Test   | Detects                                             | Category      | Required for                         |
| ------ | --------------------------------------------------- | ------------- | ------------------------------------ |
| TEST01 | Different model in perf vs accuracy run             | orchestrator  | ResNet50, BERT, SDXL, RetinaNet, …   |
| TEST04 | Caching of duplicate queries (throughput inflation) | orchestrator  | ResNet50, SDXL (LLMs largely exempt) |
| TEST06 | LLM output consistency (EOS / first-token / length) | analyzer      | llama2/3.1, mixtral, deepseek        |
| TEST07 | Accuracy ≥ threshold in perf mode                   | analyzer      | gpt-oss-120b                         |
| TEST09 | Mean output token length within ±10% of reference   | analyzer      | gpt-oss-120b                         |
| TEST08 | DLRM-v3 streaming accuracy                          | n/a (not LLM) | DLRM-v3 — **out of scope**           |

**TEST04 (mechanism).** `audit.config` sets `performance_issue_same=1` /
`performance_issue_same_index=3` so LoadGen issues the **same sample repeatedly**, then
the verification compares throughput against the standard perf run. Pass if the audit run
is **not more than 10% faster** than baseline (20% for low-throughput streams). If the SUT
caches responses for duplicate queries, throughput inflates → FAIL.

> **LLM nuance.** MLPerf exempts variable-length-input LLMs from TEST04 because prefix
> caching legitimately speeds up identical prompts. On an LLM endpoint, TEST04 will see
> real prefix-cache gains; the tolerance (and whether the audit run disables prefix cache)
> is a deliberate knob. We build it faithfully to the reference (±10% / ±20%) and expose
> the tolerance.

---

## 2. Conceptual mapping: MLPerf → this repo

This tool is its own HTTP load generator (no LoadGen). The audit module re-implements the
_intent_ over this repo's own artifacts.

| MLPerf                                 | This repo                                                      |
| -------------------------------------- | -------------------------------------------------------------- |
| `audit.config` (run-setting override)  | an **audit profile**: a `BenchmarkConfig` transform            |
| `mlperf_log_accuracy.json` (token IDs) | `events.jsonl` (must carry token IDs for token-level tests)    |
| `run_verification.py` → `verify_*.txt` | an **`AuditTest`** → `verify_<TEST>.txt` + `audit_report.json` |
| compliance submission dir layout       | mirrored under the run's report dir                            |

### Feasibility note for the token-level tests (TEST06/09)

A finished run captures decoded **text** + `finish_reason` for all adapters, but **raw
output token IDs only for the SGLang adapter** (`QueryResult.metadata["token_ids"]`).
OpenAI/completions runs lose the token-ID stream. TEST06's EOS/first-token checks and exact
TEST09 need token IDs, so faithful TEST06/09 will require a small, localized data-path
addition (capture token IDs under an audit-capture flag when the server can return them —
logprobs / SGLang `token_delta`). **This only matters when TEST06/09 are implemented;**
TEST04 (throughput-only) needs none of it.

---

## 3. Architecture

A single `AuditTest` abstraction covers **both** categories — orchestrators (must execute a
specially-configured run) and analyzers (pure post-run). It mirrors the existing `Scorer`
registry idiom (`PREDEFINED` dict, `*_ID` classvar, `__init_subclass__` auto-registration)
so adding a test is one file + one registry entry, not a new framework.

```python
class AuditTest(ABC):
    PREDEFINED: ClassVar[dict[str, type[AuditTest]]] = {}   # auto-register via __init_subclass__
    TEST_ID: ClassVar[str]                                   # "TEST04"
    REQUIRES_EXECUTION: ClassVar[bool]                       # True = orchestrator (04/01)
    REQUIRES_BASELINE: ClassVar[bool]                        # needs a reference run

    def build_audit_config(self, base: BenchmarkConfig) -> BenchmarkConfig:
        """audit.config analogue. Default identity — analyzers don't change the run."""
        return base

    @abstractmethod
    def verify(self, audit: RunArtifacts, baseline: RunArtifacts | None) -> AuditResult:
        ...
```

- **Orchestrator (TEST04, TEST01):** override `build_audit_config`, execute via the
  existing benchmark path, then `verify` against a baseline.
- **Analyzer (TEST06, TEST09, TEST07):** leave `build_audit_config` identity; `verify`
  reads a single finished run.

### TEST04 integration seam (validated)

`load_generator/sample_order.py` exposes a `SampleOrder` ABC and a single factory
`create_sample_order(settings)`. TEST04's "issue the same sample repeatedly" is a new
`FixedIndexSampleOrder` subclass selected by one settings field — the direct analogue of
`performance_issue_same_index`. Surgical, not a hack.

---

## 4. File-by-file changes

| File                                                                  | Change                                                                                                                                                             |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/inference_endpoint/compliance/audit_test.py`                     | **new** — `AuditTest` ABC, `PREDEFINED` registry, `AuditResult` / `CheckOutcome`                                                                                   |
| `src/inference_endpoint/compliance/artifacts.py`                      | **new** — `RunArtifacts`: load `config.yaml` + `final_snapshot.json` (→ `Report`) + `events.jsonl` from a report dir; reuse `evaluation/scoring.py`'s event reader |
| `src/inference_endpoint/compliance/comparison.py`                     | **new** — `within_tolerance(test_qps, ref_qps, low_throughput)` (±10% / ±20%), shared by TEST04 / future TEST01                                                    |
| `src/inference_endpoint/compliance/report.py`                         | **new** — `AuditReport` → `verify_TEST04.txt` (reference wording) + `audit_report.json` in canonical `TEST04/performance/run_1/` layout                            |
| `src/inference_endpoint/compliance/tests/test04.py`                   | **new** — `NoCachingAuditTest`: `build_audit_config` selects fixed-index sampling; `verify` = `audit_qps ≤ baseline_qps × (1 + tol)`                               |
| `src/inference_endpoint/load_generator/sample_order.py`               | **+** `FixedIndexSampleOrder`; `create_sample_order` returns it when the audit setting is set                                                                      |
| `src/inference_endpoint/config/runtime_settings.py` (+ audit profile) | **+** one field (e.g. `fixed_sample_index: int \| None`) the profile sets                                                                                          |
| `src/inference_endpoint/commands/audit.py`                            | **new** — `AuditConfig` + `execute_audit`; resolve test from registry, orchestrate run (reuse `benchmark/execute.py`) if needed, then verify                       |
| `src/inference_endpoint/main.py`                                      | **+** register `audit` subcommand (one line, like `probe`)                                                                                                         |

### CLI shape

One `audit` command. `AuditConfig` (Pydantic, cyclopts-annotated) carries `test`,
`config`, `report_dir`, `baseline_report`; a model-validator enforces the right combination
per test category.

```
inference-endpoint audit --test TEST04 --config bench.yaml --baseline-report <dir>
```

**Baseline handling:** default to `--baseline-report` pointing at an already-finished
standard benchmark (the submitter has one — don't re-run). If omitted, `--run-baseline`
executes the base config first.

---

## 5. Module layout

```
src/inference_endpoint/compliance/
├── __init__.py
├── audit_test.py      # AuditTest ABC + PREDEFINED registry + AuditResult/CheckOutcome
├── artifacts.py       # RunArtifacts loader over a report dir
├── comparison.py      # throughput-within-tolerance helper
├── report.py          # AuditReport → verify_<TEST>.txt + audit_report.json
└── tests/
    ├── __init__.py    # imports submodules so __init_subclass__ registration fires
    └── test04.py      # NoCachingAuditTest
```

---

## 6. Success criteria (goal-driven; verify before done)

1. **Unit** — `FixedIndexSampleOrder` always yields the configured index.
2. **Unit** — `verify` PASS at `audit_qps ≤ ref × 1.10`, FAIL above; boundary tests at
   ±10% / ±20%.
3. **Integration** — against `mock_http_echo_server` (no caching): baseline + TEST04 →
   `audit_qps ≈ baseline_qps` → **PASS**; plus a caching mock asserting **FAIL**.
4. **Cross-check** — `verify_TEST04.txt` wording / PASS-FAIL line matches the reference
   `compliance/nvidia/TEST04/verify_performance.py` so an MLPerf submission checker accepts it.
5. `pre-commit run --all-files` clean (ruff / mypy / license headers).

---

## 7. Extension path (TEST06 / TEST07 / TEST09)

- **TEST06** (analyzer): `REQUIRES_EXECUTION=False`. Sub-checks: EOS-token, first-token
  equality, sample-length == token-count. Requires the token-ID capture addition (§2).
- **TEST09** (analyzer): mean OSL within `[min, max]` from the ruleset. Exact form needs
  token IDs; an approximate form can reuse the existing re-tokenized OSL series.
- **TEST07** (analyzer): reuse the existing `Scorer` pipeline, compare score to a threshold.
- **Systematic selection (optional):** add `required_audit_tests` per model in
  `config/rulesets/mlcommons/` so `audit` can look up which tests a model requires.

Each lands as one file under `compliance/tests/` + a registry entry — no changes to the
core abstraction.
