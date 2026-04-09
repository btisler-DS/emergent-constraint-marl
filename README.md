# project_12 — Protocol 6: Emergent Constraint Landscape

> **quantuminquiry.org — Protocol 6 — March 2026**

## Status: Complete

Preregistration locked, pilot complete, confirmatory runs complete (200 seeds),
statistical analysis complete, results paper published.

## Protocol 6 Lock

| Field | Value |
|-------|-------|
| Preregistration SHA-256 | c286a89037966e56630f5d3ce4cdb4a1621ce7ce9a0f7e7d477a06b8495dffc3 |
| Preregistration DOI | [10.5281/zenodo.19297509](https://doi.org/10.5281/zenodo.19297509) |
| Preregistration file | `docs/Protocol6_Preregistration_v3.pdf` |
| Pilot commit | adca6fc (tag: p6-pilot-pre-run) |
| Confirmatory pre-run tag | da5cc47 (tag: p6-confirmatory-pre-run) |
| Confirmatory complete tag | 4a12315 (tag: p6-confirmatory-complete) |
| Results Paper DOI | [10.5281/zenodo.19485185](https://doi.org/10.5281/zenodo.19485185) |

## Overview

Protocol 6 extends the Protocol 5 three-agent MARL harness with an emergent constraint field mechanism. Agents self-assemble a shared constraint landscape through signal emission. The field diffuses, decays, and modulates signal costs locally. Four conditions tested: emergent local perception (A), emergent global perception (B), fixed external constraint matched cost (C), and no constraint (D). 200 confirmatory runs (50 seeds × 4 conditions, 500 epochs each).

Key results: Mechanistic prediction strongly confirmed (median entropy–SSS r = −0.680, p < 0.001). Primary behavioral claim not confirmed (A vs. C, p = 0.069). H3 reversed — global field perception produced more behavioral variance than local, contrary to all committee predictions. Governance conclusion: emergent constraint fields are causally active but do not outperform fixed external rules. Passive emergence is insufficient as a governance strategy.

| Condition | Description |
|-----------|-------------|
| A — Emergent Local | Field active; AgentA observes own field value (obs_dim=13) |
| B — Emergent Global | Field active; AgentA observes full field vector (obs_dim=15) |
| C — Fixed External | No field; fixed communication tax = 1.2287 (matched cost) |
| D — No Constraint | No field; no communication cost; unconstrained baseline |

## Key Results

**3/6 preregistered tests supported at α = 0.05.**

| Test | Result | Statistic |
|------|--------|-----------|
| H1 proxy: field structuring (A vs. C) | Supported | U=50, p<0.001, r=0.96 |
| H2a: reward cost of constraint (A vs. D) | Supported | U=245, p<0.001, r=0.80 |
| H2b: emergent vs. fixed efficiency (A vs. C) | Not supported | p=0.069 |
| H3: local > global field variance | Not supported | reversed: Var(B)>Var(A) |
| Mechanistic: entropy–SSS coupling in A | Strongly supported | W=3, p<0.001, median r=−0.680 |
| H4: behavioral differentiation ordering | Not supported | ordering reversed |

**Primary claim (H1 + H2 both required): not fully confirmed.**
The emergent constraint field is causally active but does not produce significantly
better behavioral outcomes than a fixed external rule of equivalent cost.

## Logged Deviations

1. **Deviation 1** (commit `adca6fc`): Signal weights scaled 0.1× pre-pilot
   (DECLARE 0.3→0.03, QUERY 0.1→0.01, RESPOND 0.2→0.02).
2. **Deviation 2** (post-pilot): Field formation criterion revised from
   `field_std > 0.05` sustained >50 epochs to activity-based (`field_mean > 0.01`
   after epoch 50). Research question revised from spatial differentiation to
   temporal coupling.

## Outputs

| File | Description |
|------|-------------|
| `Protocol6_Results_Paper.docx / .pdf` | Full results paper with AI Use Declaration |
| `p6_pilot_summary.json` | Pilot summary statistics (60 runs) |
| `p6_confirmatory_summary_cond[A-D].json` | Confirmatory summary per condition |
| `analysis/p6_confirmatory_analysis.json` | Full statistical analysis (JSON) |
| `analysis/p6_confirmatory_analysis_report.txt` | Statistical analysis report |
| `docs/P6_Confirmatory_Findings_Report.txt` | Detailed findings report |
| `docs/P6_Prediction_Scoring_Report.txt` | AI committee prediction scoring |
| `results/p6_confirmatory/` | Per-seed JSON results (200 files) |

## Pilot

Parameter sweep: 6 combinations × 10 seeds = 60 runs, 200 epochs each.
Selected regime: dc=0.1, dr=0.05. Results: `p6_pilot_summary.json`.

```bash
cd backend && pip install -r requirements.txt
python run_p6_pilot.py
```

## Confirmatory

200 seeds (50 per condition), 500 epochs each. Results paper: [10.5281/zenodo.19485185](https://doi.org/10.5281/zenodo.19485185).

```bash
# Single condition (parallelisable across GPUs)
python run_p6_confirmatory.py --condition A

# Tests (70 pass)
python -m pytest tests/ -v
```

## Series Context

| Protocol | Finding | Effect |
|----------|---------|--------|
| P2 | Fixed ethical tax → query-flooding attractor | d = −2.18 |
| P3 | Epistemic opacity amplified gaming | d = +2.22 |
| P4 | Self-modeling depth: sacrifice ↑, alignment unchanged | CDI dissociated |
| P5 | Complete null across all five hypotheses | — |
| **P6** | **Emergent field causally active; behavioral null** | **median r = −0.680** |

Prior protocol DOIs: [P2](https://doi.org/10.5281/zenodo.18929040) · [P3](https://doi.org/10.5281/zenodo.19096602) · [P4](https://doi.org/10.5281/zenodo.19005417) · [P5](https://doi.org/10.5281/zenodo.19038790)
