# project_12 — Protocol 6: Emergent Constraint Landscape

> **quantuminquiry.org — Protocol 6 — March 2026**

## Protocol 6 Lock

| Field | Value |
|-------|-------|
| Preregistration SHA-256 | c286a89037966e56630f5d3ce4cdb4a1621ce7ce9a0f7e7d477a06b8495dffc3 |
| Zenodo DOI | 10.5281/zenodo.19297509 |
| Pilot commit | adca6fc (tag: p6-pilot-pre-run) |

**No confirmatory runs until the Zenodo DOI is recorded here.**

## Protocol 6 Preregistration

SHA-256: c286a89037966e56630f5d3ce4cdb4a1621ce7ce9a0f7e7d477a06b8495dffc3
DOI: 10.5281/zenodo.19297509
File: docs/Protocol6_Preregistration_v3.pdf
Status: Preregistered — awaiting confirmatory execution

## Overview

Protocol 6 extends the Protocol 5 three-agent MARL harness with an emergent
constraint field mechanism. Agents self-assemble a shared constraint landscape
through signal emission. The field diffuses, decays, and modulates signal costs
locally — each agent perceives only its own field value.

## Pilot

The pilot is a parameter sweep (6 combinations × 10 seeds = 60 runs, 200 epochs
each) to identify a viable regime for the confirmatory preregistration.

See `docs/` for the pilot spec and preregistration draft.

## Quick Start

```bash
cd backend
pip install -r requirements.txt
python run_p6_pilot.py
```

## Output

- `confirmatory_run_p6_pilot.log` — per-run epoch series
- `p6_pilot_summary.json` — summary statistics per combination
