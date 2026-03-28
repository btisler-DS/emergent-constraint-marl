# project_12 — Protocol 6: Emergent Constraint Landscape

> **quantuminquiry.org — Protocol 6 — March 2026**

## Protocol 6 Lock

| Field | Value |
|-------|-------|
| Preregistration SHA-256 | *(to be filled after Zenodo submission)* |
| Zenodo DOI | *(to be filled after Zenodo submission)* |
| Pilot commit | *(to be filled after p6-pilot-pre-run tag)* |

**No confirmatory runs until the Zenodo DOI is recorded here.**

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
