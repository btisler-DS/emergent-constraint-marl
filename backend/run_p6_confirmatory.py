"""Protocol 6 Confirmatory Run Script — four-condition harness.

Conditions
----------
A — Emergent local  (constraint field, local perception)
B — Emergent global (constraint field, global perception)
C — Fixed external  (fixed cost multiplier matched to pilot mean field)
D — No constraint   (unconstrained baseline)

Parameters: dc=0.1, dr=0.05, 500 epochs, 50 seeds per condition.
Preregistration DOI: 10.5281/zenodo.19297509

Usage:
  cd backend
  python run_p6_confirmatory.py [--config ../config_p6_confirmatory.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from simulation.p6_confirmatory_engine import P6ConfirmatoryEngine
from simulation.p6_engine import P6Config
from simulation.metrics.field_diagnostics import compute_run_summary


def setup_logging(log_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_one(
    condition: str,
    seed: int,
    base_config: dict,
    fixed_cost_multiplier: float,
) -> dict:
    config = P6Config(
        seed=seed,
        num_epochs=base_config["num_epochs"],
        episodes_per_epoch=base_config["episodes_per_epoch"],
        grid_size=base_config["grid_size"],
        num_obstacles=base_config["num_obstacles"],
        z_layers=base_config["z_layers"],
        max_steps=base_config["max_steps"],
        energy_budget=base_config["energy_budget"],
        move_cost=base_config["move_cost"],
        collision_penalty=base_config["collision_penalty"],
        signal_dim=base_config["signal_dim"],
        hidden_dim=base_config["hidden_dim"],
        depth=base_config["depth"],
        learning_rate=base_config["learning_rate"],
        gamma=base_config["gamma"],
        communication_tax_rate=base_config["communication_tax_rate"],
        survival_bonus=base_config["survival_bonus"],
        declare_cost=base_config["declare_cost"],
        query_cost=base_config["query_cost"],
        respond_cost=base_config["respond_cost"],
        signal_cost_sensitivity=base_config["signal_cost_sensitivity"],
        diffusion_coefficient=base_config["diffusion_coefficient"],
        decay_rate=base_config["decay_rate"],
        device=base_config.get("device", "cpu"),
    )

    engine = P6ConfirmatoryEngine(
        config=config,
        condition=condition,
        fixed_cost_multiplier=fixed_cost_multiplier,
    )
    t0 = time.time()
    epoch_series = engine.run()
    elapsed = time.time() - t0

    run_summary = compute_run_summary(epoch_series)

    entropies = [e["field_entropy"] for e in epoch_series]
    sss_scores = [e["sustained_structure_score"] for e in epoch_series]
    if len(set(entropies)) > 1 and len(set(sss_scores)) > 1:
        corr = float(np.corrcoef(entropies, sss_scores)[0, 1])
    else:
        corr = 0.0

    return {
        "condition": condition,
        "seed": seed,
        "elapsed_s": round(elapsed, 1),
        "field_formed": run_summary["field_formed"],
        "field_saturated": run_summary["field_saturated"],
        "field_collapsed": run_summary["field_collapsed"],
        "entropy_sss_correlation": round(corr, 4),
        "final_field_mean": round(epoch_series[-1]["field_mean"], 6),
        "final_field_std": round(epoch_series[-1]["field_std"], 6),
        "final_avg_reward_A": round(epoch_series[-1]["avg_reward_A"], 6),
        "final_avg_reward_B": round(epoch_series[-1]["avg_reward_B"], 6),
        "final_avg_reward_C": round(epoch_series[-1]["avg_reward_C"], 6),
        "final_query_rate": round(epoch_series[-1]["query_rate"], 6),
        "epoch_series": epoch_series,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol 6 confirmatory four-condition run")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "config_p6_confirmatory.yaml"),
        help="Path to config_p6_confirmatory.yaml",
    )
    parser.add_argument(
        "--condition",
        choices=["A", "B", "C", "D"],
        default=None,
        help="Run a single condition only (for parallel multi-GPU execution). "
             "Omit to run all four conditions sequentially.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root = Path(__file__).parent.parent
    results_dir = root / cfg["output_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # Condition-specific output paths when running single-condition (parallel mode)
    suffix = f"_cond{args.condition}" if args.condition else ""
    log_path = str(results_dir / f"confirmatory_run_p6{suffix}.log")
    json_path = str(root / cfg["output_json"].replace(".json", f"{suffix}.json"))

    setup_logging(log_path)
    logger = logging.getLogger(__name__)

    conditions = [args.condition] if args.condition else cfg["conditions"]
    seeds = cfg["seeds"]
    fixed_multiplier = cfg["fixed_cost_multiplier"]

    logger.info(
        "Config audit | preregistration_doi=10.5281/zenodo.19297509 "
        "dc=%.2f dr=%.2f fixed_cost_multiplier=%.4f pilot_mean_field=%.4f "
        "signal_weights declare=%.3f query=%.3f respond=%.3f",
        cfg["diffusion_coefficient"],
        cfg["decay_rate"],
        fixed_multiplier,
        cfg["pilot_mean_field"],
        cfg["signal_weights"]["declare"],
        cfg["signal_weights"]["query"],
        cfg["signal_weights"]["respond"],
    )
    logger.info(
        "Protocol 6 Confirmatory: %d conditions x %d seeds = %d total runs, %d epochs each",
        len(conditions), len(seeds), len(conditions) * len(seeds), cfg["num_epochs"],
    )

    all_results: list[dict] = []
    run_idx = 0
    total = len(conditions) * len(seeds)

    for condition in conditions:
        condition_results = []
        for seed in seeds:
            run_idx += 1
            logger.info("Run %d/%d | condition=%s seed=%d", run_idx, total, condition, seed)
            result = run_one(condition, seed, cfg, fixed_multiplier)

            run_key = f"cond{condition}_seed{seed}"
            run_path = results_dir / f"p6_confirmatory_{run_key}.json"
            run_record = {k: v for k, v in result.items() if k != "epoch_series"}
            with open(run_path, "w") as rf:
                json.dump(run_record, rf)

            condition_results.append(result)
            all_results.append(result)

        formed = sum(r["field_formed"] for r in condition_results)
        saturated = sum(r["field_saturated"] for r in condition_results)
        collapsed = sum(r["field_collapsed"] for r in condition_results)
        mean_reward = float(np.mean([r["final_avg_reward_A"] for r in condition_results]))
        logger.info(
            "  CONDITION %s | formed=%d/%d saturated=%d/%d collapsed=%d/%d mean_final_reward_A=%.4f",
            condition, formed, len(seeds), saturated, len(seeds), collapsed, len(seeds), mean_reward,
        )

    # Build summary JSON
    condition_summaries = []
    for condition in conditions:
        runs = [r for r in all_results if r["condition"] == condition]
        formed_count = sum(r["field_formed"] for r in runs)
        saturated_count = sum(r["field_saturated"] for r in runs)
        collapsed_count = sum(r["field_collapsed"] for r in runs)
        mean_corr = float(np.mean([r["entropy_sss_correlation"] for r in runs]))
        mean_final_reward_A = float(np.mean([r["final_avg_reward_A"] for r in runs]))
        mean_final_query_rate = float(np.mean([r["final_query_rate"] for r in runs]))
        condition_summaries.append({
            "condition": condition,
            "field_formed_count": formed_count,
            "field_saturated_count": saturated_count,
            "field_collapsed_count": collapsed_count,
            "mean_entropy_sss_correlation": round(mean_corr, 4),
            "mean_final_reward_A": round(mean_final_reward_A, 4),
            "mean_final_query_rate": round(mean_final_query_rate, 4),
            "n_seeds": len(runs),
        })

    summary = {
        "protocol": "P6_confirmatory",
        "preregistration_doi": "10.5281/zenodo.19297509",
        "total_runs": total,
        "conditions": condition_summaries,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary written to %s", json_path)


if __name__ == "__main__":
    main()
